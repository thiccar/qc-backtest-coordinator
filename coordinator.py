from collections import Counter
import copy
import csv
from datetime import datetime
import functools
import json
import locale
import logging
import math
from pathlib import Path
from time import sleep

from api.ratelimitedapi import RateLimitedApi
from testsets import Test, TestState

locale.setlocale(locale.LC_ALL, 'en_US.UTF8')


class Coordinator:
    logger = logging.getLogger(__name__)

    def __init__(self, test_set, api: RateLimitedApi, config: dict, data_dir: Path):
        self.test_set = test_set
        self.api = api
        self.config = config
        self.data_dir = data_dir

        self.project = None
        self.project_id = None
        self.test_set_path = None
        self.state_path = None
        self.generator_done = False
        self.tests = []  # Stores every test produced by the generator. Gets backed up to disk so we don't lose anything
        self.backtests = []  # List of backtests returned from QC API
        self.state_counter = Counter()

    def initialize(self):
        project_name = self.config["project_name"]
        self.project = self.api.get_project_by_name(project_name)
        assert self.project is not None
        self.logger.debug(self.project)

        self.project_id = self.project["projectId"]
        self.test_set_path = self.data_dir / f"{project_name}-{self.project_id}" / self.test_set.name()
        if not self.test_set_path.exists():
            self.test_set_path.mkdir(parents=True)

        self.logger.addHandler(logging.FileHandler(self.test_set_path / "log.txt"))
        self.state_path = self.test_set_path / "state.json"
            
        self.load_state()
    
    def update_test_state_from_api(self):
        """TODO: What happens if the list of tests gets really big? Is there an upper bound on what QC will allow?
        We may end up needing to only maintain a window of most recent tests on QC with the rest of the state saved
        locally only.
        """
        if not self.tests:
            return
        list_backtests_resp = self.api.list_backtests(self.project_id)
        assert list_backtests_resp["success"]
        self.backtests = list_backtests_resp["backtests"]

        self.state_counter.clear()
        for test in self.tests:
            existing_bt = next((bt for bt in self.backtests if bt["name"] == test.name), None)
            if existing_bt is None:
                if test.state != TestState.CREATED:
                    self.logger.warning(f"Inconsistent state, thought {test.name} was launched, was not")
                test.state = TestState.CREATED
            else:
                if existing_bt["status"] == "Runtime Error":
                    # Assuming that if there was a runtime error, there are bugs in the backtest code that need to be
                    # fixed.  This may necessitate re-running all the backtests.  Maybe if needed can be an option to
                    # delete all the backtests (or erroneous backtests) for a test set during initialization.
                    self.logger.error("Test runtime error")
                    self.logger.error(test.to_dict())
                    self.logger.error(existing_bt["error"])
                    assert False
                elif test.backtest_id != existing_bt["backtestId"]:
                    self.logger.error("Possible duplicate")
                    self.logger.error(f"{test.name} backtest_id: {test.backtest_id} != QC backtestId: {existing_bt['backtestId']}")
                    assert False
                elif existing_bt["completed"]:
                    self.on_test_completed(test)
                else:
                    test.state = TestState.RUNNING
            self.state_counter[test.state] += 1

    def run(self):
        self.initialize()
        concurrency = self.config["concurrency"]
        test_generator = self.test_set.permutations()
        try:
            while True:
                self.update_test_state_from_api()
                self.logger.info(f"generator_done={self.generator_done} len(tests)={len(self.tests)} " +
                                 f"len(backtests)={len(self.backtests)} state_counter={self.state_counter}")
                if self.generator_done and self.state_counter[TestState.CREATED] == 0 and self.state_counter[TestState.RUNNING] == 0:
                    break

                limit = concurrency - self.state_counter[TestState.RUNNING]
                launched = 0
                self.logger.debug(f"Launching up to {limit} new tests")
                while launched < limit:
                    test = self.get_next_test(test_generator)
                    if test is None:
                        self.logger.debug("generator done")
                        self.generator_done = True
                        break

                    # Somehow a backtest was launched for the test, despite the test not being in our internal state.
                    # Reuse the backtest
                    existing_bt = next((bt for bt in self.backtests if bt["name"] == test.name), None)
                    if existing_bt:
                        self.logger.warning(f"{test.name} was previously launched but not recorded")
                        test.backtest_id = existing_bt["backtestId"]
                        test.state = TestState.RUNNING
                    else:
                        self.launch_test(test)
                        launched += 1
                        
                    self.save_state()
                
                if launched > 0 or self.state_counter[TestState.RUNNING] > 0:
                    self.logger.info("Sleeping for 15 secs")
                    sleep(15)

            self.logger.debug("generating report")
            self.generate_csv_report()
        except Exception as exc:
            self.logger.error("Unhandled error", exc_info=exc)

    def get_next_test(self, generator):
        test = next((t for t in self.tests if t.state == TestState.CREATED), None)
        if test:
            self.logger.debug(f"Returning queued test {test.name}")
            return test
        
        test = next(generator, None)
        if not test:
            return None

        # Tests have deterministic naming, so the generator may have emitted this test previously
        existing_test = next((t for t in self.tests if t.name == test.name), None)
        if existing_test:
            self.logger.warning(f"{test.name} is a duplicate")
            return self.get_next_test(generator)
        else:
            self.logger.debug(f"New generated test {test.name}")
            self.tests.append(test)
            return test

    def launch_test(self, test):
        """update parameters file, compile, and launch backtest"""
        if not self.api.update_parameters_file(self.project_id, test.params):
            self.logger.error(f"{test.name} update_parameters_file failed")
            return

        compile_id = self.api.compile(self.project_id)
        if not compile_id:
            self.logger.error(f"{test.name} compile failed")
            return
        
        create_backtest_resp = self.api.create_backtest(self.project_id, compile_id, test.name)
        if not create_backtest_resp["success"]:
            self.logger.error(f"{test.name} create_backtest failed")
            self.logger.error(create_backtest_resp)
        else:
            test.backtest_id = create_backtest_resp["backtest"]["backtestId"]
            test.state = TestState.RUNNING
            self.logger.debug(f"{test.name} launched")

    def on_test_completed(self, test):
        """Download results for completed test and save them, also mark test state as completed.
        In future may pass this to test set to help it initialize generator state
        """
        results_path = self.get_results_path(test)
        if results_path.exists():
            test.state = TestState.COMPLETED
            if not self.read_and_validate_backtest_results(test):
                self.logger.error(f"{test.name} stored results fail validation, consider re-doing")
        else:
            self.logger.debug(f"{test.name} completed, downloading results")
            read_backtest_resp = self.api.read_backtest(self.project_id, test.backtest_id)
            
            # If read_backtest request fails, or downloaded results fail validation, don't do anything. We'll try again later.
            if read_backtest_resp["success"]:
                results = read_backtest_resp["backtest"]
                if self.validate_backtest_results(results):
                    test.state = TestState.COMPLETED
                    self.write_results(test, results)
                else:
                    self.logger.error(f"{test.name} api results fail validation, not storing")
    
    def write_results(self, test, backtest_results):
        results_path = self.get_results_path(test)
        with results_path.open('w') as f:
            gus = {"test": test.to_dict(), "backtest": backtest_results}
            json.dump(gus, f, indent=4)

    def read_and_validate_backtest_results(self, test):
        results_path = self.get_results_path(test)
        try:
            with results_path.open() as f:
                stored = json.load(f)
                if self.validate_backtest_results(stored["backtest"]):
                    return stored
                return None
        except json.decoder.JSONDecodeError:
            self.logger.error(f"{test.name} results json decode failed")
            # TODO: Delete file so it will be re-downloaded?
            raise
    
    def validate_backtest_results(self, results):
        required_keys = ["alphaRuntimeStatistics", "runtimeStatistics", "rollingWindow", "statistics", "totalPerformance"]
        return all((key in results and results[key]) for key in required_keys)  # check key presence and value not null or []

    def get_results_path(self, test):
        return self.test_set_path / f"{test.name}.json"

    # TODO maybe put this in a separate module
    def generate_csv_report(self):
        report_path = self.test_set_path / "report.csv"
        rows = []
        for test in self.tests:
            results = self.read_and_validate_backtest_results(test)
            if not results:
                self.logger.warning(f"{test.name} invalid results, not including in report")
                continue
            
            statistics = results["backtest"]["statistics"]
            statistics["PROE"] = self.proe(results["backtest"])
            rows.append((test.name, copy.deepcopy(test.params), statistics))
        
        params_keys = functools.reduce(lambda s1, s2: s1 | s2, (set(r[1].keys()) for r in rows))
        results_keys = functools.reduce(lambda s1, s2: s1 | s2, (set(r[2].keys()) for r in rows))
        field_names = ["name"] + list(params_keys) + list(results_keys)
        with report_path.open("w", newline="") as f:  # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
            writer = csv.DictWriter(f, field_names)
            writer.writeheader()
            for r in rows:
                d = dict([("name", r[0])] + list(r[1].items()) + list(r[2].items()))
                writer.writerow(d)

    def proe(self, bt_results):
        """Pessimistic return on equity. Variant of PROM (Pessimistic Return on Margin) from Pardo's book "The
        Evaluation and Optimization of Trading Strategies" (chp 9). Make gross profit more pessimistic by reducing
        number of winning trades by square root and increasing number of losing trades by square root. This adjusted
        gross profit is then used to compute an annualized return on initial account equity.
        """
        rs = bt_results["runtimeStatistics"]
        final_equity = locale.atof(rs["Equity"].strip("$"))
        retrn = locale.atof(rs["Return"].strip("%")) / 100
        initial_equity = final_equity / (1 + retrn)
        
        ts = bt_results["totalPerformance"]["TradeStatistics"]
        num_winning = ts["NumberOfWinningTrades"]
        num_losing = ts["NumberOfLosingTrades"]
        avg_win = float(ts["AverageProfit"])
        avg_loss = float(ts["AverageLoss"])
        adj_gross_profit = avg_win * (num_winning - math.sqrt(num_winning)) + avg_loss * (num_losing + math.sqrt(num_losing))

        adj_total_return = adj_gross_profit / initial_equity
        bt_start = datetime.fromisoformat(bt_results["backtestStart"])
        bt_end = datetime.fromisoformat(bt_results["backtestEnd"])
        bt_months = (bt_end.year - bt_start.year) * 12 + (bt_end.month - bt_start.month)  # https://www.kite.com/python/answers/how-to-get-the-number-of-months-between-two-dates-in-python
        adj_annualized_return = ((1 + adj_total_return)**(12/bt_months)) - 1  # https://s3.amazonaws.com/assets.datacamp.com/production/course_18408/slides/chapter2.pdf

        return adj_annualized_return

    def load_state(self):
        """Load state from JSON file, if it exists. Restores state of this object and of test_set"""
        if self.state_path.exists():
            with self.state_path.open() as f:
                state = json.load(f)
                tests = state["coordinator"]["tests"]
                self.tests = [Test.from_dict(d) for d in tests]

    def save_state(self):
        tests = [test.to_dict() for test in self.tests]
        state = {
            "coordinator": {
                "tests": tests,
            },
            "test_set": {}  # TODO if test set is stateful can store it here
        }
        # We assume here that the serialized state is always growing, so there is no risk of writing
        # new file with fewer bytes than previous
        with self.state_path.open('w') as f:
            json.dump(state, f, indent=4)
