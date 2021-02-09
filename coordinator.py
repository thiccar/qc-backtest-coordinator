from collections import Counter
import csv
import functools
import json
import logging
from pathlib import Path
from time import sleep

import dictdiffer

from api.ratelimitedapi import RateLimitedApi
from testsets import Test, TestResult, TestResultValidationException, TestSet, TestState


class CoordinatorIO:
    """Methods for reading and writing state files and test results"""

    logger = logging.getLogger(__name__)

    def __init__(self, test_set_path, read_only=True, mkdir=False):
        self.test_set_path = test_set_path
        if not read_only and mkdir and not self.test_set_path.exists():
            self.test_set_path.mkdir(parents=True)
        self.read_only = read_only
        self.state_path = self.test_set_path / "state.json"
        self.report_path = self.test_set_path / "report.csv"
        self.log_path = self.test_set_path / "log.txt"

    def read_test_result(self, test) -> TestResult:
        result_path = self.test_result_path(test)
        try:
            if result_path.exists():
                with result_path.open() as f:
                    stored = json.load(f)
                    result = TestResult.from_dict(stored)
                    test.state = TestState.COMPLETED
                    self.validate_backtest_consistency(test, result)
                    result.test = test  # Re-use the passed in object
                    return result
            return None
        except json.decoder.JSONDecodeError:
            self.logger.error(f"{test.name} result json decode failed")
            raise  # Something has gone very wrong, fail loud and fast

    @classmethod
    def validate_backtest_consistency(cls, test: Test, result: TestResult):
        td = test.to_dict()
        rd = result.test.to_dict()
        if td != rd:
            cls.logger.error(f"Inconsistent test & result: {td} {rd}")
            cls.logger.error(list(dictdiffer.diff(td, rd)))

    def write_test_result(self, result: TestResult):
        assert not self.read_only
        result.test.state = TestState.COMPLETED
        result_path = self.test_result_path(result.test)
        with result_path.open('w') as f:
            json.dump(result.to_dict(), f, indent=4)

    def test_result_exists(self, test):
        result_path = self.test_result_path(test)
        return result_path.exists()

    def test_result_path(self, test):
        name = test.name if isinstance(test, Test) else test["name"]  # Support Test or dict
        return self.test_set_path / f"{name}.json"

    def read_test_log(self, test):
        log_path = self.test_log_path(test)
        with log_path.open() as f:
            return f.read()

    def write_test_log(self, test, log):
        log_path = self.test_log_path(test)
        with log_path.open("w") as f:
            for line in log:
                f.write(line)
                f.write("\n")

    def test_log_path(self, test):
        name = test.name if isinstance(test, Test) else test["name"]  # Support Test or dict
        return self.test_set_path / f"{name}.log"

    def read_state(self):
        """Read state from JSON file, if it exists"""
        if self.state_path.exists():
            with self.state_path.open() as f:
                return json.load(f)

    def write_state(self, state):
        """Serialize given object to JSON and write to state file"""
        assert not self.read_only
        # We assume here that the serialized state is always growing, so there is no risk of writing
        # new file with fewer bytes than previous
        with self.state_path.open('w') as f:
            json.dump(state, f, indent=4)


class Coordinator:
    logger = logging.getLogger(__name__)

    def __init__(self, test_set: TestSet, api: RateLimitedApi, config: dict, data_dir: Path):
        self.test_set = test_set
        self.api = api
        self.config = config
        self.data_dir = data_dir

        self.cio = None
        self.project = None
        self.project_id = None
        self.generator_done = False
        self.generated_cnt = 0
        self.tests = []  # Stores every test produced by the generator. Gets backed up to disk so we don't lose anything
        self.backtests = []  # List of backtests returned from QC API
        self.state_counter = Counter()

    def initialize(self):
        project_name = self.config["project_name"]
        self.project = self.api.get_project_by_name(project_name)
        assert self.project is not None
        self.logger.info(self.project)

        self.project_id = self.project["projectId"]
        self.cio = CoordinatorIO(self.data_dir / f"{project_name}-{self.project_id}" / self.test_set.name(),
                                 read_only=False, mkdir=True)

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.cio.log_path)
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        root_logger.addHandler(file_handler)
            
        self.load_state()
    
    def update_tests_state_from_api(self):
        """TODO: What happens if the list of tests gets really big? Is there an upper bound on what QC will allow?
        We may end up needing to only maintain a window of most recent tests on QC with the rest of the state saved
        locally only.
        """
        list_backtests_resp = self.api.list_backtests(self.project_id)
        if list_backtests_resp["success"]:
            self.backtests = list_backtests_resp["backtests"]
            self.update_tests_state()
        return list_backtests_resp["success"]

    def update_tests_state(self):
        self.state_counter.clear()
        for test in self.tests:
            self.update_test_state(test)
            self.state_counter[test.state] += 1

    def update_test_state(self, test):
        existing_bt = next((bt for bt in self.backtests if bt["name"] == test.name), None)
        if existing_bt is None:
            if test.state != TestState.CREATED:
                self.logger.warning(f"Inconsistent state, thought {test.name} was launched, was not")
            test.state = TestState.CREATED
        else:
            # Handle error and inconsistent state cases
            if existing_bt["status"] == "Runtime Error":
                # Assuming that if there was a runtime error, there are bugs in the backtest code that need to be
                # fixed.  This may necessitate re-running all the backtests.  Maybe if needed can be an option to
                # delete all the backtests (or erroneous backtests) for a test set during initialization.
                self.logger.error("Test runtime error")
                self.logger.error(test.to_dict())
                self.logger.error(existing_bt["error"])
                assert False
            elif test.backtest_id and test.backtest_id != existing_bt["backtestId"]:
                self.logger.error("Possible duplicate")
                self.logger.error(f"{test.name} backtest_id: {test.backtest_id} != "
                                  f"QC backtestId: {existing_bt['backtestId']}")
                assert False
            else:
                test.backtest_id = existing_bt["backtestId"]
                if existing_bt["completed"]:
                    self.on_test_completed(test)
                else:
                    test.state = TestState.RUNNING

    def run(self):
        self.initialize()
        concurrency = self.config["concurrency"]
        test_generator = self.test_set.tests()
        try:
            while True:
                if not self.update_tests_state_from_api():
                    self.logger.error("Error updating tests state from API")
                    sleep(15)
                    continue

                self.logger.info(f"generator_done={self.generator_done} generated_cnt={self.generated_cnt} "
                                 f"len(tests)={len(self.tests)} "
                                 f"len(backtests)={len(self.backtests)} state_counter={self.state_counter}")
                if (self.generator_done and self.state_counter[TestState.CREATED] == 0
                        and self.state_counter[TestState.RUNNING] == 0):
                    break

                limit = concurrency - self.state_counter[TestState.RUNNING]
                launched = 0
                self.logger.debug(f"Launching up to {limit} new tests")
                while launched < limit and not self.generator_done:
                    test = self.get_next_test(test_generator)
                    if test is None:
                        self.logger.info("generator done")
                        self.generator_done = True
                        break
                    if test == TestSet.NO_OP:
                        self.logger.info("no-op test")
                        break

                    self.update_test_state(test)
                    if test.state != TestState.CREATED:
                        self.logger.info(f"{test.name} was previously launched")
                    else:
                        self.launch_test(test)
                        launched += 1
                        
                    self.save_state()
                
                if launched > 0 or self.state_counter[TestState.RUNNING] > 0 or test == TestSet.NO_OP:
                    self.logger.info("Sleeping for 15 secs")
                    sleep(15)

            self.save_state()
            self.logger.info("generating report")
            self.generate_csv_report()
        except Exception as exc:
            self.logger.error("Unhandled error", exc_info=exc)

    def get_next_test(self, generator):
        test = next((t for t in self.tests if t.state == TestState.CREATED), None)
        if test:
            self.logger.info(f"Returning queued test {test.name}")
            return test
        
        test = next(generator, None)
        if not test or test == TestSet.NO_OP:
            return test

        self.generated_cnt += 1
        # Tests have deterministic naming, so the generator may have emitted this test previously
        existing_test = next((t for t in self.tests if t.name == test.name), None)
        if existing_test:
            self.logger.info(f"{test.name} is a duplicate")
        else:
            self.logger.info(f"New generated test {test.name}")
            self.tests.append(test)
        return test

    def launch_test(self, test):
        """update parameters file, compile, and launch backtest"""
        if not self.api.update_parameters_file(self.project_id, test.params, test.extraneous_params):
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
            self.logger.info(f"{test.name} launched")

    def on_test_completed(self, test: Test):
        if not test.result_saved:
            self.save_test_result(test)

        if not test.log_saved:
            self.save_test_log(test)

    def save_test_result(self, test):
        """Download result for completed test and save it, also mark test state as completed."""
        try:
            result = self.cio.read_test_result(test)  # See if already stored result locally
            if result:
                self.logger.debug(f"{test.name} completed and result already downloaded")
            else:
                self.logger.info(f"{test.name} completed, downloading result, attempt={test.read_backtest_attempts}")
                read_backtest_resp = self.api.read_backtest(self.project_id, test.backtest_id)

                # If read_backtest request fails, or downloaded result fail validation, don't do anything.
                # We'll try again later.
                if read_backtest_resp["success"]:
                    result = TestResult(test, read_backtest_resp["backtest"])
                    self.cio.write_test_result(result)
        except TestResultValidationException:
            # Sometimes results aren't available right away, so we have retries here
            test.read_backtest_attempts += 1
            self.logger.error(f"{test.name} api result fail validation, attempts={test.read_backtest_attempts}")
            if test.read_backtest_attempts >= 10:
                # Assume by this point that it will never work, rename it and set state to created so it will be
                # launched in next loop iteration
                self.logger.error(f"giving up on read_backtest for {test.name}, will relaunch it")
                update_backtest_resp = self.api.update_backtest(self.project_id, test.backtest_id, f"(FAILED) {test.name}")
                if update_backtest_resp["success"]:
                    test.backtest_id = None
                    test.state = TestState.CREATED
                    test.read_backtest_attempts = 0
            result = None

        if result:
            self.test_set.on_test_completed(result)
            test.result_saved = True

    def save_test_log(self, test):
        read_log_resp = self.api.read_backtest_log(self.project_id, test.backtest_id)
        if read_log_resp["success"]:
            self.cio.write_test_log(test, read_log_resp["BacktestLogs"])
            test.log_saved = True

    # TODO maybe put this in a separate module
    def generate_csv_report(self):
        rows = []
        for test in self.tests:
            result = self.cio.read_test_result(test)

            statistics = result.bt_result["statistics"]
            for k in ["SortinoRatio", "ReturnOverMaxDrawdown"]:
                statistics[k] = result.bt_result["alphaRuntimeStatistics"][k]
            statistics["PROE"] = result.proe()
            rows.append((test, statistics))
        
        params_keys = functools.reduce(lambda s1, s2: s1 | s2,
                                       (set(t.params.keys()) for (t, _) in rows if isinstance(t.params, dict)))
        result_keys = functools.reduce(lambda s1, s2: s1 | s2, (set(s.keys()) for (_, s) in rows))
        field_names = ["name"] + list(params_keys) + list(result_keys)
        # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
        with self.cio.report_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, field_names)
            writer.writeheader()
            for (t, s) in rows:
                d = dict([("name", t.name)] +
                         (list(t.params.items()) if isinstance(t.params, dict) else []) +
                         list(s.items()))
                writer.writerow(d)

    def load_state(self):
        """Load state from JSON file, if it exists. Restores state of this object and of test_set"""
        state = self.cio.read_state()
        if state:
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
        self.cio.write_state(state)
