import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time

from ratelimiter import RateLimiter

from ratelimitedapi import RateLimitedApi
from coordinator_io import CoordinatorIO
from testsets import Test, TestResult, TestResultValidationException, TestSet, TestState


class Coordinator:
    logger = logging.getLogger(__name__)

    def __init__(self, test_set: TestSet, api: RateLimitedApi, data_dir: Path, project_name: str, concurrency=1):
        self.test_set = test_set
        self.api = api
        self.data_dir = data_dir
        self.project_name = project_name
        self.concurrency = concurrency

        self.cio = None
        self.project = None
        self.project_id = None
        self.initial_parameters_file = None
        self.generator_done = False
        self.generated_cnt = 0
        self.tests = []  # Stores every test produced by the generator. Gets backed up to disk so we don't lose anything
        self.backtests = []  # List of backtests returned from QC API
        self.backtests_by_name = {}  # Helper data structure to make lookup by name faster
        self.state_counter = Counter()

        self.io_tpe = ThreadPoolExecutor(thread_name_prefix="io_tpe")
        self.compile_lock = asyncio.Lock()
        self.launch_rate_limiter = RateLimiter(6, 60, self.on_launch_rate_limited)

    def initialize(self):
        self.project = self.api.get_project_by_name(self.project_name)
        assert self.project is not None
        self.logger.info(self.project)

        self.project_id = self.project["projectId"]
        self.cio = CoordinatorIO(self.data_dir / f"{self.project_name}-{self.project_id}" / self.test_set.name(),
                                 read_only=False, mkdir=True)

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.cio.log_path)
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        root_logger.addHandler(file_handler)

        self.initial_parameters_file = self.api.read_parameters_file(self.project_id)
        assert self.initial_parameters_file
            
        self.load_state()
    
    def update_tests_state_from_api(self):
        """TODO: What happens if the list of tests gets really big? Is there an upper bound on what QC will allow?
        We may end up needing to only maintain a window of most recent tests on QC with the rest of the state saved
        locally only.
        """
        list_backtests_resp = self.api.list_backtests(self.project_id)
        if list_backtests_resp["success"]:
            self.backtests = list_backtests_resp["backtests"]
            self.backtests_by_name = {bt["name"]: bt for bt in self.backtests}
            self.update_tests_state()
        return list_backtests_resp["success"]

    def update_tests_state(self):
        self.state_counter.clear()
        for test in self.tests:
            self.update_test_state(test)
            self.state_counter[test.state] += 1

    def update_test_state(self, test):
        if test.name in self.backtests_by_name:
            existing_bt = self.backtests_by_name[test.name]
            # Handle error and inconsistent state cases
            if existing_bt["status"] == "Runtime Error":
                # Assuming that if there was a runtime error, there are bugs in the backtest code that need to be
                # fixed.  This may necessitate re-running all the backtests.  Maybe if needed can be an option to
                # delete all the backtests (or erroneous backtests) for a test set during initialization.
                self.logger.error("Test runtime error")
                self.logger.error(test.to_dict())
                self.logger.error(existing_bt["error"])
                assert False
            else:
                # Sometimes a backtest gets launched but isn't returned in the list of backtests, so it gets launched
                # again.
                if test.backtest_id and test.backtest_id != existing_bt["backtestId"]:
                    self.logger.error(f"Possible duplicate {test.name} backtest_id: {test.backtest_id} != "
                                      f"QC backtestId: {existing_bt['backtestId']}")

                test.backtest_id = existing_bt["backtestId"]
                if existing_bt["completed"]:
                    test.state = TestState.COMPLETED
                else:
                    test.state = TestState.RUNNING
        else:
            if test.state != TestState.CREATED:
                self.logger.warning(f"Inconsistent state, thought {test.name} was launched, was not")
            test.state = TestState.CREATED

    async def run(self):
        self.initialize()
        test_generator = self.test_set.tests()

        while True:
            while not self.update_tests_state_from_api():
                self.logger.error("Error updating tests state from API")
                await asyncio.sleep(15)

            self.logger.info(f"generator_done={self.generator_done} generated_cnt={self.generated_cnt} "
                             f"len(tests)={len(self.tests)} "
                             f"len(backtests)={len(self.backtests)} state_counter={self.state_counter}")
            if (self.generator_done and self.state_counter[TestState.CREATED] == 0
                    and self.state_counter[TestState.RUNNING] == 0
                    and all(test.result_saved and test.log_saved for test in self.tests)):
                break

            on_completed_tasks = []
            for test in self.tests:
                if test.state == TestState.COMPLETED and not (test.result_saved and test.log_saved):
                    on_completed_tasks.append(asyncio.create_task(self.on_test_completed(test)))
            # Await here in case any tests had errors and go back to CREATED state
            self.logger.debug(f"Awaiting {len(on_completed_tasks)} on_completed tasks")
            await asyncio.gather(*on_completed_tasks)

            limit = self.concurrency - self.state_counter[TestState.RUNNING]
            launched_tasks = []
            created_tests = [t for t in self.tests if t.state == TestState.CREATED]
            self.logger.debug(f"Launching up to {limit} new tests, queued={len(created_tests)}")
            for test in created_tests:
                if len(launched_tasks) >= limit:
                    break

                launched_tasks.append(asyncio.create_task(self.launch_test(test)))

            while len(launched_tasks) < limit and not self.generator_done:
                test = self.get_next_test(test_generator)
                if test is None:
                    self.generator_done = True
                    break
                if test == TestSet.NO_OP:
                    self.logger.info("no-op test")
                    break

                self.update_test_state(test)
                if test.state != TestState.CREATED:
                    self.logger.info(f"{test.name} was previously launched")
                else:
                    launched_tasks.append(asyncio.create_task(self.launch_test(test)))

            self.logger.debug(f"Awaiting {len(launched_tasks)} launched tasks and 15 sec sleep")
            await asyncio.gather(*launched_tasks, asyncio.sleep(15))

        self.save_state()

    def get_next_test(self, generator):
        while True:
            test = next(generator, None)
            if not test or test == TestSet.NO_OP:
                return test

            self.generated_cnt += 1
            # Tests have deterministic naming, so the generator may have emitted this test previously
            existing_test = next((t for t in self.tests if t.name == test.name), None)
            if existing_test:
                self.logger.info(f"{test.name} is a duplicate, skipping")
                continue
            else:
                self.logger.info(f"New generated test {test.name}")
                self.tests.append(test)
                return test

    async def launch_test(self, test):
        """update parameters file, compile, and launch backtest"""
        if not test.compile_id or test.launch_backtest_attempts > 5:
            # Only do 1 compile at a time since we have to change the project code each time.  Most likely lock is not
            # necessary since this will also have the GIL, but why take chances.
            async with self.compile_lock:
                if not self.api.update_parameters_file(self.project_id, self.initial_parameters_file, test.params,
                                                       test.extraneous_params):
                    self.logger.error(f"{test.name} update_parameters_file failed")
                    return

                create_compile_resp = self.api.create_compile(self.project_id)
                if create_compile_resp["success"] and create_compile_resp["state"] in ["BuildSuccess", "InQueue"]:
                    test.compile_id = create_compile_resp["compileId"]
                    self.logger.info(f"{test.name} compile state={create_compile_resp['state']} compile_id={test.compile_id}")
                    test.launch_backtest_attempts = 0
                else:
                    self.logger.error(f"create_compile failure: {create_compile_resp}")
                    return

        await asyncio.sleep(6)
        async with self.launch_rate_limiter:
            create_backtest_resp = await asyncio.get_running_loop().run_in_executor(
                self.io_tpe, self.api.create_backtest, self.project_id, test.compile_id, test.name)
            if create_backtest_resp["success"]:
                test.backtest_id = create_backtest_resp["backtest"]["backtestId"]
                test.state = TestState.RUNNING
                self.logger.info(f"{test.name} launched compile_id={test.compile_id} backtest_id={test.backtest_id}")
                self.save_state()
            else:
                test.launch_backtest_attempts += 1
                self.logger.error(f"{test.name} create_backtest failed compile_id={test.compile_id}")
                self.logger.error(create_backtest_resp)

    async def on_launch_rate_limited(self, until):
        duration = int(round(until - time.time()))
        self.logger.info(f"Launch rate limited for {duration} secs")

    async def on_test_completed(self, test: Test):
        if not test.result_saved:
            await self.save_test_result(test)

        # Only want to save test log once result has been validated and saved
        if test.result_saved and not test.log_saved:
            await self.save_test_log(test)

    async def save_test_result(self, test):
        """Download result for completed test and save it, also mark test state as completed."""
        already_saved = False
        try:
            result = await asyncio.get_running_loop().run_in_executor(self.io_tpe, self.cio.read_test_result, test)
            if result:
                already_saved = True
                self.logger.info(f"{test.name} result already downloaded")
            else:
                self.logger.info(f"{test.name} downloading result, attempt={test.read_backtest_attempts}")
                read_backtest_resp = await asyncio.get_running_loop().run_in_executor(
                    self.io_tpe, self.api.read_backtest, self.project_id, test.backtest_id)

                # If read_backtest request fails, or downloaded result fail validation, don't do anything.
                # We'll try again later.
                if read_backtest_resp["success"]:
                    result = TestResult(test, read_backtest_resp["backtest"])

            if result:
                self.test_set.on_test_completed(result)  # Test set can validate result too
                test.state = TestState.COMPLETED
                if not already_saved:
                    await asyncio.get_running_loop().run_in_executor(self.io_tpe, self.cio.write_test_result, result)
                test.result_saved = True
        except TestResultValidationException as e:
            # Unsupported case: We have already saved the result but it fails validation.  Don't try to clean up, just
            # fail loudly.
            if already_saved:
                raise

            test.read_backtest_attempts += 1
            self.logger.error(f"{test.name} validation exc retriable={e.retriable} attempts={test.read_backtest_attempts}")
            if not e.retriable or test.read_backtest_attempts >= 40:
                # Assume by this point that it will never work, rename it and set state to created so it will be
                # launched in next loop iteration
                self.logger.error(f"giving up on read_backtest for {test.name}, will relaunch it")
                update_backtest_resp = await asyncio.get_running_loop().run_in_executor(
                    self.io_tpe, self.api.update_backtest, self.project_id, test.backtest_id, f"(FAILED) {test.name}")
                if update_backtest_resp["success"]:
                    test.backtest_id = None
                    test.state = TestState.CREATED
                    test.read_backtest_attempts = 0

    async def save_test_log(self, test):
        if self.cio.test_log_exists(test):
            self.logger.info(f"{test.name} log already downloaded")
            test.log_saved = True
            return

        read_log_resp = await asyncio.get_running_loop().run_in_executor(
            self.io_tpe, self.api.read_backtest_log, self.project_id, test.backtest_id)
        if read_log_resp["success"]:
            self.cio.write_test_log(test, read_log_resp["BacktestLogs"])
            test.log_saved = True
        self.logger.info(f"{test.name} save_test_log success={read_log_resp['success']}")

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
