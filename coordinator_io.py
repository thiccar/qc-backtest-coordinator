import json
import logging

import dictdiffer

from testsets import Test, TestResult, TestState


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
        assert result.test.state == TestState.COMPLETED
        result_path = self.test_result_path(result.test)
        with result_path.open('w') as f:
            json.dump(result.to_dict(), f, indent=4)

    def test_result_exists(self, test):
        result_path = self.test_result_path(test)
        return result_path.exists()

    def test_result_path(self, test):
        return self.test_set_path / f"{test.name}.json"

    def read_test_log(self, test):
        log_path = self.test_log_path(test)
        with log_path.open() as f:
            return f.readlines()

    def write_test_log(self, test, log):
        log_path = self.test_log_path(test)
        with log_path.open("w") as f:
            for line in log:
                f.write(line)
                f.write("\n")

    def test_log_exists(self, test):
        log_path = self.test_log_path(test)
        return log_path.exists()

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
