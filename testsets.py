from abc import *
import copy
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from enum import Enum
import hashlib
import json
import logging
import math
import random

from babel.numbers import parse_decimal
import coolname
import coolname.data
from sklearn.model_selection import ParameterGrid


class TestState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


USA_EQ_BEAR_MARKETS = [
    (date(2000, 1, 1), date(2002, 1, 1)),
    (date(2007, 10, 1), date(2009, 3, 1)),  # extremely high volatility
]
USA_EQ_LOW_VOL_BULL_MARKETS = [
    (date(2013, 1, 1),  date(2014, 6, 30)),
]
USA_EQ_HIGH_VOL_BULL_MARKETS = [
    (date(2018, 1, 1),  date(2020, 12, 31)),
]
USA_EQ_BULL_MARKETS = [
    (date(2002, 1, 1),  date(2007, 10, 1)),
    (date(2009, 3, 1),  date(2014, 12, 1)),
    (date(2016, 11, 1),  date(2021, 2, 1)),
]
USA_EQ_SIDEWAYS_MARKETS = [
    (date(2014, 12, 1),  date(2016, 11, 1)),  # Includes volatility spike in 8/2015
]


class Test:
    """Utility class for grouping together various pieces of information about a test."""

    def __init__(self, name: str, params, backtest_id=None, state=TestState.CREATED, extraneous_params=None):
        self.name = name

        # Use datetime here to be more general, however most of the test sets work with dates
        if isinstance(params, dict):
            self.start = datetime.fromisoformat(params["start"])
            self.end = datetime.fromisoformat(params["end"])
        elif isinstance(params, list):
            assert all(isinstance(p, dict) for p in params)
            assert all("start" in p for p in params)
            assert all("end" in p for p in params)
            self.start = datetime.fromisoformat(params[0]["start"])
            self.end = datetime.fromisoformat(params[-1]["end"])

        self.params = params
        self.backtest_id = backtest_id
        self.state = state
        self.extraneous_params = extraneous_params

        self.read_backtest_attempts = 0
        self.result_saved = False
        self.log_saved = False
    
    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params, "backtest_id": self.backtest_id, "state": self.state.name}

    @staticmethod
    def from_dict(d: dict):
        return Test(d["name"], d["params"], d["backtest_id"], TestState(d["state"]))

    @staticmethod
    def generate_name(prefix, params):
        seed = hashlib.sha256(f"{prefix}{json.dumps(params)}".encode("utf-8")).digest()
        generator = coolname.RandomGenerator(coolname.data.config, random.Random(seed))
        cool_name = ''.join(x.capitalize() for x in generator.generate())
        return f"{prefix}_{cool_name}"


class TestResultValidationException(Exception):
    pass


class TestResult:

    required_keys = ["alphaRuntimeStatistics", "runtimeStatistics", "statistics", "totalPerformance"]

    def __init__(self, test: Test, bt_result: dict):
        self.test = test
        if not self.validate_backtest_results(bt_result):
            raise TestResultValidationException()
        self.bt_result = bt_result
        self.runtime_statistics = bt_result["runtimeStatistics"]
        self.statistics = bt_result["statistics"]
        self.alpha_runtime_statistics = bt_result["alphaRuntimeStatistics"]
        self.trade_statistics = bt_result["totalPerformance"]["TradeStatistics"]

    @classmethod
    def validate_backtest_results(cls, bt_result):
        return all((key in bt_result and bt_result[key]) for key in cls.required_keys)

    def to_dict(self) -> dict:
        return {"test": self.test.to_dict(), "backtest": self.bt_result}

    @staticmethod
    def from_dict(d: dict):
        return TestResult(Test.from_dict(d["test"]), d["backtest"])

    """
    Utility methods for accessing result fields
    """
    def duration(self):
        return self.test.end - self.test.start

    def final_equity(self):
        return self.parse_dollars(self.runtime_statistics["Equity"])

    def total_trades(self):
        return int(self.statistics["Total Trades"])

    def winning_trades(self):
        return int(self.trade_statistics["NumberOfWinningTrades"])

    def losing_trades(self):
        return int(self.trade_statistics["NumberOfLosingTrades"])

    def avg_win(self):
        return Decimal(self.trade_statistics["AverageProfit"])

    def avg_loss(self):
        return Decimal(self.trade_statistics["AverageLoss"])

    def net_profit(self):
        return self.parse_dollars(self.runtime_statistics["Net Profit"])

    def annualized_net_profit(self):
        return self.net_profit() * Decimal(timedelta(365) / self.duration())

    def drawdown(self):
        return self.parse_percent(self.statistics["Drawdown"])

    def annualized_return_over_max_drawdown(self):
        return Decimal(self.alpha_runtime_statistics["ReturnOverMaxDrawdown"])

    def win_rate(self):
        return self.parse_percent(self.statistics["Win Rate"])

    def compounding_annual_return(self):
        return self.parse_percent(self.statistics["Compounding Annual Return"])

    def total_return(self):
        return self.parse_percent(self.runtime_statistics["Return"])

    def sortino_ratio(self):
        return Decimal(self.alpha_runtime_statistics["SortinoRatio"])

    def sharpe_ratio(self):
        return Decimal(self.statistics["Sharpe Ratio"])

    def probabilistic_sharpe_ratio(self):
        return self.parse_percent(self.statistics["Probabilistic Sharpe Ratio"])

    def proe(self):
        """Pessimistic return on equity. Variant of PROM (Pessimistic Return on Margin) from Pardo's book "The
        Evaluation and Optimization of Trading Strategies" (chp 9). Make gross profit more pessimistic by reducing
        number of winning trades by square root and increasing number of losing trades by square root. This adjusted
        gross profit is then used to compute an annualized return on initial account equity.
        """
        initial_equity = self.final_equity() / (1 + self.total_return())

        adj_gain = self.avg_win() * Decimal(self.winning_trades() - math.sqrt(self.winning_trades()))
        adj_loss = self.avg_loss() * Decimal(self.losing_trades() + math.sqrt(self.losing_trades()))

        adj_total_return = (adj_gain + adj_loss) / initial_equity
        if adj_total_return <= -1:  # Lost everything
            return Decimal(-1)

        bt_years = (self.test.end - self.test.start) / timedelta(365)
        adj_annualized_return = ((1 + adj_total_return) ** Decimal(1 / bt_years)) - 1

        return adj_annualized_return

    @classmethod
    def parse_dollars(cls, s: str) -> Decimal:
        assert s.startswith("$")
        return parse_decimal(s.strip("$"), locale='en_US')

    @classmethod
    def parse_percent(cls, s: str) -> Decimal:
        assert s.endswith("%")
        return parse_decimal(s.strip("%"), locale='en_US') / 100


class TestSet(ABC):
    NO_OP = Test("no-op", None)  # Returned when don't have a new test yet, but are not done generating tests.

    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def tests(self):
        pass
    
    def on_test_completed(self, results: TestResult):
        pass


class MultiPeriod(TestSet):
    """Exercise the algorithm with a constant set of parameters over a range of time periods"""
    def __init__(self, periods: list, params: dict):
        self.periods = periods
        self.params = params

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"mp_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        """Return a generator that yields Test objects"""
        for (start, end) in self.periods:
            params = copy.deepcopy(self.params)
            params["start"] = start.isoformat()
            params["end"] = end.isoformat()
            name = Test.generate_name(f"mp_{start.isoformat()}_{end.isoformat()}", params)
            test = Test(name, params)
            yield test


class MultiPeriodYearly(MultiPeriod):
    def __init__(self, start_year: int, years: int, params: dict):
        periods = []
        for i in range(start_year, start_year + years):
            start = date(i, 1, 1)
            end = start.replace(year=start.year + 1) - timedelta(days=1)
            periods.append((start, end))
        super().__init__(periods, params)


class MultiPeriodInterval(MultiPeriod):
    """Exercise the algorithm with a constant set of parameters over a range of time periods"""
    def __init__(self, start: date, end: date, intervals: int, params: dict):
        interval = (end - start) / intervals
        periods = []
        for i in range(0, intervals):
            end = start + interval
            periods.append((start, end))
            start = end + timedelta(days=1)
        super().__init__(periods, params)


class ParamSignificance(TestSet):
    """Cycle one parameter at a time while keeping the rest at passed in defaults. So the total number of tests run
    (if using a single time range) will be the sum of the number of values in each parameter range"""

    def __init__(self, periods: list, defaults: dict, param_ranges: dict):
        self.periods = periods
        self.defaults = defaults
        self.param_ranges = param_ranges

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"ps_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        for (start, end) in self.periods:
            for (key, param_range) in self.param_ranges.items():
                for value in param_range:
                    params = copy.deepcopy(self.defaults)
                    params[key] = value
                    params["start"] = start.isoformat()
                    params["end"] = end.isoformat()
                    name = Test.generate_name(f"{self.name()}_{key}", params)
                    test = Test(name, params)
                    yield test


class GridSearch(TestSet):
    def __init__(self, periods: list, param_grid: dict, params_filter=None, extraneous_params=None):
        self.periods = periods
        self.param_grid = param_grid
        self.params_filter = params_filter
        self.extraneous_params = extraneous_params

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"gs_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        for (start, end) in self.periods:
            for params in ParameterGrid(self.param_grid):
                if not self.params_filter or self.params_filter(params):
                    params["start"] = start.isoformat()
                    params["end"] = end.isoformat()
                    name = Test.generate_name(f"gs_{start.isoformat()}_{end.isoformat()}", params)
                    test = Test(name, params, extraneous_params=self.extraneous_params)
                    yield test


class WalkForwardSingle(TestSet):
    """Executes a single "walk forward" step, performing a grid search over the optimization period, selecting
    the best parameter set according to the ranking provided by the objective function, and then running that
    parameter set over the OOS period
    """
    logger = logging.getLogger(__name__)

    # TODO: Consider using relativedelta here
    def __init__(self, opt_start: date, opt_months: int, oos_months: int, param_grid: dict,
                 objective_fn, params_filter=None, extraneous_params=None):
        """Assumption is that objective_fn produces higher scores for better results"""
        self.opt_start = opt_start
        self.opt_months = opt_months
        self.oos_months = oos_months
        self.opt_end = self.opt_start + relativedelta(months=self.opt_months) - timedelta(1)
        self.oos_start = self.opt_end + timedelta(1)
        self.oos_end = self.oos_start + relativedelta(months=self.oos_months) - timedelta(1)

        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.params_filter = params_filter
        self.extraneous_params = extraneous_params

        self.opt_tests = []
        self.oos_test = None

    def name(self):
        return f"wf_{self.opt_months}_{self.oos_months}_{self.opt_start}_{self.oos_end}"

    def tests(self):
        # First we execute all the optimization tests
        for params in ParameterGrid(self.param_grid):
            if self.params_filter and not self.params_filter(params):
                continue
            params["start"] = self.opt_start.isoformat()
            params["end"] = self.opt_end.isoformat()
            name = Test.generate_name(f"wf_{self.opt_months}_{self.oos_months}_opt_{self.opt_start}_{self.opt_end}", params)
            test = Test(name, params, extraneous_params=self.extraneous_params)
            self.opt_tests.append((test, None))
            yield test

        # Until we get all test results back, no-op
        while True:
            if all(r is not None for (t, r) in self.opt_tests):
                break
            else:
                yield TestSet.NO_OP

        if not self.oos_test:  # Don't want to keep returning the oos test
            self.oos_test = self.generate_oos_test()
            self.logger.info(f"Finished optimization, beginning OOS with best param set: {self.oos_test.to_dict()}")
            yield self.oos_test

    def on_test_completed(self, results):
        for (i, (t, r)) in enumerate(self.opt_tests):
            if not r and t == results.test or t.params == results.test.params:
                self.opt_tests[i] = (t, results)

    def generate_oos_test(self):
        obj_values = [self.objective_fn(r) for (t, r) in self.opt_tests]
        best = max((r for (t, r) in self.opt_tests), key=self.objective_fn)
        self.logger.info(f"obj_values={obj_values} max={max(obj_values)}")
        params = copy.deepcopy(best.test.params)
        params["start"] = self.oos_start.isoformat()
        params["end"] = self.oos_end.isoformat()
        name = Test.generate_name(f"wf_{self.opt_months}_{self.oos_months}_oos_{self.oos_start}_{self.oos_end}", params)
        return Test(name, params, extraneous_params=self.extraneous_params)


class WalkForwardMultiple(TestSet):
    logger = logging.getLogger(__name__)

    # TODO: Consider using relativedelta here
    def __init__(self, start: date, end: date, opt_months: int, oos_months: int, param_grid: dict,
                 objective_fn, params_filter=None, extraneous_params={}):
        assert opt_months != oos_months, "Use different (ideally higher) optimization window from oos window"
        self.start = start
        self.end = end
        self.opt_months = opt_months
        self.oos_months = oos_months
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.params_filter = params_filter
        self.extraneous_params = extraneous_params

        self.walk_forwards = self.sub_tests()
        self.oos_test = None

    def name(self):
        return f"wf_{self.opt_months}_{self.oos_months}_{self.start.isoformat()}_{self.end.isoformat()}"

    def tests(self):
        """Currently this runs all tests from one walk forward step before beginning any from the next step.  This is
        useful since it groups all tests for a single step together in the backtest list on QC.  However, it is slower
        since it has to wait for all optimization tests for a walk-forward step to finish before it can move forward."""
        for wf in self.walk_forwards:
            for test in wf.tests():
                # Because a multi-step walk forward analysis produces so many tests, set a higher log level on each of
                # the sub-tests since they are short and can be re-run quickly to debug issues.
                if test != self.NO_OP:
                    test.extraneous_params["logLevel"] = "INFO"
                yield test
            self.logger.info(f"Finished all tests for walk forward opt={wf.opt_start} - {wf.opt_end}"
                             f" oos={wf.oos_start} - {wf.oos_end}")

        self.logger.info("Launching combined oos backtest")
        self.oos_test = self.generate_oos_test()
        yield self.oos_test

    def generate_oos_test(self):
        params_list = [wf.oos_test.params for wf in self.walk_forwards]
        start = params_list[0]["start"]
        end = params_list[-1]["end"]
        name = Test.generate_name(f"wf_{self.opt_months}_{self.oos_months}_oos_{self.start}_{self.end}", params_list)

        # The combined OOS test takes a long time to run so we leave debug logging on for it
        loglevel = {"logLevel": "DEBUG"}
        return Test(name, params_list, extraneous_params={**self.extraneous_params, **loglevel})

    def sub_tests(self):
        sub = []
        opt_start = self.start
        while opt_start + relativedelta(months=self.opt_months + self.oos_months) - timedelta(1) <= self.end:
            wfs = WalkForwardSingle(opt_start, self.opt_months, self.oos_months, self.param_grid,
                                    self.objective_fn, self.params_filter, self.extraneous_params)
            sub.append(wfs)

            opt_start += relativedelta(months=self.oos_months)

        return sub

    def on_test_completed(self, results):
        if self.oos_test and (results.test == self.oos_test or results.test.params == self.oos_test.params):
            return
        for wf in self.walk_forwards:
            # Implicit assumption here is that opt window size and oos window size are different (one of the reasons
            # for the assert statement in constructor).
            if ((results.test.start.date() == wf.opt_start and results.test.end.date() == wf.opt_end)
                    or (results.test.start.date() == wf.oos_start and results.test.end.date() == wf.oos_end)):
                wf.on_test_completed(results)
                return
