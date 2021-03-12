from abc import *
import copy
from datetime import date, datetime, timedelta
import dateutil as du
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from enum import Enum
import hashlib
import json
import logging
import math
import random

from babel.numbers import parse_decimal
import ciso8601
import coolname
import coolname.data
import numpy as np
import pandas as pd
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

        self.compile_id = None
        self.launch_backtest_attempts = 0
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
    """Sometimes results aren't available right away, so we have retries here.  Other times there is an actual issue
    that affected the running of the test so the results are completely invalid.  In which case, the only recourse is to
    re-run the test.
    """
    def __init__(self, retriable=True):
        self.retriable = retriable


class TestResult:

    required_keys = ["alphaRuntimeStatistics", "runtimeStatistics", "statistics", "totalPerformance"]
    runtime_statistics_required_keys = ["Equity", "Net Profit", "Return"]
    statistics_required_keys = ["Total Trades", "Drawdown", "Win Rate", "Compounding Annual Return", "Sharpe Ratio",
                                "Probabilistic Sharpe Ratio"]
    total_performance_required_keys = ["TradeStatistics", "ClosedTrades"]
    trade_statistics_required_keys = ["NumberOfWinningTrades", "NumberOfLosingTrades", "AverageProfit", "AverageLoss"]

    # All properties in AlphaRuntimeStatistics.cs have JsonProperty(DefaultValueHandling = DefaultValueHandling.Ignore)
    # annotation
    # https://github.com/QuantConnect/Lean/blob/47b5178c0b486ab36507249733adb9b66a9289b7/Common/AlphaRuntimeStatistics.cs
    # This means that they will not be returned if value is 0 so we can't treat them being missing as an error.
    alpha_runtime_statistics_required_keys = []  # ["ReturnOverMaxDrawdown", "SortinoRatio"]

    def __init__(self, test: Test, bt_result: dict):
        self.test = test
        if not self.validate_backtest_results(bt_result):
            raise TestResultValidationException()
        self.bt_result = bt_result
        self.runtime_statistics = bt_result["runtimeStatistics"]
        self.statistics = bt_result["statistics"]
        self.trade_statistics = bt_result["totalPerformance"]["TradeStatistics"]
        self.alpha_runtime_statistics = bt_result["alphaRuntimeStatistics"]

        self._closed_trades_df = None
        self._equity_time_series = None

    @classmethod
    def validate_backtest_results(cls, bt_result):
        return (
            all(isinstance(bt_result.get(k), dict) for k in cls.required_keys) and
            all(k in bt_result["alphaRuntimeStatistics"] for k in cls.alpha_runtime_statistics_required_keys) and
            all(k in bt_result["runtimeStatistics"] for k in cls.runtime_statistics_required_keys) and
            all(k in bt_result["statistics"] for k in cls.statistics_required_keys) and
            all(k in bt_result["totalPerformance"] for k in cls.total_performance_required_keys) and
            all(k in bt_result["totalPerformance"]["TradeStatistics"] for k in cls.trade_statistics_required_keys)
        )

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

    def closed_trades_df(self):
        if self._closed_trades_df is None:
            ct = self.bt_result["totalPerformance"]["ClosedTrades"]
            for t in ct:
                t["Symbol"] = t["Symbol"]["ID"]
                t["EntryTime"] = entry_time = ciso8601.parse_datetime(t["EntryTime"])
                t["ExitTime"] = exit_time = ciso8601.parse_datetime(t["ExitTime"])
                t["Duration"] = entry_time - exit_time

            self._closed_trades_df = pd.DataFrame(ct)
            del self.bt_result["totalPerformance"]["ClosedTrades"]  # Free up memory

        return self._closed_trades_df

    def total_trades(self):
        return int(self.statistics["Total Trades"])

    def total_winning_trades(self):
        return int(self.trade_statistics["NumberOfWinningTrades"])

    def total_losing_trades(self):
        return int(self.trade_statistics["NumberOfLosingTrades"])

    def winning_trades_df(self):
        df = self.closed_trades_df()
        return df[df["ProfitLoss"] > 0]

    def losing_trades_df(self):
        df = self.closed_trades_df()
        return df[df["ProfitLoss"] < 0]

    def avg_win(self):
        return Decimal(self.trade_statistics["AverageProfit"])

    def stdev_win(self):
        return Decimal(self.winning_trades_df()["ProfitLoss"].std())

    def avg_loss(self):
        return Decimal(self.trade_statistics["AverageLoss"])

    def stdev_loss(self):
        return Decimal(self.trade_statistics["ProfitLossDownsideDeviation"])

    def max_win(self):
        return Decimal(self.trade_statistics["LargestProfit"])

    def max_loss(self):
        return Decimal(self.trade_statistics["LargestLoss"])

    def max_win_trade(self):
        df = self.winning_trades_df()
        return df.loc[df["ProfitLoss"].idxmax()]

    def min_win_trade(self):
        df = self.winning_trades_df()
        return df.loc[df["ProfitLoss"].idxmin()]

    def max_loss_trade(self):
        df = self.losing_trades_df()
        return df.loc[df["ProfitLoss"].idxmin()]

    def min_loss_trade(self):
        df = self.losing_trades_df()
        return df.loc[df["ProfitLoss"].idxmax()]

    def net_profit(self):
        return self.parse_dollars(self.runtime_statistics["Net Profit"])

    def annualized_net_profit(self):
        return self.net_profit() * Decimal(timedelta(365) / self.duration())

    def drawdown(self):
        return self.parse_percent(self.statistics["Drawdown"])

    def annualized_return_over_max_drawdown(self):
        return Decimal(self.alpha_runtime_statistics.get("ReturnOverMaxDrawdown", 0))

    def win_rate(self):
        return self.parse_percent(self.statistics["Win Rate"])

    def compounding_annual_return(self):
        return self.parse_percent(self.statistics["Compounding Annual Return"])

    def total_return(self):
        return self.parse_percent(self.runtime_statistics["Return"])

    def sortino_ratio(self):
        return Decimal(self.alpha_runtime_statistics.get("SortinoRatio", 0))

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

        adj_gain = self.avg_win() * Decimal(self.total_winning_trades() - math.sqrt(self.total_winning_trades()))
        adj_loss = self.avg_loss() * Decimal(self.total_losing_trades() + math.sqrt(self.total_losing_trades()))

        adj_total_return = (adj_gain + adj_loss) / initial_equity
        if adj_total_return <= -1:  # Lost everything
            return Decimal(-1)

        bt_years = (self.test.end - self.test.start) / timedelta(365)
        adj_annualized_return = ((1 + adj_total_return) ** Decimal(1 / bt_years)) - 1

        return adj_annualized_return

    def equity_time_series(self):
        if self._equity_time_series is None:
            ts = self.bt_result["charts"]["Strategy Equity"]["Series"]["Equity"]["Values"]
            df = pd.DataFrame(ts)
            df["x"] = pd.to_datetime(df["x"], unit="s", utc=True)  # See BaseResultsHandler.Sample() in LEAN for UTC
            df["x"] = df["x"].dt.tz_convert("America/New_York")
            df.set_index("x", inplace=True)

            self._equity_time_series = df["y"]

        return self._equity_time_series

    def daily_returns(self):
        return self.equity_time_series().resample("1D").last().dropna().pct_change().dropna()

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
        """Return a generator that yields Test objects"""
        pass
    
    def on_test_completed(self, result: TestResult):
        pass


class MultiPeriod(TestSet):
    """Exercise the algorithm with a constant set of parameters over a range of time periods"""
    def __init__(self, periods: list, params: dict, extraneous_params=None):
        self.periods = periods
        self.params = params
        self.extraneous_params = extraneous_params

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"mp_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        for (i, (start, end)) in enumerate(self.periods):
            params = copy.deepcopy(self.params)
            params["start"] = start.isoformat()
            params["end"] = end.isoformat()
            name = Test.generate_name(f"{format(i, '02d')}_{start.isoformat()}_{end.isoformat()}", params)
            test = Test(name, params, extraneous_params=self.extraneous_params)
            yield test


class MultiPeriodYearly(MultiPeriod):
    def __init__(self, start_year: int, years: int, params: dict, interval_years=1, extraneous_params=None):
        periods = []
        for i in range(start_year, start_year + years, interval_years):
            start = date(i, 1, 1)
            end = start.replace(year=start.year + interval_years) - timedelta(days=1)
            periods.append((start, end))
        super().__init__(periods, params, extraneous_params=extraneous_params)


class MultiPeriodInterval(MultiPeriod):
    """Exercise the algorithm with a constant set of parameters over a range of time periods"""
    def __init__(self, start: date, end: date, intervals: int, params: dict, extraneous_params=None):
        interval = (end - start) / intervals
        periods = []
        for i in range(0, intervals):
            end = start + interval
            periods.append((start, end))
            start = end + timedelta(days=1)
        super().__init__(periods, params, extraneous_params=extraneous_params)


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
    def __init__(self, periods: list, param_grids: list, params_filter_fn=None, extraneous_params=None,
                 validation_fn=None):
        self.periods = periods
        self.param_grids = param_grids
        self.params_filter_fn = params_filter_fn
        self.extraneous_params = extraneous_params
        self.validation_fn = validation_fn

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"gs_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        num = 0
        for (start, end) in self.periods:
            for param_grid in self.param_grids:
                for params in ParameterGrid(param_grid):
                    if not self.params_filter_fn or self.params_filter_fn(params):
                        params["start"] = start.isoformat()
                        params["end"] = end.isoformat()
                        name = Test.generate_name(f"{format(num, '05d')}_{start.isoformat()}_{end.isoformat()}", params)
                        test = Test(name, params, extraneous_params=self.extraneous_params)
                        yield test
                        num += 1

    def on_test_completed(self, result: TestResult):
        if self.validation_fn:
            self.validation_fn(result)


class WalkForwardSingle(TestSet):
    """Executes a single "walk forward" step, performing a grid search over the in-sample(INS) period, selecting
    the best parameter set according to the ranking provided by the objective function, and then running that
    parameter set over the out-of-sample(OOS) period
    """
    logger = logging.getLogger(f"{__name__}.WalkForwardSingle")

    # TODO: Consider using relativedelta here
    def __init__(self, ins_start: date, ins_months: int, oos_months: int, param_grid: dict,
                 objective_fn, params_filter_fn=None, validation_fn=None, extraneous_params=None,
                 run_oos_rejects=False):
        """Assumption is that objective_fn produces higher scores for better results"""
        self.ins_start = ins_start
        self.ins_months = ins_months
        self.oos_months = oos_months
        self.ins_end = self.ins_start + relativedelta(months=self.ins_months) - timedelta(1)
        self.oos_start = self.ins_end + timedelta(1)
        self.oos_end = self.oos_start + relativedelta(months=self.oos_months) - timedelta(1)

        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.validation_fn = validation_fn
        self.params_filter_fn = params_filter_fn
        self.extraneous_params = extraneous_params
        self.run_oos_rejects = run_oos_rejects  # TODO: Better name?

        self.ins_tests = {}
        self.oos_test = None

    def name(self):
        return f"wf_{self.ins_months}_{self.oos_months}_{self.ins_start}_{self.oos_end}"

    def tests(self):
        # First we execute all the optimization tests
        ins_params = []
        for params in ParameterGrid(self.param_grid):
            if self.params_filter_fn and not self.params_filter_fn(params):
                continue
            ins_params.append(params)
            params["start"] = self.ins_start.isoformat()
            params["end"] = self.ins_end.isoformat()
            name = Test.generate_name(f"wf_{self.ins_months}_{self.oos_months}_ins_{self.ins_start}_{self.ins_end}", params)
            if name in self.ins_tests:
                self.logger.info(f"{name} already generated, skipping")
                continue
            test = Test(name, params, extraneous_params=self.extraneous_params)
            self.ins_tests[name] = (test, None)
            yield test

        # Until we get all test results back, no-op
        while not self.are_ins_tests_complete():
            yield TestSet.NO_OP

        if not self.oos_test:  # Don't want to keep returning the oos test
            self.oos_test = self.generate_oos_test()
            self.logger.info(f"Finished optimization, beginning OOS with best param set: {self.oos_test.to_dict()}")
            yield self.oos_test

        if self.run_oos_rejects:
            for params in ins_params:
                params = copy.deepcopy(params)
                params["start"] = self.oos_start.isoformat()
                params["end"] = self.oos_end.isoformat()
                if params != self.oos_test.params:
                    name = Test.generate_name(f"wf_{self.ins_months}_{self.oos_months}_oosrej_{self.oos_start}_{self.oos_end}", params)
                    yield Test(name, params, extraneous_params=self.extraneous_params)

    def are_ins_tests_complete(self):
        return all(r is not None for (_, r) in self.ins_tests.values())

    def on_test_completed(self, result):
        if self.validation_fn:
            self.validation_fn(result)

        if "_ins_" in result.test.name:
            self.ins_tests[result.test.name] = (result.test, self.objective_fn(result))

    def generate_oos_test(self):
        obj_values = [obj_val for (test, obj_val) in self.ins_tests.values()]
        best_ins_test, best_ins_obj_val = max(self.ins_tests.values(), key=lambda tup: tup[1])
        self.logger.info(f"max={best_ins_obj_val} obj_values={obj_values}")
        params = copy.deepcopy(best_ins_test.params)
        params["start"] = self.oos_start.isoformat()
        params["end"] = self.oos_end.isoformat()
        name = Test.generate_name(f"wf_{self.ins_months}_{self.oos_months}_ooswf_{self.oos_start}_{self.oos_end}", params)
        return Test(name, params, extraneous_params=self.extraneous_params)


class WalkForwardMultiple(TestSet):
    logger = logging.getLogger(__name__)

    # TODO: Consider using relativedelta here
    def __init__(self, start: date, end: date, ins_months: int, oos_months: int, param_grid: dict,
                 objective_fn, params_filter_fn=None, validation_fn=None, extraneous_params={}, run_combined=False,
                 run_oos_rejects=False):
        assert ins_months != oos_months, "Use different (ideally higher) optimization window from oos window"
        self.start = start
        self.end = end
        self.ins_months = ins_months
        self.oos_months = oos_months
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.params_filter_fn = params_filter_fn
        self.validation_fn = validation_fn
        self.extraneous_params = extraneous_params
        self.run_combined = run_combined
        self.run_oos_rejects = run_oos_rejects

        self.walk_forwards = self.sub_tests()
        self.oos_combined = None

    def name(self):
        return f"wf_{self.ins_months}_{self.oos_months}_{self.start.isoformat()}_{self.end.isoformat()}"

    def tests(self):
        """Currently this runs all tests from one walk forward step before beginning any from the next step.  This is
        useful since it groups all tests for a single step together in the backtest list on QC.  However, it is slower
        since it has to wait for all optimization tests for a walk-forward step to finish before it can move forward."""
        for wf in self.walk_forwards:
            for test in wf.tests():
                # Because a multi-step walk forward analysis produces so many tests, set a higher log level on each of
                # the sub-tests since they are short and can be re-run quickly to debug issues.  This helps avoid eating
                # into QC's daily log budget.
                if test != self.NO_OP:
                    test.extraneous_params["logLevel"] = "INFO"
                yield test
            self.logger.info(f"Finished all tests for walk forward opt={wf.ins_start} - {wf.ins_end}"
                             f" oos={wf.oos_start} - {wf.oos_end}")

        if self.run_combined:
            self.logger.info("Launching combined oos backtest")
            self.oos_combined = self.generate_oos_combined()
            for test in self.oos_combined:
                yield test

    def generate_oos_combined(self):
        """Run backtest over the entire OOS period covered by the walk forward analysis, using the same parameters
        for each individual walk forward's OOS.  This will help reveal if/where the strategy runs into issues with
        a growing capital base.  Also run a test with margin interest and tax modeling enabled, to give us a more
        realistic sense of real-world returns.
        """
        params_list = [copy.deepcopy(wf.oos_test.params) for wf in self.walk_forwards]
        start = params_list[0]["start"]
        end = params_list[-1]["end"]

        # The combined OOS test takes a long time to run so we leave debug logging on for it
        loglevel = {"logLevel": "DEBUG"}

        combined_name = Test.generate_name(f"wf_{self.ins_months}_{self.oos_months}_ooscombined_{start}_{end}", params_list)
        combined = Test(combined_name, params_list, extraneous_params={**self.extraneous_params, **loglevel})

        combined_mt_name = Test.generate_name(f"wf_{self.ins_months}_{self.oos_months}_ooscombinedmt_{start}_{end}", params_list)
        mt_extraneous = {"margin": {"rate": 0.01}, "tax": {"rate": 0.4}}
        combined_mt = Test(combined_mt_name, params_list,
                           extraneous_params={**self.extraneous_params, **loglevel, **mt_extraneous})

        return [combined, combined_mt]

    def sub_tests(self):
        sub = []
        ins_start = self.start
        while ins_start + relativedelta(months=self.ins_months) - timedelta(1) <= self.end:
            wfs = WalkForwardSingle(ins_start, self.ins_months, self.oos_months, self.param_grid,
                                    self.objective_fn, self.params_filter_fn, self.validation_fn,
                                    self.extraneous_params, self.run_oos_rejects)
            sub.append(wfs)

            ins_start += relativedelta(months=self.oos_months)

        return sub

    def on_test_completed(self, result):
        if self.oos_combined:
            if any(result.test == oos_test or result.test.params == oos_test.params for oos_test in self.oos_combined):
                return

        for wf in self.walk_forwards:
            # Implicit assumption here is that opt window size and oos window size are different (one of the reasons
            # for the assert statement in constructor).
            if ((result.test.start.date() == wf.ins_start and result.test.end.date() == wf.ins_end)
                    or (result.test.start.date() == wf.oos_start and result.test.end.date() == wf.oos_end)):
                wf.on_test_completed(result)
                return
