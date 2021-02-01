from abc import *
import copy
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from enum import Enum

import hashlib
import json
import coolname
import coolname.data
import random
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
    def __init__(self, name: str, params: dict, backtest_id=None, state=TestState.CREATED):
        self.name = name
        self.params = params
        self.backtest_id = backtest_id
        self.state = state
    
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


# TODO: Flesh out abstract class once have multiple generator types
# TODO: Maybe a method that exposes if it is a fixed size
class TestSet(ABC):
    """Can have a module per algorithm researched, with different implementations of this for different stages
    May eventually want to feed backtest results back into here, if it will dynamically influence test
    generation. If we get to that point, we'll want to see what's already implemented in modules out there"""
    @abstractmethod
    def name(self):
        pass
    
    # TODO: Rename to tests?
    @abstractmethod
    def tests(self):
        pass
    
    @abstractmethod
    def report(self, results):
        pass


class MultiPeriod:
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


class ParamSignificance:
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


class GridSearch:
    def __init__(self, periods: list, param_grid: dict, params_filter=None):
        self.periods = periods
        self.param_grid = param_grid
        self.params_filter = params_filter

    def name(self):
        start = self.periods[0][0]
        end = self.periods[-1][1]
        return f"gs_{start.isoformat()}_{end.isoformat()}"

    def tests(self):
        for (start, end) in self.periods:
            for params in ParameterGrid(self.param_grid)
                if not self.params_filter or self.params_filter(params):
                    params["start"] = start.isoformat()
                    params["end"] = end.isoformat()
                    name = Test.generate_name(f"{self.name()}", params)
                    test = Test(name, params)
                    yield test


class WalkForwardAnalysis:
    def __init__(self, start: date, end: date, opt_months: int, oos_months: int, param_grid: dict, params_filter=None):
        self.start = start
        self.end = end
        self.opt_months = opt_months
        self.oos_months = oos_months
        self.param_grid = param_grid
        self.params_filter = params_filter

    def name(self):
        return f"wfa_{self.start.isoformat()}_{self.end.isoformat()}_{self.opt_months}_{self.oos_months}"

    def tests(self):
        for ((opt_start, opt_end), (oos_start, oos_end)) in self.windows():
            for params in ParameterGrid(self.param_grid):
                if not self.params_filter or self.params_filter(params):
                    params["start"] = opt_start.isoformat()
                    params["end"] = opt_end.isoformat()
                    name = Test.generate_name(f"{self.name()}", params)
                    test = Test(name, params)
                    yield test
            # OMG HAVE TO REPORT BACK RESULTS

    def windows(self):
        w = []
        opt_start = self.start
        while True:
            opt_end = opt_start + relativedelta(months=self.opt_months) - timedelta(1)
            oos_start = opt_end + timedelta(1)
            oos_end = oos_start + relativedelta(months=self.oos_months) - timedelta(1)

            if oos_start >= self.end:
                break

            w.append(((opt_start, opt_end), (oos_start, oos_end)))
            opt_start += relativedelta(months=self.oos_months)
        return w
