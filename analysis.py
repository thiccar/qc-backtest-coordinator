"""Code that can be shared among Jupyter notebooks"""
import csv
from datetime import datetime, timedelta
import functools
from itertools import groupby
import json
import statistics

from tabulate import tabulate

from coordinator_io import CoordinatorIO
from testsets import Test, TestState


class Analysis:
    def __init__(self, test_dir):
        self.cio = CoordinatorIO(test_dir)

    def tests(self):
        state = self.cio.read_state()
        return [Test.from_dict(t) for t in state["coordinator"]["tests"]]

    def results(self, only_keep_useful=False):
        for t in self.tests():
            # Allows us to run this while backtests are running to see intermediate results
            if t.state == TestState.RUNNING:
                continue
            result = self.cio.read_test_result(t)

            # If the definition of useful varies a lot across tests, we may need to be more flexible here
            if only_keep_useful:
                useful = {k: result.bt_result[k] for k in result.required_keys}
                useful["charts"] = {"Strategy Equity": result.bt_result["charts"]["Strategy Equity"]}
                result.bt_result = useful
            yield result

    def generate_csv_report(self):
        rows = []
        for result in self.results():
            stats = result.bt_result["statistics"]
            for k in ["SortinoRatio", "ReturnOverMaxDrawdown"]:
                stats[k] = result.bt_result["alphaRuntimeStatistics"][k]
            stats["PROE"] = result.proe()
            rows.append((result.test, stats))

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

    @classmethod
    def results_by_period(cls, results, sort_by_params=False):
        grouped = list(list(g) for (_, g) in groupby(results, key=lambda r: (r.test.start, r.test.end)))

        # Sort each period results so that the result at the same index has same params
        def params_sort_key(result):
            keys = sorted(k for k in result.test.params.keys() if k not in ["start", "end"])
            return list(result.test.params[k] for k in keys)

        if sort_by_params:
            for period_results in grouped:
                period_results.sort(key=params_sort_key)
            for tup in zip(*grouped):
                psk = [params_sort_key(r) for r in list(tup)]
                assert all(p == psk[0] for p in psk)

        return grouped

    @classmethod
    def tests_ending_early(cls, results) -> str:
        """Sanity check - find backtests whose equity curves end early. Returns a formatted table"""

        rows = []
        for (i, r) in enumerate(results):
            e = r.bt_result["charts"]["Strategy Equity"]["Series"]["Equity"]["Values"]
            last_date = datetime.fromtimestamp(e[-1]["x"])
            if r.test.end - last_date > timedelta(days=7):
                rows.append([i, r.test.name, last_date, json.dumps(r.test.params)])

        headers = ["index", "name", "last_date", "params"]
        return tabulate(rows, headers=headers)

    def error_logs(self):
        mapping = {}
        for t in self.tests():
            # Allows us to run this while backtests are running to see intermediate results
            if t.state == TestState.RUNNING:
                continue
            errors = []
            log = self.cio.read_test_log(t)
            for line in log:
                if "[ERROR]" in line:
                    errors.append(line)
            if errors:
                mapping[t] = errors

        return mapping

    @classmethod
    def profitability_summary_statistics(cls, results) -> str:
        """Return a formatted table showing summary statistics on profitability"""
        returns = [r.compounding_annual_return() for r in results]
        pos_returns = [r for r in returns if r > 0]
        neg_returns = [r for r in returns if r <= 0]

        headers = ["", "Number", "Percentage", "Average Annual Return", "Std Dev", "Max", "Min"]
        table = [
            ["Number of tests", len(results), 1, statistics.mean(returns), statistics.pstdev(returns), max(returns),
             min(returns)],
            ["Profitable tests", len(pos_returns), len(pos_returns) / len(results), statistics.mean(pos_returns),
             statistics.pstdev(pos_returns), max(pos_returns), min(pos_returns)],
            ["Losing tests", len(neg_returns), len(neg_returns) / len(results), statistics.mean(neg_returns),
             statistics.pstdev(neg_returns), max(neg_returns), min(neg_returns)],
        ]
        return tabulate(table, headers=headers)
