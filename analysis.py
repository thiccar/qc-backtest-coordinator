"""Code that can be shared among Jupyter notebooks"""
from collections import Counter
import csv
from datetime import datetime, timedelta
from decimal import Decimal
import functools
import itertools
import json
import logging
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tabulate import tabulate

from coordinator_io import CoordinatorIO
from testsets import Test, TestResult, TestState


class Analysis:
    logger = logging.getLogger(__name__)

    def __init__(self, test_dir):
        self.cio = CoordinatorIO(test_dir)

    def tests(self):
        state = self.cio.read_state()
        return [Test.from_dict(t) for t in state["coordinator"]["tests"]]

    @classmethod
    def tests_with_duplicate_params(cls, tests):
        counts = Counter(tuple(t.params.items()) for t in tests)
        dupes = []
        for (p, c) in counts.items():
            if c > 1:
                dupes.extend(t for t in tests if tuple(t.params.items()) == p)
        return dupes

    def results(self, exclude_trades=False, exclude_rolling_window=False, exclude_charts=False, test_filter_fn=None):
        for t in self.tests():
            # Allows us to run this while backtests are running to see intermediate results
            if t.state == TestState.RUNNING:
                continue
            if test_filter_fn is None or test_filter_fn(t):
                result = self.cio.read_test_result(t)
                assert result is not None, f"{t.name} result missing"
                if exclude_trades:
                    del result.bt_result["totalPerformance"]["ClosedTrades"]
                if exclude_rolling_window:
                    del result.bt_result["rollingWindow"]
                if exclude_charts:
                    del result.bt_result["charts"]
                yield result

    @classmethod
    def all_params_keys(cls, results):
        return functools.reduce(
            lambda s1, s2: s1 | s2, (set(r.test.params.keys()) for r in results if isinstance(r.test.params, dict)))

    def generate_csv_report(self):
        rows = []
        results = list(self.results())
        for result in results:
            stats = result.bt_result["statistics"]
            for k in ["SortinoRatio", "ReturnOverMaxDrawdown"]:
                stats[k] = result.bt_result["alphaRuntimeStatistics"].get(k, 0)
            stats["PROE"] = result.proe()
            rows.append((result.test, stats))

        params_keys = self.all_params_keys(results)
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
        grouped = list(list(g) for (_, g) in itertools.groupby(results, key=lambda r: (r.test.start, r.test.end)))

        # Sort each period results so that the result at the same index has same params
        def params_sort_key(result):
            # https://stackoverflow.com/questions/5884066/hashing-a-dictionary
            return hash(frozenset(item for item in result.test.params.items() if item[0] not in ["start", "end"]))

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
            last_date = r.equity_time_series().index[-1]
            if pd.Timestamp(r.test.end, tz="America/New_York") - last_date > timedelta(days=7):
                rows.append([i, r.test.name, last_date, r.bt_result["nodeName"], json.dumps(r.test.params)])

        headers = ["index", "name", "last_date", "node", "params"]
        return tabulate(rows, headers=headers)

    def error_logs(self):
        search = [
            "[ERROR]",
            "log data per backtest",
            "Please upgrade your account",
        ]
        mapping = {}
        for t in self.tests():
            # Allows us to run this while backtests are running to see intermediate results
            if t.state == TestState.RUNNING:
                continue
            errors = []
            log = self.cio.read_test_log(t)
            for line in log:
                if any(s in line for s in search):
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
        pos_returns_row = ["Profitable tests", len(pos_returns), len(pos_returns) / len(results)]
        if pos_returns:
            pos_returns_row.extend([statistics.mean(pos_returns), statistics.pstdev(pos_returns),
                                    max(pos_returns), min(pos_returns)])
        neg_returns_row = ["Losing tests", len(neg_returns), len(neg_returns) / len(results)]
        if neg_returns:
            neg_returns_row.extend([statistics.mean(neg_returns), statistics.pstdev(neg_returns),
                                    max(neg_returns), min(neg_returns)])
        table = [
            ["Number of tests", len(results), 1, statistics.mean(returns), statistics.pstdev(returns), max(returns),
             min(returns)],
            pos_returns_row,
            neg_returns_row,
        ]
        return tabulate(table, headers=headers)

    @classmethod
    def wfa_grouped_results(cls, results, ins_marker="_ins_", oos_wf_marker="_ooswf_", oos_rej_marker="_oosrej_"):
        oos_wf_sorted = list(sorted([r for r in results if oos_wf_marker in r.test.name], key=lambda r: r.test.start))

        combined_start = oos_wf_sorted[0].test.start
        combined_end = oos_wf_sorted[-1].test.end
        oos_combined = filter(
            lambda r: (r.test.start, r.test.end) == (combined_start, combined_end),
            oos_wf_sorted)
        oos_combined = next(oos_combined, None)
        if oos_combined:
            oos_wf_sorted.remove(oos_combined)

        ins_grouped = itertools.groupby([r for r in results if ins_marker in r.test.name], key=lambda r: r.test.start)
        ins_sorted = sorted([(d, list(g)) for (d, g) in ins_grouped], key=lambda tup: tup[0])

        oos_rej_grouped = itertools.groupby([r for r in results if oos_rej_marker in r.test.name], key=lambda r: r.test.start)
        oos_rej_sorted = sorted([(d, list(g)) for (d, g) in oos_rej_grouped], key=lambda tup: tup[0])

        if len(oos_rej_sorted) > 0 and len(oos_rej_sorted) != len(oos_wf_sorted):
            oos_wf_dates = [r.test.start.date().isoformat() for r in oos_wf_sorted]
            oos_rej_dates = [t[0].date().isoformat() for t in oos_rej_sorted]
            msg = f"len(oos_rej_sorted) != len(oos_wf_sorted. oos_wf_dates={oos_wf_dates} oos_rej_dates={oos_rej_dates}"
            raise AssertionError(msg)
        if len(oos_rej_sorted) == 0:
            oos_rej_sorted = [[]] * len(oos_wf_sorted)

        grouped_results = [(ins, oos_wf, oos_rej) for ((_, ins), oos_wf, (_, oos_rej)) in zip(ins_sorted, oos_wf_sorted, oos_rej_sorted)]

        for (ins, oos_wf, oos_rej) in grouped_results:
            assert oos_wf.test.start == ins[0].test.end + timedelta(1), "OOS walk-forward date not adjacent to in-sample"
            assert all(r.test.start == oos_wf.test.start and r.test.end == oos_wf.test.end for r in oos_rej),\
                   "oos_rej dates don't match oos_wf"
            assert len(oos_rej) == len(ins) - 1,\
                f"expected {len(ins)-1} oos_rej, got {len(oos_rej)} start={oos_wf.test.start.date().isoformat()}"

        return grouped_results

    @classmethod
    def wfa_summary_statistics(cls, wfa_grouped_results, objective_fn, show_params=True):
        """Return a formatted table with WFA summary statistics a la Pardo (see https://pasteboard.co/JMmVgHk.png)"""
        headers = ["", "INS\nStart", "INS\nEnd", "Best INS P/L\nAnnualized", "Best INS\nMax Drawdown", "OOS\nStart",
                   "OOS\nEnd", "Net P/L", "Net P/L\nAnnualized", "Max\nDrawdown", "ROMAD\nAnnualized", "Win %",
                   "Walk-Forward\nEfficiency", "Sortino\nRatio", "Sharpe\nRatio", "PSR", "OOS Params"]
        table = []
        for (ins, oos_wf, _) in wfa_grouped_results:
            best_ins = max(ins, key=objective_fn)
            row = [
                None,
                best_ins.test.start.date(),
                best_ins.test.end.date(),
                best_ins.annualized_net_profit(),
                round(best_ins.drawdown(), 3),
                oos_wf.test.start.date(),
                oos_wf.test.end.date(),
                oos_wf.net_profit(),
                oos_wf.annualized_net_profit(),
                oos_wf.drawdown(),
                oos_wf.annualized_return_over_max_drawdown(),
                oos_wf.win_rate(),
                oos_wf.annualized_net_profit() / best_ins.annualized_net_profit(),
                oos_wf.sortino_ratio(),
                oos_wf.sharpe_ratio(),
                oos_wf.probabilistic_sharpe_ratio(),
                json.dumps(oos_wf.test.params) if show_params else "",
            ]
            table.append(row)

        mean_ins_annualized_pl = statistics.mean(r[3] for r in table)
        max_ins_drawdown = max(r[4] for r in table)
        summary = [
            "Aggregate", "", "",
            mean_ins_annualized_pl,
            max_ins_drawdown,
            wfa_grouped_results[0][1].test.start.date(), wfa_grouped_results[-1][1].test.end.date(),
            sum(r[7] for r in table),
            statistics.mean(r[8] for r in table),
            max(r[9] for r in table),
            statistics.mean(r[10] for r in table),
            statistics.mean(r[11] for r in table),
            statistics.mean(r[12] for r in table),
            statistics.mean(r[13] for r in table),
            statistics.mean(r[14] for r in table),
            statistics.mean(r[15] for r in table),
        ]
        table.append(summary)

        return tabulate(table, headers=headers, numalign="right", floatfmt=",.2f")

    @classmethod
    def stitch_oos_equity_curve(cls, wfa_grouped_results):
        combined = None
        for (_, oos_result) in wfa_grouped_results:
            e = oos_result.equity_time_series()
            if combined is None:
                combined = e
            else:
                multiplier = combined.iloc[-1] / e.iloc[0]
                e = e * multiplier
                combined = combined.append(e)

        duration = combined.index[-1] - combined.index[0]
        total_return = combined.iloc[-1] / combined.iloc[0]
        annualized_return = (total_return - 1) ** (1 / (duration / timedelta(365)))
        print(f"Stitched: Duration = {duration} Total Return = {total_return - 1}"
              f"Annualized Return = {annualized_return - 1}")

        return combined

    @classmethod
    def total_trades_bar_graph(cls, fig, ax, results, label_fn=None):
        """Sanity check - Graph number of trades to look for outliers"""
        xs = [label_fn(r) for r in results] if label_fn else None
        ys = [r.total_trades() for r in results]

        ax.bar(x=range(len(ys)), height=ys, color="green", tick_label=xs)
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def oos_return_drawdown_bar_graph(cls, fig, ax, wfa_grouped_results):
        """Graph individual OOS test metrics over time, see evolution. Followed example of
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
        """
        labels = [oos_result.test.start.date() for (_, oos_result) in wfa_grouped_results]
        tr = [oos_result.total_return() for (_, oos_result) in wfa_grouped_results]
        dd = [oos_result.drawdown() for (_, oos_result) in wfa_grouped_results]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, tr, width, label="Total Return", color="green")
        ax.bar(x + width / 2, dd, width, label="Drawdown", color="red")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def oos_sharpe_sortino_bar_graph(cls, fig, ax, wfa_grouped_results):
        labels = [oos_result.test.start.date() for (_, oos_result) in wfa_grouped_results]
        sharpe = [oos_result.sharpe_ratio() for (_, oos_result) in wfa_grouped_results]
        sortino = [oos_result.sortino_ratio() for (_, oos_result) in wfa_grouped_results]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, sharpe, width, label="Sharpe", color="green")
        ax.bar(x + width / 2, sortino, width, label="Sortino", color="blue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def oos_walk_forward_efficiency(cls, wfa_grouped_results, objective_fn):
        wfe = []
        for (ins, oos_wf, _) in wfa_grouped_results:
            best_opt = max(ins, key=objective_fn)
            wfe.append(oos_wf.annualized_net_profit() / best_opt.annualized_net_profit())
        return wfe

    @classmethod
    def oos_walk_forward_efficiency_bar_graph(cls, fig, ax, wfa_grouped_results, objective_fn):
        wfe = cls.oos_walk_forward_efficiency(wfa_grouped_results, objective_fn)
        labels = [oos_wf.test.start.date() for (_, oos_wf, _) in wfa_grouped_results]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x, wfe, width, label="Walk Forward Efficiency", color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def oos_walk_forward_vs_in_sample(cls, wfa_grouped_results, metric_fn):
        ins_metrics = [[metric_fn(r) for r in ins] for (ins, _, _) in wfa_grouped_results]
        oos_wf_metrics = [metric_fn(oos_wf) for (_, oos_wf, _) in wfa_grouped_results]
        percentiles = [percentileofscore(ins, oos) for (ins, oos) in zip(ins_metrics, oos_wf_metrics)]
        return ins_metrics, oos_wf_metrics, percentiles

    @classmethod
    def oos_walk_forward_vs_in_sample_graph(cls, fig, ax, wfa_grouped_results, metric_fn, metric_name):
        ins_metrics, oos_metrics, percentiles = cls.oos_walk_forward_vs_in_sample(wfa_grouped_results, metric_fn)
        ax.boxplot(ins_metrics,
                   positions=np.arange(len(wfa_grouped_results)),
                   labels=[oos_wf.test.start.date() for (_, oos_wf, _) in wfa_grouped_results])
        lines = ax.plot(oos_metrics)
        xy = lines[0].get_xydata()
        for (i, pct) in enumerate(percentiles):
            ax.annotate(f"{round(pct,1)}%", xy=xy[i])
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
        ax.set_title(f"OOS vs INS {metric_name}")

    @classmethod
    def oos_walk_forward_vs_oos_rejected(cls, wfa_grouped_results, metric_fn):
        oos_rej_metrics = [[metric_fn(r) for r in oos_rej] for (_, _, oos_rej) in wfa_grouped_results]
        oos_wf_metrics = [metric_fn(oos_wf) for (_, oos_wf, _) in wfa_grouped_results]
        percentiles = [percentileofscore(oos_rej, oos) for (oos_rej, oos) in zip(oos_rej_metrics, oos_wf_metrics)]
        return oos_rej_metrics, oos_wf_metrics, percentiles

    @classmethod
    def oos_walk_forward_vs_oos_rejected_graph(cls, fig, ax, wfa_grouped_results, metric_fn, metric_name):
        oos_rej_metrics, oos_metrics, percentiles = cls.oos_walk_forward_vs_oos_rejected(wfa_grouped_results, metric_fn)
        ax.boxplot(oos_rej_metrics,
                   positions=np.arange(len(wfa_grouped_results)),
                   labels=[oos_wf.test.start.date() for (_, oos_wf, _) in wfa_grouped_results])
        lines = ax.plot(oos_metrics)
        xy = lines[0].get_xydata()
        for (i, pct) in enumerate(percentiles):
            ax.annotate(f"{round(pct,1)}%", xy=xy[i])
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
        ax.set_title(f"OOS WF vs OOS Rejected {metric_name}")

    @classmethod
    def top_btm_days(cls, results, n=10):
        top_each = [r.daily_returns().nlargest(n).to_frame().assign(name=r.test.name) for r in results]
        btm_each = [r.daily_returns().nsmallest(n).to_frame().assign(name=r.test.name) for r in results]

        return pd.concat(top_each, copy=False).nlargest(n, "y"),\
               pd.concat(btm_each, copy=False).nsmallest(n, "y")

    @classmethod
    def top_btm_trades(cls, results, n=10):
        """Default TradeBuilder does not track splits leading to extreme values at top/bottom, see
        https://github.com/QuantConnect/Lean/issues/5325
        """
        top_each = [r.closed_trades_df().nlargest(n, "ProfitLoss").assign(name=r.test.name) for r in results]
        btm_each = [r.closed_trades_df().nsmallest(n, "ProfitLoss").assign(name=r.test.name) for r in results]

        top = pd.concat(top_each, ignore_index=True, copy=False).nlargest(n, "ProfitLoss")
        btm = pd.concat(btm_each, ignore_index=True, copy=False).nsmallest(n, "ProfitLoss")

        # Many backtests can have the same trade, so collapse duplicates keeping the most extreme value.
        top = top.groupby(["Symbol", "EntryTime", "ExitTime"], as_index=False, sort=False).first()
        btm = btm.groupby(["Symbol", "EntryTime", "ExitTime"], as_index=False, sort=False).first()

        return top, btm

    @classmethod
    def wfa_equity_curve_plot(cls, fig, ax, wfa_grouped_results, oos_combined, label=""):
        """Graph equity curve stitched together from individual OOS tests.  NOTE: It may look like the stitched curve
        starts later than the single curve, that is because they perfectly overlap for the duration of the first OOS
        test.
        """
        stitched = cls.stitch_oos_equity_curve(wfa_grouped_results)

        ax.plot(stitched, label=f"{label}_stitched" if label else "stitched")
        ax.set_yscale("log")

        if oos_combined:
            print(f"Combined: Duration = {oos_combined.duration()} Total Return = {oos_combined.total_return()} "
                  f"Annualized Return = {oos_combined.compounding_annual_return()}")
            ax.plot(oos_combined.equity_time_series(), label=f"{label}_continuous" if label else "continuous")
        ax.legend()

    @classmethod
    def params_plot(cls, fig, results):
        """Graph change in parameters used"""
        params_keys = list(k for k in results[0].test.params.keys() if k not in ["start", "end"])
        xs = [r.test.start for r in results]
        axs = fig.subplots(len(params_keys), 1)
        for (ax, key) in zip(axs, params_keys):
            ys = [r.test.params[key] for r in results]
            ax.plot(xs, ys, label=key)
            ax.set_title(key)

    @classmethod
    def wfa_evaluation_profile(cls, oos_wf_results):
        """Produce a table like https://pasteboard.co/JO2OL6Y.png (Table 14.1 from "The Evaluation and Optimization of
        Trading Strategies"), as well as similar stats for rolling windows.
        """
        return "\n".join([
            cls.wfa_evaluation_profile_header(oos_wf_results),
            cls.wfa_evaluation_profile_trade_analysis(oos_wf_results),
            cls.wfa_evaluation_profile_equity_swings(oos_wf_results),
            cls.wfa_evaluation_rolling_window(oos_wf_results, pd.Timedelta(7, "D")),
            cls.wfa_evaluation_rolling_window(oos_wf_results, pd.Timedelta(30, "D")),
            cls.wfa_evaluation_rolling_window(oos_wf_results, pd.Timedelta(90, "D")),
            cls.wfa_evaluation_rolling_window(oos_wf_results, pd.Timedelta(180, "D")),
            cls.wfa_evaluation_rolling_window(oos_wf_results, pd.Timedelta(360, "D"))
        ])

    @classmethod
    def wfa_evaluation_profile_header(cls, oos_wf_results):
        oos_start = oos_wf_results[0].test.start.date()
        oos_end = oos_wf_results[-1].test.end.date()
        years = Decimal((oos_end - oos_start).days / 360)
        net_pl = sum(oos_wf.net_profit() for oos_wf in oos_wf_results)
        annualized_pl = net_pl / years
        total_trades = sum(oos_wf.total_trades() for oos_wf in oos_wf_results)
        annual_trades = total_trades / years
        avg_trade = net_pl / total_trades
        header = [
            ["Net P&L", net_pl],
            ["Annualized P&L", annualized_pl],
            ["Number of trades", total_trades],
            ["Avg. annual trades", annual_trades],
            ["Avg trade", avg_trade]
        ]
        return f"{oos_start.isoformat()} to {oos_end.isoformat()}" + "\n" +\
               tabulate(header, numalign="right", floatfmt=",.2f")

    @classmethod
    def wfa_evaluation_profile_trade_analysis(cls, oos_wf_results):
        max_win = max((oos_wf.max_win_trade() for oos_wf in oos_wf_results), key=lambda t: t["ProfitLoss"])
        max_loss = min((oos_wf.max_loss_trade() for oos_wf in oos_wf_results), key=lambda t: t["ProfitLoss"])
        min_win = min((oos_wf.min_win_trade() for oos_wf in oos_wf_results), key=lambda t: t["ProfitLoss"])
        min_loss = max((oos_wf.min_loss_trade() for oos_wf in oos_wf_results), key=lambda t: t["ProfitLoss"])

        all_wins = pd.concat([oos_wf.winning_trades_df() for oos_wf in oos_wf_results], ignore_index=True, copy=False)
        all_losses = pd.concat([oos_wf.losing_trades_df() for oos_wf in oos_wf_results], ignore_index=True, copy=False)

        avg_win = all_wins["ProfitLoss"].mean()
        avg_win_dur = all_wins["Duration"].mean()
        stdev_win = all_wins["ProfitLoss"].std()
        stdev_win_dur = all_wins["Duration"].std()

        avg_loss = all_losses["ProfitLoss"].mean()
        avg_loss_dur = all_losses["Duration"].mean()
        stdev_loss = all_losses["ProfitLoss"].std()
        stdev_loss_dur = all_losses["Duration"].std()

        trade_analysis_header = ["Analysis of Trades", "Price Wins", "Time", "Price Losses", "Time"]
        trade_analysis = [
            ["Maximum", max_win["ProfitLoss"], max_win["Duration"], max_loss["ProfitLoss"], max_loss["Duration"]],
            ["Minimum", min_win["ProfitLoss"], min_win["Duration"], min_loss["ProfitLoss"], min_loss["Duration"]],
            ["Average", avg_win, avg_win_dur.round("1s"), avg_loss, avg_loss_dur.round("1s")],
            ["StDev", stdev_win, stdev_win_dur.round("1s"), stdev_loss, stdev_loss_dur.round("1s")],
            ["+1 StDev", avg_win + stdev_win, (avg_win_dur + stdev_win_dur).round("1s"), avg_loss + stdev_loss, (avg_loss_dur + stdev_loss_dur).round("1s")],
            ["-1 StDev", avg_win - stdev_win, (avg_win_dur - stdev_win_dur).round("1s"), avg_loss - stdev_loss, (avg_loss_dur - stdev_loss_dur).round("1s")],
        ]
        return tabulate(trade_analysis, headers=trade_analysis_header, numalign="right", floatfmt=",.2f")

    @classmethod
    def wfa_evaluation_profile_equity_swings(cls, oos_wf_results):
        # https://stackoverflow.com/questions/31543697/how-to-split-pandas-dataframe-based-on-difference-of-values-in-a-column
        # Best to debug this logic in a Jupyter notebook
        df = pd.concat(oos_wf.daily_returns() for oos_wf in oos_wf_results).to_frame()
        df.columns = ["pct"]
        df["sign"] = np.sign(df["pct"])

        directional = df[df["sign"] != 0]
        df["run"] = (directional["sign"] != directional["sign"].shift()).cumsum()
        df["run"].fillna(method="ffill", inplace=True)

        runs = df.reset_index().groupby("run").agg({
            "x": lambda g: g.max() - g.min() + pd.Timedelta(1, "D"),
            "pct": lambda g: (g + 1).prod()
        })
        ups = runs[runs["pct"] > 1]
        downs = runs[runs["pct"] < 1]

        max_up = ups.loc[ups["pct"].idxmax()]
        min_up = ups.loc[ups["pct"].idxmin()]
        max_down = downs.loc[downs["pct"].idxmin()]
        min_down = downs.loc[downs["pct"].idxmax()]
        avg_up = ups["pct"].mean()
        avg_up_dur = ups["x"].mean()
        avg_down = downs["pct"].mean()
        avg_down_dur = downs["x"].mean()
        stdev_up = ups["pct"].std()
        stdev_up_dur = ups["x"].std()
        stdev_down = downs["pct"].std()
        stdev_down_dur = downs["x"].std()

        equity_analysis_header = ["Analysis of Equity Swings", "Equity Run-up", "Time", "Equity Drawdown", "Time"]
        equity_analysis = [
            ["Maximum", max_up[1], pd.Timedelta(max_up[0]), max_down[1], pd.Timedelta(max_down[0])],
            ["Minimum", min_up[1], pd.Timedelta(min_up[0]), min_down[1], pd.Timedelta(min_down[0])],
            ["Average", avg_up, avg_up_dur.round("1s"), avg_down, avg_down_dur.round("1s")],
            ["StDev", stdev_up, stdev_up_dur.round("1s"), stdev_down, stdev_down_dur.round("1s")],
            ["+1 StDev", avg_up + stdev_up, (avg_up_dur + stdev_up_dur).round("1s"), avg_down + stdev_down, (avg_down_dur + stdev_down_dur).round("1s")],
            ["-1 StDev", avg_up - stdev_up, (avg_up_dur - stdev_up_dur).round("1s"), avg_down - stdev_down, (avg_down_dur - stdev_down_dur).round("1s")],
        ]
        return tabulate(equity_analysis, headers=equity_analysis_header, numalign="right", floatfmt=",.2f")

    @classmethod
    def wfa_evaluation_rolling_window(cls, oos_wf_results, windowsize):
        daily_returns = pd.concat(oos_wf.daily_returns() for oos_wf in oos_wf_results)

        def window_return(rows):
            if rows.index[-1] - daily_returns.index[0] >= windowsize:
                return (rows + 1).prod()
            else:
                return np.nan

        rolling_windows = daily_returns.rolling(windowsize, closed="neither").apply(window_return).dropna()
        up_windows = rolling_windows[rolling_windows > 1]
        down_windows = rolling_windows[rolling_windows < 1]
        header = [f"Rolling Window: {windowsize}", "Equity Run-Up", "Equity Drawdown"]
        body = [
            ["Maximum", up_windows.max(), down_windows.min()],
            ["Minimum", up_windows.min(), down_windows.max()],
            ["Average", up_windows.mean(), down_windows.mean()],
            ["StDev", up_windows.std(), down_windows.std()],
            ["+1 StDev", up_windows.mean() + up_windows.std(), down_windows.mean() + down_windows.std()],
            ["-1 StDev", up_windows.mean() - up_windows.std(), down_windows.mean() - down_windows.std()],
        ]
        return tabulate(body, headers=header, numalign="right", floatfmt=",.2f")

    @staticmethod
    def load_sfp():
        sfp = pd.read_csv(r"C:\Users\karth\OneDrive\Documents\Data\SHARADAR_SFP_233bb9e7eb4cd1e64e2ab1117cd9dffa.zip")
        sfp["date"] = pd.to_datetime(sfp["date"])
        sfp["date"] = sfp["date"].dt.tz_localize("America/New_York")

        sfp.set_index(["ticker", "date"], inplace=True)
        sfp.sort_index(inplace=True)
        return sfp

    @staticmethod
    def benchmark_daily_rets(sfp, benchmark="SPY"):
        return sfp.loc[benchmark]["close"].pct_change().dropna()

    @staticmethod
    def benchmark_monthly_rets(sfp, benchmark="SPY"):
        return sfp.loc[benchmark]["close"].resample("1M").last().pct_change().dropna()
