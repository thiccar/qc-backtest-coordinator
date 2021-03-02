"""Code that can be shared among Jupyter notebooks"""
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

    def results(self):
        for t in self.tests():
            # Allows us to run this while backtests are running to see intermediate results
            if t.state == TestState.RUNNING:
                continue
            result = self.cio.read_test_result(t)

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
    def group_wfa_results(cls, results):
        oos_sorted = list(sorted([r for r in results if "_oos_" in r.test.name], key=lambda r: r.test.start))

        combined_start = oos_sorted[0].test.start
        combined_end = oos_sorted[-1].test.end
        oos_combined = filter(
            lambda r: (r.test.start, r.test.end) == (combined_start, combined_end),
            oos_sorted)
        oos_combined = next(oos_combined, None)
        if oos_combined:
            oos_sorted.remove(oos_combined)

        opt_grouped = itertools.groupby([r for r in results if "_opt_" in r.test.name], key=lambda r: r.test.start)
        opt_sorted = sorted([(d, list(g)) for (d, g) in opt_grouped], key=lambda tup: tup[0])
        wfa_results = [(opt_results, oos_result) for ((_, opt_results), oos_result) in zip(opt_sorted, oos_sorted)]

        for (opt_results, oos_result) in wfa_results:
            assert oos_result.test.start == opt_results[0].test.end + timedelta(1)

        # TODO: Consider creating a class to group together this information
        return wfa_results, oos_combined

    @classmethod
    def wfa_summary_statistics(cls, wfa_results, oos_combined, objective_fn):
        """Return a formatted table with WFA summary statistics a la Pardo (see https://pasteboard.co/JMmVgHk.png)"""
        headers = ["", "Opt\nStart", "Opt\nEnd", "Best Opt P/L\nAnnualized", "Best Opt\nMax Drawdown", "OOS\nStart",
                   "OOS\nEnd", "Net P/L", "Net P/L\nAnnualized", "Max\nDrawdown", "ROMAD\nAnnualized", "Win %",
                   "Walk-Forward\nEfficiency", "Sharpe\nRatio", "PSR", "OOS Params"]
        table = []
        for (opt_results, oos_result) in wfa_results:
            best_opt = max(opt_results, key=objective_fn)
            row = [
                None,
                best_opt.test.start.date(),
                best_opt.test.end.date(),
                best_opt.annualized_net_profit(),
                round(best_opt.drawdown(), 3),
                oos_result.test.start.date(),
                oos_result.test.end.date(),
                oos_result.net_profit(),
                oos_result.annualized_net_profit(),
                oos_result.drawdown(),
                oos_result.annualized_return_over_max_drawdown(),
                oos_result.win_rate(),
                oos_result.annualized_net_profit() / best_opt.annualized_net_profit(),
                oos_result.sharpe_ratio(),
                oos_result.probabilistic_sharpe_ratio(),
                json.dumps(oos_result.test.params),
            ]
            table.append(row)

        mean_opt_annualized_pl = statistics.mean(r[3] for r in table)
        max_opt_drawdown = max(r[4] for r in table)
        summary = [
            "Aggregate", "", "",
            mean_opt_annualized_pl,
            max_opt_drawdown,
            wfa_results[0][1].test.start.date(), wfa_results[-1][1].test.end.date(),
            sum(r[7] for r in table),
            statistics.mean(r[8] for r in table),
            max(r[9] for r in table),
            statistics.mean(r[10] for r in table),
            statistics.mean(r[11] for r in table),
            statistics.mean(r[12] for r in table),
            statistics.mean(r[13] for r in table),
            statistics.mean(r[14] for r in table),
        ]
        table.append(summary)

        if oos_combined:
            combined = [
                "Combined", "", "",
                mean_opt_annualized_pl,
                max_opt_drawdown,
                wfa_results[0][1].test.start.date(), wfa_results[-1][1].test.end.date(),
                oos_combined.net_profit(),
                oos_combined.annualized_net_profit(),
                oos_combined.drawdown(),
                oos_combined.annualized_return_over_max_drawdown(),
                oos_combined.win_rate(),
                oos_combined.annualized_net_profit() / mean_opt_annualized_pl,
                oos_combined.sharpe_ratio(),
                oos_combined.probabilistic_sharpe_ratio(),
            ]
            table.append(combined)
        return tabulate(table, headers=headers, numalign="right", floatfmt=",.2f")

    @classmethod
    def stitch_oos_equity_curve(cls, wfa_results):
        previous = None
        combined_xs = []
        combined_ys = []
        for (_, oos_result) in wfa_results:
            e = oos_result.equity_timeseries()
            xs = [datetime.fromtimestamp(v["x"]) for v in e]
            ys = [v["y"] for v in e]

            initial = ys[0]
            if not previous:
                previous = initial
            adj = previous / initial
            # print(f"{s.test.params['start']} {previous} {initial} {adj}")
            adj_ys = [adj * y for y in ys]
            previous = adj_ys[-1]

            combined_xs.extend(xs)
            combined_ys.extend(adj_ys)

        duration = combined_xs[-1] - combined_xs[0]
        total_return = combined_ys[-1] / combined_ys[0]
        annualized_return = (total_return - 1) ** (1 / (duration / timedelta(365)))
        print(f"Stitched: Duration = {duration} Total Return = {total_return - 1}"
              f"Annualized Return = {annualized_return - 1}")

        return combined_xs, combined_ys

    @classmethod
    def total_trades_bar_graph(cls, fig, ax, results, label_fn=None):
        """Sanity check - Graph number of trades to look for outliers"""
        xs = [label_fn(r) for r in results] if label_fn else None
        ys = [r.total_trades() for r in results]

        ax.bar(x=range(len(ys)), height=ys, color="green", tick_label=xs)
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def oos_return_drawdown_bar_graph(cls, fig, ax, wfa_results):
        """Graph individual OOS test metrics over time, see evolution. Followed example of
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
        """
        labels = [oos_result.test.start.date() for (_, oos_result) in wfa_results]
        tr = [oos_result.total_return() for (_, oos_result) in wfa_results]
        dd = [oos_result.drawdown() for (_, oos_result) in wfa_results]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, tr, width, label="Total Return", color="green")
        ax.bar(x + width / 2, dd, width, label="Drawdown", color="red")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.setp(fig.axes[0].get_xticklabels(), fontsize=10, rotation='vertical')

    @classmethod
    def top_btm_days(cls, results, n=10):
        top = []
        btm = []
        for result in results:
            df = pd.DataFrame(result.bt_result["charts"]["Strategy Equity"]["Series"]["Equity"]["Values"])
            df["ts"] = pd.to_datetime(df["x"], unit="s").apply(
                lambda ts: ts.tz_localize("UTC").tz_convert("America/New_York"))
            r = df.resample("D", on="ts").last()
            r["prev_y"] = r.shift(1)["y"]
            r.dropna(inplace=True)
            r["daily_return"] = r["y"] / r["prev_y"]
            for tup in r.itertuples():
                d = tup._asdict()
                d["test"] = result.test.name

                top.append(d)
                top.sort(key=lambda x: x["daily_return"], reverse=True)
                top = top[:n]

                btm.append(d)
                btm.sort(key=lambda x: x["daily_return"])
                btm = btm[:n]

        return top, btm

    @classmethod
    def top_btm_trades(cls, results, n=10):
        """Default TradeBuilder does not track splits leading to extreme values at top/bottom, see
        https://github.com/QuantConnect/Lean/issues/5325
        """
        top = []
        btm = []

        for result in results:
            for trade in result.closed_trades:
                trade["test"] = result.test.name
                top.append(trade)
                top.sort(key=lambda t: t["ProfitLoss"], reverse=True)
                top = top[:n]

                btm.append(trade)
                btm.sort(key=lambda t: t["ProfitLoss"])
                btm = btm[:n]

        return top, btm

    @classmethod
    def wfa_equity_curve_plot(cls, fig, ax, wfa_results, oos_combined, label=""):
        """Graph equity curve stitched together from individual OOS tests.  NOTE: It may look like the stitched curve
        starts later than the single curve, that is because they perfectly overlap for the duration of the first OOS
        test.
        """
        combined_xs, combined_ys = cls.stitch_oos_equity_curve(wfa_results)

        ax.plot(combined_xs, combined_ys, label=f"{label}_stitched" if label else "stitched")
        ax.set_yscale("log")

        if oos_combined:
            print(f"Combined: Duration = {oos_combined.duration()} Total Return = {oos_combined.total_return()} "
                  f"Annualized Return = {oos_combined.compounding_annual_return()}")
            e = oos_combined.bt_result["charts"]["Strategy Equity"]["Series"]["Equity"]["Values"]
            xs = [datetime.fromtimestamp(v["x"]) for v in e]
            ys = [v["y"] for v in e]
            ax.plot(xs, ys, label=f"{label}_continuous" if label else "continuous")
        ax.legend()

    @classmethod
    def wfa_params_plot(cls, fig, wfa_results):
        """Graph change in parameters used"""
        params_keys = list(k for k in wfa_results[0][1].test.params.keys() if k not in ["start", "end"])
        xs = [oos_result.test.start for (_, oos_result) in wfa_results]
        axs = fig.subplots(len(params_keys), 1)
        for (ax, key) in zip(axs, params_keys):
            ys = [oos_result.test.params[key] for (_, oos_result) in wfa_results]
            ax.plot(xs, ys, label=key)
            ax.set_title(key)

    @classmethod
    def wfa_evaluation_profile(cls, wfa_results, oos_combined):
        """Produce a table like https://pasteboard.co/JO2OL6Y.png (Table 14.1 from "The Evaluation and Optimization of
        Trading Strategies"), as well as similar stats for rolling windows.
        """
        return "\n".join([
            cls.wfa_evaluation_profile_header(wfa_results, oos_combined),
            cls.wfa_evaluation_profile_trade_analysis(wfa_results, oos_combined),
            cls.wfa_evaluation_profile_equity_swings(wfa_results, oos_combined),
            cls.wfa_evaluation_rolling_window(wfa_results, oos_combined, pd.Timedelta(7, "D")),
            cls.wfa_evaluation_rolling_window(wfa_results, oos_combined, pd.Timedelta(30, "D")),
            cls.wfa_evaluation_rolling_window(wfa_results, oos_combined, pd.Timedelta(90, "D")),
            cls.wfa_evaluation_rolling_window(wfa_results, oos_combined, pd.Timedelta(180, "D")),
            cls.wfa_evaluation_rolling_window(wfa_results, oos_combined, pd.Timedelta(360, "D"))
        ])

    @classmethod
    def wfa_evaluation_profile_header(cls, wfa_results, oos_combined):
        oos_results = [oos_result for (_, oos_result) in wfa_results]
        oos_start = oos_results[0].test.start.date()
        oos_end = oos_results[-1].test.end.date()
        years = Decimal((oos_end - oos_start).days / 360)
        net_pl = sum(oos_result.net_profit() for oos_result in oos_results)
        annualized_pl = net_pl / years
        total_trades = sum(oos_result.total_trades() for oos_result in oos_results)
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
    def wfa_evaluation_profile_trade_analysis(cls, wfa_results, oos_combined):
        oos_results = [oos_result for (_, oos_result) in wfa_results]
        max_win = max((oos_result.max_win_trade() for oos_result in oos_results), key=lambda t: t["ProfitLoss"])
        max_loss = min((oos_result.max_loss_trade() for oos_result in oos_results), key=lambda t: t["ProfitLoss"])
        min_win = min((oos_result.min_win_trade() for oos_result in oos_results), key=lambda t: t["ProfitLoss"])
        min_loss = max((oos_result.min_loss_trade() for oos_result in oos_results), key=lambda t: t["ProfitLoss"])

        all_wins = list(itertools.chain.from_iterable(oos_result.winning_trades() for oos_result in oos_results))
        all_losses = list(itertools.chain.from_iterable(oos_result.losing_trades() for oos_result in oos_results))
        avg_win = np.mean([t["ProfitLoss"] for t in all_wins])
        avg_win_dur = pd.Timedelta(np.mean([TestResult.trade_duration(t).total_seconds() for t in all_wins]), "s")
        stdev_win = np.std([t["ProfitLoss"] for t in all_wins])
        stdev_win_dur = pd.Timedelta(np.std([TestResult.trade_duration(t).total_seconds() for t in all_wins]), "s")
        avg_loss = np.mean([t["ProfitLoss"] for t in all_losses])
        avg_loss_dur = pd.Timedelta(np.mean([TestResult.trade_duration(t).total_seconds() for t in all_losses]), "s")
        stdev_loss = np.std([t["ProfitLoss"] for t in all_losses])
        stdev_loss_dur = pd.Timedelta(np.std([TestResult.trade_duration(t).total_seconds() for t in all_losses]), "s")

        trade_analysis_header = ["Analysis of Trades", "Price Wins", "Time", "Price Losses", "Time"]
        trade_analysis = [
            ["Maximum", max_win["ProfitLoss"], pd.Timedelta(TestResult.trade_duration(max_win)), max_loss["ProfitLoss"], pd.Timedelta(TestResult.trade_duration(max_loss))],
            ["Minimum", min_win["ProfitLoss"], pd.Timedelta(TestResult.trade_duration(min_win)), min_loss["ProfitLoss"], pd.Timedelta(TestResult.trade_duration(min_loss))],
            ["Average", avg_win, avg_win_dur.round("1s"), avg_loss, avg_loss_dur.round("1s")],
            ["StDev", stdev_win, stdev_win_dur.round("1s"), stdev_loss, stdev_loss_dur.round("1s")],
            ["+1 StDev", avg_win + stdev_win, (avg_win_dur + stdev_win_dur).round("1s"), avg_loss + stdev_loss, (avg_loss_dur + stdev_loss_dur).round("1s")],
            ["-1 StDev", avg_win - stdev_win, (avg_win_dur - stdev_win_dur).round("1s"), avg_loss - stdev_loss, (avg_loss_dur - stdev_loss_dur).round("1s")],
        ]
        return tabulate(trade_analysis, headers=trade_analysis_header, numalign="right", floatfmt=",.2f")

    @classmethod
    def wfa_evaluation_profile_equity_swings(cls, wfa_results, oos_combined):
        ups = []
        downs = []
        cur = []
        for d in oos_combined.equity_timeseries():
            if len(cur) > 2:
                sign = np.sign(d["y"] - cur[-1]["y"])
                cur_sign = np.sign(cur[-1]["y"] - cur[0]["y"])
                if not(sign == 0 or cur_sign == 0 or sign == cur_sign):
                    run = (datetime.fromtimestamp(cur[-1]["x"]) - datetime.fromtimestamp(cur[0]["x"]),
                           cur[-1]["y"] / cur[0]["y"])
                    (ups if cur_sign > 0 else downs).append(run)
                    cur = []
            cur.append(d)

        max_up = max(ups, key=lambda tup: tup[1])
        min_up = min(ups, key=lambda tup: tup[1])
        max_down = min(downs, key=lambda tup: tup[1])
        min_down = max(downs, key=lambda tup: tup[1])
        avg_up = np.mean([tup[1] for tup in ups])
        avg_up_dur = pd.Timedelta(np.mean([tup[0].total_seconds() for tup in ups]), "s")
        avg_down = np.mean([tup[1] for tup in downs])
        avg_down_dur = pd.Timedelta(np.mean([tup[0].total_seconds() for tup in downs]), "s")
        stdev_up = np.std([tup[1] for tup in ups])
        stdev_up_dur = pd.Timedelta(np.std([tup[0].total_seconds() for tup in ups]), "s")
        stdev_down = np.std([tup[1] for tup in downs])
        stdev_down_dur = pd.Timedelta(np.std([tup[0].total_seconds() for tup in downs]), "s")

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
    def wfa_evaluation_rolling_window(cls, wfa_results, oos_combined, windowsize):
        df = oos_combined.equity_timeseries_df()

        def window_return(rows):
            if rows.index[-1] - df.index[0] >= windowsize:
                return rows[-1] / rows[0]
            else:
                return np.nan

        rolling_windows = df.rolling(windowsize, closed="neither").apply(window_return).dropna()
        up_windows = rolling_windows[rolling_windows["y"] > 1]
        down_windows = rolling_windows[rolling_windows["y"] < 1]
        header = [f"Rolling Window: {windowsize}", "Equity Run-Up", "Equity Drawdown"]
        body = [
            ["Maximum", up_windows["y"].max(), down_windows["y"].min()],
            ["Minimum", up_windows["y"].min(), down_windows["y"].max()],
            ["Average", up_windows["y"].mean(), down_windows["y"].mean()],
            ["StDev", up_windows["y"].std(), down_windows["y"].std()],
            ["+1 StDev", up_windows["y"].mean() + up_windows["y"].std(), down_windows["y"].mean() + down_windows["y"].std()],
            ["-1 StDev", up_windows["y"].mean() - up_windows["y"].std(), down_windows["y"].mean() - down_windows["y"].std()],
        ]
        return tabulate(body, headers=header, numalign="right", floatfmt=",.2f")


