"""CLI: out-of-sample VaR backtest of every risk-model variant main.py offers.

For each calibration x process x shock distribution, re-calibrates on a
rolling window at every test date (using only data available then),
forecasts the VaR, and checks the realized losses with Kupiec's coverage
test and Christoffersen's independence test. This is the evidence behind
any claim that one variant is better calibrated than another.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from data import fetch_silver_prices
from plot_style import BLUE, CRITICAL, INK_SECONDARY, MUTED, apply_style
from risk_model import CALIBRATIONS, DISTS, PROCESS_MODELS, all_specs, backtest_variants


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="10y", help="price history to backtest over")
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--horizon-days", type=int, default=1,
                   help="VaR horizon; windows never overlap, so longer horizons give far fewer test points")
    p.add_argument("--confidence", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--window", type=int, default=504, help="rolling calibration window (trading days)")
    p.add_argument("--n-sims", type=int, default=5_000, help="Monte Carlo paths per test date")
    p.add_argument("--refit-every", type=int, default=21,
                   help="refit GARCH / bootstrap estimates every N trading days (cheap parts update daily)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrations", nargs="+", choices=CALIBRATIONS, default=list(CALIBRATIONS))
    p.add_argument("--models", nargs="+", choices=PROCESS_MODELS, default=list(PROCESS_MODELS))
    p.add_argument("--dists", nargs="+", choices=DISTS, default=list(DISTS))
    p.add_argument("--t-df", type=float, default=5.0)
    p.add_argument("--plot", default=None, help="save a chart of exception rates per variant to this path")
    return p.parse_args()


def plot_exception_rates(table: pd.DataFrame, confidence_levels: list[float], path: str) -> None:
    apply_style()
    labels = table["model"].tolist()
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, len(confidence_levels), figsize=(5.2 * len(confidence_levels), 0.32 * len(labels) + 1.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, confidence_levels):
        rate = table[f"rate_{c:.0%}"].to_numpy()
        rejected = table[f"rejected_{c:.0%}"].to_numpy()
        expected = 1 - c
        ax.axvline(expected, color=INK_SECONDARY, linewidth=1)
        ax.text(expected, len(labels) - 0.4, f" expected {expected:.0%}", color=INK_SECONDARY, fontsize=8, va="bottom")
        ax.scatter(rate[~rejected], y[~rejected], s=42, color=BLUE, zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(rate[rejected], y[rejected], s=48, color=CRITICAL, marker="X", zorder=3, edgecolors="white",
                   linewidths=1)
        ax.set_title(f"{c:.0%} VaR: exception rate", loc="left")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.1%}"))
        ax.set_xlim(0, max(rate.max(), expected) * 1.25)
        ax.set_ylim(-0.7, len(labels) - 0.1)
    axes[0].set_yticks(y, labels)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="not rejected"),
        Line2D([], [], marker="X", linestyle="", color=CRITICAL, markersize=7,
               label="rejected (Kupiec or clustering, 5%)"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=8)
    fig.text(0.01, 0.01, "Rolling out-of-sample backtest: each VaR uses only data before its date.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


def main() -> None:
    args = parse_args()
    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    n_tests = len(range(args.window, len(prices) - args.horizon_days, args.horizon_days))
    print(f"Backtesting {args.horizon_days}-day VaR on {len(prices)} prices "
          f"({prices.index[0].date()} to {prices.index[-1].date()}): {n_tests} non-overlapping test windows, "
          f"each calibrated on the previous {args.window} trading days")
    for c in args.confidence:
        print(f"  {c:.0%}: ~{n_tests * (1 - c):.1f} exceptions expected from a correctly calibrated model")

    started = time.perf_counter()

    def report(spec, results) -> None:
        nonlocal started
        verdicts = ", ".join(f"{c:.0%} {bt.verdict}" for c, bt in results.items())
        print(f"  {spec.label:<32} {verdicts}  ({time.perf_counter() - started:.0f}s)")
        started = time.perf_counter()

    specs = all_specs(args.calibrations, args.models, args.dists, t_df=args.t_df)
    table = backtest_variants(prices, specs, args.horizon_days, args.confidence, window=args.window,
                              n_sims=args.n_sims, refit_every=args.refit_every, seed=args.seed, on_result=report)
    shown = table.drop(columns=[c for c in table.columns
                                if c.startswith(("rejected_", "verdict_")) or c in ("calibration", "process", "dist")])
    print("\nException rates and test p-values (Kupiec: right number of exceptions? "
          "cluster: Christoffersen independence):")
    print(shown.to_string(index=False, formatters={
        **{c: "{:.2%}".format for c in shown.columns if c.startswith("rate_")},
        **{c: "{:.3f}".format for c in shown.columns if c.startswith(("kupiec_p_", "cluster_p_"))},
    }))

    passing = table.loc[~table[[f"rejected_{c:.0%}" for c in args.confidence]].any(axis=1), "model"]
    print(f"\nPass at every confidence level: {', '.join(passing) if len(passing) else 'none'}")
    print("(Each variant is tested separately at 5%, so with this many variants one or two "
          "verdicts can flip by chance; look for patterns across variants, not single cells.)")

    if args.plot:
        plot_exception_rates(table, args.confidence, args.plot)


if __name__ == "__main__":
    main()
