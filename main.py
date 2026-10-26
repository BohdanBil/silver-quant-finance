"""Monte Carlo silver price risk: simulated paths, VaR / CVaR, and an out-of-sample VaR backtest."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from data import fetch_silver_prices, log_returns_of
from plot_style import BLUE, BLUE_RAMP, INK, INK_SECONDARY, ORANGE, SURFACE, apply_style, thousands
from risk import CoverageTest, var_cvar
from risk_model import (
    CALIBRATIONS,
    DISTS,
    PROCESS_MODELS,
    CalibratedModel,
    RiskModelSpec,
    backtest_spec,
    calibrate,
    simulate_paths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "--model and --dist are independent: --model gbm --dist t is a fat-tailed GBM; "
            "--model gbm-jump adds Merton jumps estimated from the history, with or without --dist t."
        ),
    )
    p.add_argument("--period", default="5y", help="history window for calibration, e.g. 5y, 2y, 1y")
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--horizon-days", type=int, default=21, help="trading days to simulate forward")
    p.add_argument("--n-sims", type=int, default=20_000, help="number of Monte Carlo paths")
    p.add_argument("--position-oz", type=float, default=1000.0, help="position size in troy ounces")
    p.add_argument("--confidence", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="silver_forecast.png", help="path to save the plot")
    p.add_argument("--no-show", action="store_true", help="save the plot but don't open a window")

    p.add_argument("--calibration", choices=CALIBRATIONS, default="point",
                   help="how to estimate mu/sigma from history (default: point = flat mean/std over --period)")
    p.add_argument("--ewma-lambda", type=float, default=0.94, help="EWMA decay factor for --calibration ewma")
    p.add_argument("--block-size", type=int, default=21,
                   help="block length (trading days) for --calibration bootstrap")

    p.add_argument("--model", choices=PROCESS_MODELS, default="gbm")
    p.add_argument("--dist", choices=DISTS, default="normal", help="innovation distribution")
    p.add_argument("--t-df", type=float, default=5.0, help="degrees of freedom when --dist t")
    p.add_argument("--antithetic", action="store_true", help="use antithetic variates (variance reduction)")
    p.add_argument("--jump-threshold", type=float, default=4.0,
                   help="a daily return beyond this many local std devs counts as a jump (--model gbm-jump)")

    p.add_argument("--backtest-window", type=int, default=504,
                   help="rolling calibration window (trading days) for the out-of-sample VaR backtest")
    p.add_argument("--backtest-sims", type=int, default=5_000, help="Monte Carlo paths per backtest date")
    p.add_argument("--no-backtest", action="store_true", help="skip the out-of-sample VaR backtest")
    return p.parse_args()


def spec_from_args(args: argparse.Namespace) -> RiskModelSpec:
    return RiskModelSpec(calibration=args.calibration, model=args.model, dist=args.dist, t_df=args.t_df,
                         ewma_lambda=args.ewma_lambda, block_size=args.block_size,
                         jump_threshold=args.jump_threshold)


def describe_calibration(model: CalibratedModel, args: argparse.Namespace) -> None:
    params = model.params
    method = {
        "point": f"point (flat {args.period} mean/std)",
        "ewma": f"EWMA sigma (lambda={args.ewma_lambda})",
        "garch": f"GARCH(1,1) sigma, averaged over the {args.horizon_days}d horizon",
        "bootstrap": f"block bootstrap ({args.n_sims:,} draws, block={args.block_size}d)",
    }[args.calibration]
    print(f"calibration      = {method}")
    for note in model.notes:
        print(f"  {note}")

    if model.jumps is not None:
        j = model.jumps
        print(f"  jumps: {j.n_detected} detected ({j.intensity:.1f}/yr), mean log size {j.mean:+.2%}, "
              f"std {j.std:.2%}; sigma below is the diffusion vol with those days removed")

    if np.ndim(params.sigma) == 0:
        if args.calibration in ("ewma", "garch"):
            print(f"  sigma: flat={model.point_sigma:.2%}  ->  {args.calibration}={params.sigma:.2%}")
        else:
            print(f"  sigma: {params.sigma:.2%}")
        print(f"  mu (expected annual return): {params.mu:+.2%}")
    else:
        mu, sigma = np.asarray(params.mu), np.asarray(params.sigma)
        print(f"  mu:    mean={mu.mean():+.2%}  std={mu.std():.2%}  "
              f"5-95pct=[{np.quantile(mu, .05):+.2%}, {np.quantile(mu, .95):+.2%}]")
        print(f"  sigma: mean={sigma.mean():.2%}  std={sigma.std():.2%}  (flat: {model.point_sigma:.2%})")


def print_backtest(title: str, results: dict[float, CoverageTest]) -> None:
    print(title)
    for c, bt in results.items():
        print(f"    {c:.0%}: {bt.n_exceptions}/{bt.n_obs} exceptions ({bt.exception_rate:.2%} vs "
              f"{bt.expected_rate:.2%} expected)  Kupiec p={bt.kupiec_pvalue:.3f}  "
              f"clustering p={bt.independence_pvalue:.3f}  -> {bt.verdict}")


def run_backtests(prices, spec: RiskModelSpec, args: argparse.Namespace) -> None:
    window = args.backtest_window
    if len(prices) <= window + args.horizon_days:
        print(f"\n(skipping backtest: need more than {window + args.horizon_days} prices, have {len(prices)})")
        return

    print(f"\nOut-of-sample VaR backtest -- each forecast calibrated on the {window} trading days before it "
          f"and nothing later; losses in % of position value:")
    one_day = backtest_spec(prices, spec, 1, args.confidence, window=window, n_sims=args.backtest_sims,
                            seed=args.seed)
    print_backtest("  1-day VaR of the same model, every day:", one_day)

    if args.horizon_days > 1:
        h_day = backtest_spec(prices, spec, args.horizon_days, args.confidence, window=window,
                              n_sims=args.backtest_sims, seed=args.seed)
        n_windows = next(iter(h_day.values())).n_obs
        expected_99 = n_windows * (1 - max(args.confidence))
        print_backtest(f"  {args.horizon_days}-day VaR, non-overlapping windows "
                       f"(low power: ~{expected_99:.1f} exceptions expected at {max(args.confidence):.0%}):", h_day)


def main() -> None:
    args = parse_args()
    spec = spec_from_args(args)

    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    print(f"Calibrated from {len(prices)} daily prices, {prices.index[0].date()} to {prices.index[-1].date()}")
    model = calibrate(prices, spec, horizon_days=args.horizon_days, n_sims=args.n_sims, seed=args.seed)
    describe_calibration(model, args)
    s0 = model.params.s0
    print(f"  price (s0) = {s0:.4f}  (front-month futures)")

    print(f"\nmodel = {spec.label}" + (", antithetic" if args.antithetic else ""))
    paths = simulate_paths(model, args.horizon_days, args.n_sims, antithetic=args.antithetic, seed=args.seed)
    terminal_prices = paths[:, -1]

    pnl = (terminal_prices - s0) * args.position_oz
    print(f"\n{args.horizon_days}-day Monte Carlo forecast ({args.n_sims:,} paths)")
    print(f"  mean terminal price = {terminal_prices.mean():.4f}")
    print(f"  5th / 95th pct      = {np.percentile(terminal_prices, 5):.4f} / "
          f"{np.percentile(terminal_prices, 95):.4f}")

    daily_returns = log_returns_of(prices)
    worst_daily_simple_move = np.exp(float(daily_returns.min())) - 1
    worst_daily_pnl = s0 * worst_daily_simple_move * args.position_oz
    frac_worse = float(np.mean(pnl < worst_daily_pnl))
    print(f"\nSanity check: worst historical 1-day move over {args.period} was {worst_daily_simple_move:.2%} "
          f"(${-worst_daily_pnl:,.0f} on this position at today's price).")
    print(f"  {frac_worse:.2%} of simulated {args.horizon_days}-day outcomes lose more than that "
          f"single worst historical day.")

    notional = s0 * args.position_oz
    print(f"\nRisk metrics on a {args.position_oz:,.0f} oz long position (${notional:,.0f}):")
    for c in args.confidence:
        metrics = var_cvar(pnl, confidence=c)
        print(f"  {c:.0%} VaR  = ${metrics.var:,.2f}  ({metrics.var / notional:.2%})")
        print(f"  {c:.0%} CVaR = ${metrics.cvar:,.2f}  ({metrics.cvar / notional:.2%})")

    if not args.no_backtest:
        run_backtests(prices, spec, args)

    plot_results(paths, pnl, spec, args)


def plot_results(paths: np.ndarray, pnl: np.ndarray, spec: RiskModelSpec, args: argparse.Namespace) -> None:
    apply_style()
    fig, (ax_fan, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.6))

    days = np.arange(paths.shape[1])
    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    ax_fan.fill_between(days, q05, q95, color=BLUE_RAMP[100], linewidth=0, label="5-95% of paths")
    ax_fan.fill_between(days, q25, q75, color=BLUE_RAMP[250], linewidth=0, label="25-75%")
    ax_fan.plot(days, q50, color=BLUE_RAMP[600], linewidth=2, label="median")
    ax_fan.set_title(f"Simulated silver price, next {args.horizon_days} trading days", loc="left")
    ax_fan.set_xlabel("trading days ahead")
    ax_fan.set_ylabel("price ($/oz)")
    ax_fan.set_xlim(0, days[-1])
    ax_fan.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_fan.legend(loc="upper left")

    ax_hist.hist(pnl, bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    for c, color in zip(sorted(args.confidence), (ORANGE, INK, INK_SECONDARY)):
        metrics = var_cvar(pnl, confidence=c)
        ax_hist.axvline(-metrics.var, color=color, linewidth=1.5,
                        label=f"{c:.0%} VaR ${metrics.var:,.0f}  (CVaR ${metrics.cvar:,.0f})")
    ax_hist.set_title(f"{args.horizon_days}-day P&L, {args.position_oz:,.0f} oz long", loc="left")
    ax_hist.set_xlabel("P&L ($)")
    ax_hist.set_ylabel("simulated paths")
    ax_hist.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax_hist.legend(loc="upper right")

    fig.suptitle(f"model: {spec.label}   calibration window: {args.period}", fontsize=9, color=INK_SECONDARY,
                 x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
