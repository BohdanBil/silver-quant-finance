"""Monte Carlo silver price forecast with VaR / CVaR risk metrics."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data import (
    GBMParams,
    calibrate_gbm,
    calibrate_gbm_bootstrap,
    calibrate_gbm_ewma,
    calibrate_gbm_garch,
    fetch_silver_prices,
    log_returns_of,
)
from monte_carlo import simulate_gbm_jump_paths, simulate_gbm_paths
from risk import backtest_var, var_cvar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "model x dist are independent axes: --model gbm --dist t reproduces a plain "
            "fat-tailed GBM ('t' in the earlier sketch); --model gbm-jump adds Merton jumps "
            "on top, with or without --dist t."
        ),
    )
    p.add_argument("--period", default="5y", help="history window for calibration, e.g. 5y, 2y, 1y")
    p.add_argument("--horizon-days", type=int, default=21, help="trading days to simulate forward")
    p.add_argument("--n-sims", type=int, default=20_000, help="number of Monte Carlo paths")
    p.add_argument("--position-oz", type=float, default=1000.0, help="position size in troy ounces")
    p.add_argument("--confidence", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="silver_forecast.png", help="path to save the plot")
    p.add_argument("--no-show", action="store_true", help="save the plot but don't open a window")

    p.add_argument("--calibration", choices=["point", "ewma", "garch", "bootstrap"], default="point",
                    help="how to estimate mu/sigma from history (default: point = flat 5y mean/std)")
    p.add_argument("--ewma-lambda", type=float, default=0.94, help="EWMA decay factor for --calibration ewma")
    p.add_argument("--block-size", type=int, default=21,
                    help="block length (trading days) for --calibration bootstrap")

    p.add_argument("--model", choices=["gbm", "gbm-jump"], default="gbm")
    p.add_argument("--dist", choices=["normal", "t"], default="normal", help="innovation distribution")
    p.add_argument("--t-df", type=float, default=5.0, help="degrees of freedom when --dist t")
    p.add_argument("--antithetic", action="store_true", help="use antithetic variates (variance reduction)")

    p.add_argument("--jump-intensity", type=float, default=2.0, help="expected jumps/year (--model gbm-jump)")
    p.add_argument("--jump-mean", type=float, default=-0.02, help="mean log jump size (--model gbm-jump)")
    p.add_argument("--jump-std", type=float, default=0.05, help="std of log jump size (--model gbm-jump)")

    return p.parse_args()


def calibrate(prices, args) -> GBMParams:
    point = calibrate_gbm(prices)

    if args.calibration == "point":
        print(f"calibration      = point (flat {args.period} mean/std)")
        return point

    if args.calibration == "ewma":
        params = calibrate_gbm_ewma(prices, lam=args.ewma_lambda)
        print(f"calibration      = EWMA (lambda={args.ewma_lambda}) sigma, point mu")
        print(f"  point sigma={point.sigma:.4%}  ->  EWMA sigma={params.sigma:.4%}")
        return params

    if args.calibration == "garch":
        try:
            params = calibrate_gbm_garch(prices)
        except RuntimeError as e:
            print(f"  GARCH(1,1) fit failed ({e}); falling back to EWMA")
            return calibrate_gbm_ewma(prices, lam=args.ewma_lambda)
        print(f"calibration      = GARCH(1,1) sigma, point mu")
        print(f"  point sigma={point.sigma:.4%}  ->  GARCH sigma={params.sigma:.4%}")
        return params

    # bootstrap: one (mu, sigma) draw per simulated path, so parameter
    # uncertainty (mu especially) is baked into the terminal distribution
    # instead of collapsed to a single number.
    boot = calibrate_gbm_bootstrap(prices, n_boot=args.n_sims, block_size=args.block_size, seed=args.seed)
    print(f"calibration      = block bootstrap ({args.n_sims:,} draws, block={args.block_size}d)")
    print(f"  mu:    point={point.mu:.4%}  bootstrap mean={boot['mu'].mean():.4%}  "
          f"std={boot['mu'].std():.4%}  5-95pct=[{boot['mu'].quantile(.05):.4%}, {boot['mu'].quantile(.95):.4%}]")
    print(f"  sigma: point={point.sigma:.4%}  bootstrap mean={boot['sigma'].mean():.4%}  "
          f"std={boot['sigma'].std():.4%}")
    return GBMParams(s0=point.s0, mu=boot["mu"].values, sigma=boot["sigma"].values)


def simulate(params: GBMParams, args) -> np.ndarray:
    if args.model == "gbm":
        return simulate_gbm_paths(
            params, args.horizon_days, args.n_sims,
            dist=args.dist, t_df=args.t_df, antithetic=args.antithetic, seed=args.seed,
        )
    return simulate_gbm_jump_paths(
        params, args.horizon_days, args.n_sims,
        jump_intensity=args.jump_intensity, jump_mean=args.jump_mean, jump_std=args.jump_std,
        dist=args.dist, t_df=args.t_df, antithetic=args.antithetic, seed=args.seed,
    )


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period)
    print(f"Calibrated from {len(prices)} daily prices")
    params = calibrate(prices, args)
    s0 = params.s0
    print(f"  spot (s0) = {s0:.4f}")

    print(f"\nmodel = {args.model}, dist = {args.dist}" + (f" (df={args.t_df})" if args.dist == "t" else "")
          + (f", antithetic" if args.antithetic else ""))
    paths = simulate(params, args)
    terminal_prices = paths[:, -1]

    pnl = (terminal_prices - s0) * args.position_oz
    print(f"\n{args.horizon_days}-day Monte Carlo forecast ({args.n_sims:,} paths)")
    print(f"  mean terminal price = {terminal_prices.mean():.4f}")
    print(f"  5th / 95th pct      = {np.percentile(terminal_prices, 5):.4f} / "
          f"{np.percentile(terminal_prices, 95):.4f}")

    daily_returns = log_returns_of(prices)
    worst_daily_log_move = float(daily_returns.min())
    worst_daily_simple_move = np.exp(worst_daily_log_move) - 1
    worst_daily_pnl = s0 * worst_daily_simple_move * args.position_oz
    frac_worse = float(np.mean(pnl < worst_daily_pnl))
    print(f"\nSanity check: worst historical 1-day move over {args.period} was {worst_daily_simple_move:.2%} "
          f"(${-worst_daily_pnl:,.0f} on this position).")
    print(f"  {frac_worse:.2%} of simulated {args.horizon_days}-day outcomes lose more than that "
          f"single worst historical day.")

    print(f"\nRisk metrics on a {args.position_oz:,.0f} oz long position:")
    for c in args.confidence:
        metrics = var_cvar(pnl, confidence=c)
        print(f"  {c:.0%} VaR  = ${metrics.var:,.2f}")
        print(f"  {c:.0%} CVaR = ${metrics.cvar:,.2f}")

        bt = backtest_var(prices, var_threshold=metrics.var, position_oz=args.position_oz,
                           horizon_days=args.horizon_days, confidence=c)
        verdict = "REJECTED (miscalibrated)" if bt.reject_at_5pct else "not rejected"
        print(f"    backtest: {bt.n_exceptions}/{bt.n_obs} historical windows breached VaR "
              f"({bt.exception_rate:.2%} vs {bt.expected_rate:.2%} expected), "
              f"Kupiec p={bt.p_value:.4f} -> {verdict}")

    plot_results(paths, pnl, args)


def plot_results(paths, pnl, args) -> None:
    fig, (ax_paths, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    n_show = min(200, paths.shape[0])
    ax_paths.plot(paths[:n_show].T, linewidth=0.5, alpha=0.3, color="tab:blue")
    ax_paths.plot(paths.mean(axis=0), linewidth=2, color="black", label="mean path")
    ax_paths.set_title(f"Simulated silver price paths ({args.horizon_days}d)")
    ax_paths.set_xlabel("trading day")
    ax_paths.set_ylabel("price")
    ax_paths.legend()

    ax_hist.hist(pnl, bins=80, color="tab:blue", alpha=0.7)
    for c in args.confidence:
        metrics = var_cvar(pnl, confidence=c)
        ax_hist.axvline(-metrics.var, color="tab:red", linestyle="--",
                         label=f"{c:.0%} VaR = ${metrics.var:,.0f}")
        ax_hist.axvline(-metrics.cvar, color="tab:orange", linestyle=":",
                         label=f"{c:.0%} CVaR = ${metrics.cvar:,.0f}")
    ax_hist.set_title(f"P&L distribution ({args.position_oz:,.0f} oz position)")
    ax_hist.set_xlabel("P&L ($)")
    ax_hist.legend(fontsize=8)

    dist_label = args.dist if args.dist == "normal" else f"{args.dist} (df={args.t_df})"
    fig.suptitle(f"model={args.model}  dist={dist_label}  calibration={args.calibration}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
