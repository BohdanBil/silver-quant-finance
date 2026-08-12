"""CLI: price a European option on silver -- Black-Scholes closed form vs Monte Carlo.

Pulls live spot + a GARCH(1,1) vol estimate from silver's own price history
(data.py) and a live risk-free rate from FRED, the same data plumbing the
rest of this project uses -- so this isn't priced off made-up numbers by
default, though every input can be overridden.
"""

import argparse

from data import TRADING_DAYS_PER_YEAR, calibrate_gbm_garch, fetch_silver_prices
from forecast_data import fetch_fred_series
from option_pricing import black_scholes_greeks, black_scholes_price, monte_carlo_option_price

RISK_FREE_SERIES = "DGS3MO"  # 3-month T-bill: a reasonable proxy for a near-term risk-free rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strike", type=float, default=None, help="defaults to at-the-money (= spot)")
    p.add_argument("--maturity-days", type=int, default=63, help="trading days to expiry (default ~3 months)")
    p.add_argument("--option-type", choices=["call", "put"], default="call")
    p.add_argument("--rate", type=float, default=None,
                    help="annualized risk-free rate (e.g. 0.04); default: live 3M T-bill from FRED")
    p.add_argument("--sigma", type=float, default=None,
                    help="annualized vol (e.g. 0.40); default: GARCH(1,1) forecast from 5y silver history")
    p.add_argument("--n-sims", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--no-control-variate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period="5y")
    params = calibrate_gbm_garch(prices)
    s0 = params.s0
    sigma = args.sigma if args.sigma is not None else params.sigma
    strike = args.strike if args.strike is not None else s0

    if args.rate is not None:
        r = args.rate
    else:
        r = float(fetch_fred_series(RISK_FREE_SERIES).iloc[-1]) / 100

    T = args.maturity_days / TRADING_DAYS_PER_YEAR

    print(f"Silver {args.option_type}: spot=${s0:.4f}  strike=${strike:.4f}  "
          f"r={r:.3%}  sigma={sigma:.2%}  T={T:.3f}y ({args.maturity_days}d)")
    print("(sigma is a GARCH estimate from historical silver returns, used as a proxy for "
          "market-implied vol -- there's no silver options chain wired in to pull real implied vol from)")

    bs_price = black_scholes_price(s0, strike, r, sigma, T, option_type=args.option_type)
    greeks = black_scholes_greeks(s0, strike, r, sigma, T, option_type=args.option_type)

    print(f"\nBlack-Scholes price = ${bs_price:.4f}")
    print(f"  delta={greeks.delta:+.4f}  gamma={greeks.gamma:.4f}  "
          f"vega={greeks.vega / 100:.4f} (per 1 vol pt)  "
          f"theta={greeks.theta / 365:.4f} (per calendar day)  "
          f"rho={greeks.rho / 100:.4f} (per 1% rate move)")

    mc = monte_carlo_option_price(
        s0, strike, r, sigma, T, n_sims=args.n_sims, option_type=args.option_type,
        antithetic=not args.no_antithetic, control_variate=not args.no_control_variate, seed=args.seed,
    )
    diff = mc.price - bs_price
    ok = abs(diff) < 1.96 * mc.std_error
    print(f"\nMonte Carlo ({args.n_sims:,} paths): price = ${mc.price:.4f} "
          f"(95% CI: [{mc.ci_95[0]:.4f}, {mc.ci_95[1]:.4f}])")
    print(f"  diff from Black-Scholes: {diff:+.4f} -- "
          f"{'within 95% CI, sanity check passed' if ok else 'OUTSIDE 95% CI (expected ~5% of runs by chance; rerun with a different --seed if this persists)'}")

    plain = monte_carlo_option_price(
        s0, strike, r, sigma, T, n_sims=args.n_sims, option_type=args.option_type,
        antithetic=False, control_variate=False, seed=args.seed,
    )
    print(f"\nVariance reduction, same {args.n_sims:,} paths and seed:")
    print(f"  plain MC:               std error = {plain.std_error:.5f}")
    print(f"  antithetic + control variate:  std error = {mc.std_error:.5f}  "
          f"({plain.std_error / mc.std_error:.1f}x tighter, "
          f"~{(plain.std_error / mc.std_error) ** 2:.0f}x fewer paths needed for the same precision)")


if __name__ == "__main__":
    main()
