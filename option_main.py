"""CLI: price a European option on silver futures -- Black-76 closed form vs Monte Carlo.

Inputs come from the same data plumbing as the rest of the project: the
front-month COMEX futures price (SI=F), a GARCH(1,1) vol averaged over the
option's life, and the 3-month T-bill rate from FRED. Every input can be
overridden.
"""

import argparse

from data import TRADING_DAYS_PER_YEAR, fetch_silver_prices, fit_garch11, garch11_term_volatility, log_returns_of
from forecast_data import fetch_fred_series
from option_pricing import monte_carlo_option_price, option_greeks, option_price

RISK_FREE_SERIES = "DGS3MO"  # 3-month T-bill: a reasonable proxy for a near-term risk-free rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strike", type=float, default=None, help="defaults to at-the-money (= futures price)")
    p.add_argument("--maturity-days", type=int, default=63, help="trading days to expiry (default ~3 months)")
    p.add_argument("--option-type", choices=["call", "put"], default="call")
    p.add_argument("--rate", type=float, default=None,
                   help="annualized risk-free rate (e.g. 0.04); default: 3M T-bill from FRED")
    p.add_argument("--sigma", type=float, default=None,
                   help="annualized vol (e.g. 0.40); default: GARCH(1,1) forecast averaged to expiry")
    p.add_argument("--period", default="5y", help="price history used for the GARCH fit")
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--n-sims", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--no-control-variate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    f0 = float(prices.iloc[-1])
    if args.sigma is not None:
        sigma = args.sigma
    else:
        r_hist = log_returns_of(prices).to_numpy()
        sigma = garch11_term_volatility(r_hist, fit_garch11(r_hist), horizon_days=args.maturity_days)
    strike = args.strike if args.strike is not None else f0

    if args.rate is not None:
        r = args.rate
    else:
        r = float(fetch_fred_series(RISK_FREE_SERIES, as_of=args.as_of).iloc[-1]) / 100

    T = args.maturity_days / TRADING_DAYS_PER_YEAR
    kind = args.option_type

    print(f"Silver futures {kind} ({prices.index[-1].date()}): F=${f0:.4f}  K=${strike:.4f}  "
          f"r={r:.3%}  sigma={sigma:.2%}  T={T:.3f}y ({args.maturity_days} trading days)")
    print("  F is the front-month COMEX price; listed options are on a specific contract month, so this "
          "treats the front month as that contract.")
    print("  sigma is a GARCH(1,1) forecast averaged over the option's life, a stand-in for implied vol -- "
          "no options chain is wired in to pull real implied vol from.")

    price = option_price(f0, strike, r, sigma, T, option_type=kind, underlying="futures")
    greeks = option_greeks(f0, strike, r, sigma, T, option_type=kind, underlying="futures")
    bs_as_spot = option_price(f0, strike, r, sigma, T, option_type=kind, underlying="spot")

    print(f"\nBlack-76 price = ${price:.4f}")
    print(f"  delta={greeks.delta:+.4f}  gamma={greeks.gamma:.4f}  "
          f"vega={greeks.vega / 100:.4f} (per 1 vol pt)  "
          f"theta={greeks.theta / TRADING_DAYS_PER_YEAR:.4f} (per trading day)  "
          f"rho={greeks.rho / 100:.4f} (per 1% rate move)")
    print(f"  (Black-Scholes, wrongly treating F as a spot price, would say ${bs_as_spot:.4f}: "
          f"{bs_as_spot / price - 1:+.1%})")

    mc = monte_carlo_option_price(
        f0, strike, r, sigma, T, n_sims=args.n_sims, option_type=kind, underlying="futures",
        antithetic=not args.no_antithetic, control_variate=not args.no_control_variate, seed=args.seed,
    )
    diff = mc.price - price
    ok = abs(diff) < 1.96 * mc.std_error
    print(f"\nMonte Carlo ({mc.n_paths:,} paths): price = ${mc.price:.4f} "
          f"(95% CI: [{mc.ci_95[0]:.4f}, {mc.ci_95[1]:.4f}])")
    print(f"  diff from Black-76: {diff:+.4f} -- "
          f"{'within 95% CI, sanity check passed' if ok else 'OUTSIDE 95% CI (expected ~5% of runs by chance; rerun with a different --seed if this persists)'}")

    print(f"\nVariance reduction, same {args.n_sims:,} paths and seed:")
    plain_se = None
    for label, antithetic, control in [("plain", False, False), ("antithetic", True, False),
                                       ("control variate", False, True), ("both", True, True)]:
        res = monte_carlo_option_price(f0, strike, r, sigma, T, n_sims=args.n_sims, option_type=kind,
                                       underlying="futures", antithetic=antithetic, control_variate=control,
                                       seed=args.seed)
        plain_se = plain_se or res.std_error
        print(f"  {label:<16} std error = {res.std_error:.5f}  "
              f"(~{(plain_se / res.std_error) ** 2:.1f}x fewer paths than plain for the same precision)")


if __name__ == "__main__":
    main()
