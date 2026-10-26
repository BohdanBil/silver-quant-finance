"""CLI: price a European option on silver futures -- Black-76 closed form vs Monte Carlo.

Inputs come from the same data plumbing as the rest of the project: the
front-month COMEX futures price (SI=F), a GARCH(1,1) vol averaged over the
option's life, and the 3-month T-bill rate from FRED. Every input can be
overridden.
"""

import argparse
from dataclasses import dataclass

import pandas as pd

from data import TRADING_DAYS_PER_YEAR, fetch_silver_prices, fit_garch11, garch11_term_volatility, log_returns_of
from forecast_data import fetch_fred_series
from option_pricing import Greeks, MonteCarloPriceResult, monte_carlo_option_price, option_greeks, option_price

RISK_FREE_SERIES = "DGS3MO"  # 3-month T-bill: a reasonable proxy for a near-term risk-free rate
VARIANCE_REDUCTION_SETTINGS = [("plain", False, False), ("antithetic", True, False),
                               ("control variate", False, True), ("both", True, True)]


@dataclass
class OptionReport:
    option_type: str
    futures_price: float
    strike: float
    rate: float
    sigma: float
    maturity_days: int
    T: float
    price: float  # Black-76
    greeks: Greeks
    bs_as_spot: float  # Black-Scholes treating the futures price as a spot price, for comparison
    mc: MonteCarloPriceResult
    variance_reduction: dict[str, MonteCarloPriceResult]  # same paths and seed, each setting


def default_inputs(prices: pd.Series, maturity_days: int, as_of: str | None = None) -> tuple[float, float]:
    """(GARCH(1,1) vol averaged to expiry, latest 3-month T-bill rate)."""
    r_hist = log_returns_of(prices).to_numpy()
    sigma = garch11_term_volatility(r_hist, fit_garch11(r_hist), horizon_days=maturity_days)
    rate = float(fetch_fred_series(RISK_FREE_SERIES, as_of=as_of).iloc[-1]) / 100
    return sigma, rate


def price_silver_option(futures_price: float, strike: float, rate: float, sigma: float, maturity_days: int,
                        option_type: str = "call", n_sims: int = 200_000, seed: int = 42,
                        antithetic: bool = True, control_variate: bool = True) -> OptionReport:
    T = maturity_days / TRADING_DAYS_PER_YEAR
    common = dict(s=futures_price, k=strike, r=rate, sigma=sigma, T=T, option_type=option_type)
    mc_common = dict(common, n_sims=n_sims, underlying="futures", seed=seed)
    return OptionReport(
        option_type=option_type, futures_price=futures_price, strike=strike, rate=rate, sigma=sigma,
        maturity_days=maturity_days, T=T,
        price=option_price(**common, underlying="futures"),
        greeks=option_greeks(**common, underlying="futures"),
        bs_as_spot=option_price(**common, underlying="spot"),
        mc=monte_carlo_option_price(**mc_common, antithetic=antithetic, control_variate=control_variate),
        variance_reduction={
            label: monte_carlo_option_price(**mc_common, antithetic=anti, control_variate=cv)
            for label, anti, cv in VARIANCE_REDUCTION_SETTINGS
        },
    )


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
    default_sigma, default_rate = default_inputs(prices, args.maturity_days, args.as_of)
    report = price_silver_option(
        f0,
        strike=args.strike if args.strike is not None else f0,
        rate=args.rate if args.rate is not None else default_rate,
        sigma=args.sigma if args.sigma is not None else default_sigma,
        maturity_days=args.maturity_days, option_type=args.option_type, n_sims=args.n_sims, seed=args.seed,
        antithetic=not args.no_antithetic, control_variate=not args.no_control_variate,
    )

    print(f"Silver futures {report.option_type} ({prices.index[-1].date()}): F=${f0:.4f}  K=${report.strike:.4f}  "
          f"r={report.rate:.3%}  sigma={report.sigma:.2%}  T={report.T:.3f}y ({args.maturity_days} trading days)")
    print("  F is the front-month COMEX price; listed options are on a specific contract month, so this "
          "treats the front month as that contract.")
    print("  sigma is a GARCH(1,1) forecast averaged over the option's life, a stand-in for implied vol -- "
          "no options chain is wired in to pull real implied vol from.")

    g = report.greeks
    print(f"\nBlack-76 price = ${report.price:.4f}")
    print(f"  delta={g.delta:+.4f}  gamma={g.gamma:.4f}  "
          f"vega={g.vega / 100:.4f} (per 1 vol pt)  "
          f"theta={g.theta / TRADING_DAYS_PER_YEAR:.4f} (per trading day)  "
          f"rho={g.rho / 100:.4f} (per 1% rate move)")
    print(f"  (Black-Scholes, wrongly treating F as a spot price, would say ${report.bs_as_spot:.4f}: "
          f"{report.bs_as_spot / report.price - 1:+.1%})")

    mc = report.mc
    diff = mc.price - report.price
    ok = abs(diff) < 1.96 * mc.std_error
    print(f"\nMonte Carlo ({mc.n_paths:,} paths): price = ${mc.price:.4f} "
          f"(95% CI: [{mc.ci_95[0]:.4f}, {mc.ci_95[1]:.4f}])")
    print(f"  diff from Black-76: {diff:+.4f} -- "
          f"{'within 95% CI, sanity check passed' if ok else 'OUTSIDE 95% CI (expected ~5% of runs by chance; rerun with a different --seed if this persists)'}")

    print(f"\nVariance reduction, same {args.n_sims:,} paths and seed:")
    plain_se = report.variance_reduction["plain"].std_error
    for label, res in report.variance_reduction.items():
        print(f"  {label:<16} std error = {res.std_error:.5f}  "
              f"(~{(plain_se / res.std_error) ** 2:.1f}x fewer paths than plain for the same precision)")


if __name__ == "__main__":
    main()
