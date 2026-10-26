"""European option pricing, closed form + risk-neutral Monte Carlo.

underlying="futures" is Black (1976), for options on a futures price like
SI=F; underlying="spot" is Black-Scholes, for options on a non-dividend spot
asset. Both are the generalized Black-Scholes-Merton formula, differing only
in the cost of carry b (see _carry).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]
Underlying = Literal["futures", "spot"]


def _carry(r: float, underlying: Underlying) -> float:
    """Cost of carry b: the underlying's drift under the risk-neutral measure.

    A spot asset with no income must grow at r under Q. A futures position
    costs nothing to enter, so the futures price has zero drift under Q.
    Pricing a futures price with Black-Scholes' drift r overstates its
    forward value by e^{rT} -- about 7% on a 3-month at-the-money silver call
    at a 4% rate.
    """
    if underlying == "futures":
        return 0.0
    if underlying == "spot":
        return r
    raise ValueError(f"unknown underlying: {underlying!r}")


def _validate(s: float, k: float, sigma: float, T: float) -> None:
    if min(s, k, sigma, T) <= 0:
        raise ValueError("underlying price, strike, sigma and T must all be positive")


def _d1_d2(s: float, k: float, b: float, sigma: float, T: float) -> tuple[float, float]:
    vol = sigma * np.sqrt(T)
    d1 = (np.log(s / k) + (b + 0.5 * sigma**2) * T) / vol
    return d1, d1 - vol


def option_price(s: float, k: float, r: float, sigma: float, T: float,
                 option_type: OptionType = "call", underlying: Underlying = "futures") -> float:
    _validate(s, k, sigma, T)
    b = _carry(r, underlying)
    d1, d2 = _d1_d2(s, k, b, sigma, T)
    underlying_pv = s * np.exp((b - r) * T)  # discounted forward
    strike_pv = k * np.exp(-r * T)
    if option_type == "call":
        return float(underlying_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2))
    return float(strike_pv * norm.cdf(-d2) - underlying_pv * norm.cdf(-d1))


@dataclass
class Greeks:
    delta: float  # d(price)/d(underlying)
    gamma: float  # d(delta)/d(underlying)
    vega: float  # d(price)/d(sigma), sigma in absolute units (0.40, not 40)
    theta: float  # d(price)/dt per year of time passing (= -d(price)/dT); usually negative for a long option
    rho: float  # d(price)/d(r), r in absolute units (0.05, not 5)


def option_greeks(s: float, k: float, r: float, sigma: float, T: float,
                  option_type: OptionType = "call", underlying: Underlying = "futures") -> Greeks:
    _validate(s, k, sigma, T)
    b = _carry(r, underlying)
    d1, d2 = _d1_d2(s, k, b, sigma, T)
    carry_df = np.exp((b - r) * T)
    strike_pv = k * np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = carry_df * pdf_d1 / (s * sigma * np.sqrt(T))
    vega = s * carry_df * pdf_d1 * np.sqrt(T)
    time_decay = -s * carry_df * pdf_d1 * sigma / (2 * np.sqrt(T))

    if option_type == "call":
        delta = carry_df * norm.cdf(d1)
        theta = time_decay - (b - r) * s * carry_df * norm.cdf(d1) - r * strike_pv * norm.cdf(d2)
        carry_rho = T * s * carry_df * norm.cdf(d1)
    else:
        delta = carry_df * (norm.cdf(d1) - 1)
        theta = time_decay + (b - r) * s * carry_df * norm.cdf(-d1) + r * strike_pv * norm.cdf(-d2)
        carry_rho = -T * s * carry_df * norm.cdf(-d1)

    # Holding the futures price fixed, r only enters through discounting, so
    # rho = -T * price. For spot, b = r moves with r as well, adding carry_rho.
    price = option_price(s, k, r, sigma, T, option_type, underlying)
    rho = -T * price + (carry_rho if underlying == "spot" else 0.0)

    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega),
                  theta=float(theta), rho=float(rho))


@dataclass
class MonteCarloPriceResult:
    price: float
    std_error: float
    ci_95: tuple[float, float]
    n_paths: int


def monte_carlo_option_price(
    s: float, k: float, r: float, sigma: float, T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    underlying: Underlying = "futures",
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MonteCarloPriceResult:
    """Risk-neutral Monte Carlo price of a European option.

    Samples the terminal price directly from GBM's exact lognormal
    distribution in one shot -- unlike monte_carlo.py, which walks
    day-by-day because risk reporting needs the whole path. A European
    payoff only depends on the endpoint, so stepping through it a day at a
    time would just add simulation noise for nothing.

    Uses the risk-neutral drift (carry b: 0 for futures, r for spot), not
    the real-world historical mu used for risk elsewhere in this project.
    Option value is a discounted expectation under Q, not a forecast of
    where the price will actually end up.

    antithetic pairs each z with -z. The pair is then the unit of
    independent sampling: a draw and its mirror are negatively correlated,
    so they're averaged into one observation before the standard error is
    computed. Treating all draws as independent would overstate the error
    and understate what the pairing buys. control_variate uses the terminal
    price, whose risk-neutral mean s*e^{bT} is known exactly, with its
    coefficient estimated on those same units.
    """
    _validate(s, k, sigma, T)
    b = _carry(r, underlying)
    rng = np.random.default_rng(seed)
    n_units = (n_sims + 1) // 2 if antithetic else n_sims
    z = rng.standard_normal(n_units)

    def terminal(shocks: np.ndarray) -> np.ndarray:
        return s * np.exp((b - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * shocks)

    def discounted_payoff(st: np.ndarray) -> np.ndarray:
        intrinsic = st - k if option_type == "call" else k - st
        return np.exp(-r * T) * np.maximum(intrinsic, 0.0)

    st = terminal(z)
    y = discounted_payoff(st)
    x = st
    if antithetic:
        st_mirror = terminal(-z)
        y = 0.5 * (y + discounted_payoff(st_mirror))
        x = 0.5 * (st + st_mirror)

    if control_variate:
        cov = np.cov(y, x)
        coef = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        y = y - coef * (x - s * np.exp(b * T))

    price = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n_units))
    return MonteCarloPriceResult(price=price, std_error=se, ci_95=(price - 1.96 * se, price + 1.96 * se),
                                 n_paths=2 * n_units if antithetic else n_units)
