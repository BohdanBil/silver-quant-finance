"""European option pricing: Black-Scholes closed form + risk-neutral Monte Carlo."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(s0: float, k: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(s0: float, k: float, r: float, sigma: float, T: float,
                         option_type: OptionType = "call") -> float:
    d1, d2 = _d1_d2(s0, k, r, sigma, T)
    if option_type == "call":
        return s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


@dataclass
class Greeks:
    delta: float  # d(price)/d(spot)
    gamma: float  # d(delta)/d(spot)
    vega: float  # d(price)/d(sigma), sigma in absolute units (0.40, not 40)
    theta: float  # d(price)/d(T), T in years -- note this is *positive* T decaying, see option_main.py for sign
    rho: float  # d(price)/d(r), r in absolute units (0.05, not 5)


def black_scholes_greeks(s0: float, k: float, r: float, sigma: float, T: float,
                          option_type: OptionType = "call") -> Greeks:
    d1, d2 = _d1_d2(s0, k, r, sigma, T)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (s0 * sigma * np.sqrt(T))
    vega = s0 * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -s0 * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * k * np.exp(-r * T) * norm.cdf(d2)
        rho = k * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -s0 * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * k * np.exp(-r * T) * norm.cdf(-d2)
        rho = -k * T * np.exp(-r * T) * norm.cdf(-d2)

    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega),
                   theta=float(theta), rho=float(rho))


@dataclass
class MonteCarloPriceResult:
    price: float
    std_error: float
    ci_95: tuple[float, float]
    n_sims: int


def monte_carlo_option_price(
    s0: float, k: float, r: float, sigma: float, T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MonteCarloPriceResult:
    """Risk-neutral Monte Carlo price of a European option.

    Samples S_T directly from GBM's exact terminal lognormal distribution
    in one shot -- unlike monte_carlo.py's simulate_gbm_paths, which walks
    day-by-day because it needs the whole *path* for risk reporting. A
    European payoff only depends on the endpoint, and that endpoint has a
    closed-form distribution under GBM, so stepping through it a day at a
    time would just add simulation noise for nothing.

    Uses risk-neutral drift r, not the real-world/historical mu used
    elsewhere in this project (data.py's calibrate_gbm, monte_carlo.py's
    risk paths). Option value is a discounted expectation under the
    risk-neutral measure Q, not a forecast of where the price will
    actually end up -- those are different drifts answering different
    questions, and using historical mu here would misprice the option.

    antithetic: pairs each z with -z (halves simulation noise from skew in
    the driving normal draws). control_variate: uses S_T itself as a
    control -- its risk-neutral mean E[S_T] = s0*exp(rT) is known exactly,
    so the gap between the simulated mean and that closed form cancels
    part of the payoff estimator's noise too. Both are optional and can be
    disabled to see the variance-reduction effect (see option_main.py).
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = (n_sims + 1) // 2
        z = rng.standard_normal(n_half)
        z = np.concatenate([z, -z])[:n_sims]
    else:
        z = rng.standard_normal(n_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    st = s0 * np.exp(drift + diffusion * z)

    if option_type == "call":
        payoff = np.maximum(st - k, 0.0)
    else:
        payoff = np.maximum(k - st, 0.0)
    discounted = np.exp(-r * T) * payoff

    if control_variate:
        st_mean_theoretical = s0 * np.exp(r * T)
        var_st = np.var(st)
        b = np.cov(discounted, st)[0, 1] / var_st if var_st > 0 else 0.0
        samples = discounted - b * (st - st_mean_theoretical)
    else:
        samples = discounted

    price = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n_sims))
    return MonteCarloPriceResult(price=price, std_error=se, ci_95=(price - 1.96 * se, price + 1.96 * se),
                                  n_sims=n_sims)
