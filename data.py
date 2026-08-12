"""Fetch historical silver prices and calibrate a GBM model from them."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

SILVER_TICKER = "SI=F"  # COMEX silver futures. Use "SLV" for the ETF instead.
TRADING_DAYS_PER_YEAR = 252


@dataclass
class GBMParams:
    s0: float
    mu: float | np.ndarray  # annualized drift (scalar, or one value per sim path)
    sigma: float | np.ndarray  # annualized volatility (scalar, or one value per sim path)


@dataclass
class Garch11Params:
    omega: float
    alpha: float
    beta: float
    converged: bool


def fetch_silver_prices(period: str = "5y", ticker: str = SILVER_TICKER) -> pd.Series:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    close = df["Close"].dropna()
    return close.squeeze()


def log_returns_of(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def calibrate_gbm(prices: pd.Series) -> GBMParams:
    """Point estimate: single mu/sigma from the full-period mean/std.

    mu is estimated from a 5y sample mean of daily returns, which is
    notoriously noisy (drift estimation error scales with sample length,
    not sample size in any way that helps over a few years). Treat mu as
    directional at best; see calibrate_gbm_bootstrap() for an estimate of
    how uncertain it actually is.
    """
    log_returns = log_returns_of(prices)
    mu_annual = log_returns.mean() * TRADING_DAYS_PER_YEAR
    sigma_annual = log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    s0 = float(prices.iloc[-1])
    return GBMParams(s0=s0, mu=float(mu_annual), sigma=float(sigma_annual))


def ewma_volatility(prices: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics-style EWMA annualized volatility.

    Weights recent squared returns most heavily (halflife ~11 trading days
    at lam=0.94), so it tracks the current vol regime instead of averaging
    over 5 years of mostly irrelevant history.
    """
    r = log_returns_of(prices).values
    var = r[0] ** 2
    for ret in r[1:]:
        var = lam * var + (1 - lam) * ret**2
    return float(np.sqrt(var) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _garch11_neg_loglik(x: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = x
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-14)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2)
    return -ll


def fit_garch11(prices: pd.Series, n_starts: int = 8, seed: int = 0) -> Garch11Params:
    """MLE fit of a GARCH(1,1) model to daily log returns (normal innovations).

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Fits on returns scaled to "percent" units (x100). Raw daily log returns
    are ~1e-2 while alpha/beta are O(0.1-1), and omega ends up ~1e-5 in raw
    units — that four-order-of-magnitude spread badly conditions SLSQP's
    finite-difference gradient and it silently returns the initial guess.
    Percent-scaling brings omega to a comparable O(0.1-1) scale; alpha/beta
    are scale-invariant so only omega needs rescaling back afterwards.
    Also multi-starts, since the likelihood surface is flat enough that a
    single start can land on a poor local optimum.
    """
    r_raw = log_returns_of(prices).values
    r = (r_raw - r_raw.mean()) * 100
    var_target = np.var(r)
    bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
    constraints = ({"type": "ineq", "fun": lambda x: 0.999 - (x[1] + x[2])},)

    rng = np.random.default_rng(seed)
    candidates = [np.array([var_target * 0.05, 0.05, 0.90])]
    for _ in range(n_starts - 1):
        a = rng.uniform(0.01, 0.3)
        b = rng.uniform(0.3, 0.95 - a)
        candidates.append(np.array([var_target * max(1 - a - b, 0.01), a, b]))

    best = None
    for x0 in candidates:
        result = minimize(
            _garch11_neg_loglik, x0, args=(r,), method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise RuntimeError("GARCH(1,1) MLE failed to converge from any starting point")

    omega_pct, alpha, beta = best.x
    omega = omega_pct / 100**2  # undo the percent-return scaling (alpha, beta are scale-invariant)
    return Garch11Params(omega=float(omega), alpha=float(alpha), beta=float(beta), converged=True)


def garch11_volatility(prices: pd.Series, fit: Garch11Params | None = None) -> float:
    """One-step-ahead GARCH(1,1) volatility forecast, annualized."""
    r = log_returns_of(prices).values
    r = r - r.mean()
    if fit is None:
        fit = fit_garch11(prices)

    sigma2 = np.empty(len(r))
    sigma2[0] = np.var(r)
    for t in range(1, len(r)):
        sigma2[t] = fit.omega + fit.alpha * r[t - 1] ** 2 + fit.beta * sigma2[t - 1]
    next_var = fit.omega + fit.alpha * r[-1] ** 2 + fit.beta * sigma2[-1]
    return float(np.sqrt(next_var) * np.sqrt(TRADING_DAYS_PER_YEAR))


def calibrate_gbm_ewma(prices: pd.Series, lam: float = 0.94) -> GBMParams:
    """Point mu (still noisy, see calibrate_gbm) with EWMA-based current-regime sigma."""
    point = calibrate_gbm(prices)
    return GBMParams(s0=point.s0, mu=point.mu, sigma=ewma_volatility(prices, lam=lam))


def calibrate_gbm_garch(prices: pd.Series) -> GBMParams:
    """Point mu (still noisy, see calibrate_gbm) with GARCH(1,1)-forecast sigma."""
    point = calibrate_gbm(prices)
    return GBMParams(s0=point.s0, mu=point.mu, sigma=garch11_volatility(prices))


def calibrate_gbm_bootstrap(
    prices: pd.Series,
    n_boot: int = 10_000,
    block_size: int = 21,
    seed: int | None = None,
) -> pd.DataFrame:
    """Moving-block bootstrap distribution of annualized (mu, sigma).

    Resamples overlapping blocks of `block_size` consecutive log-returns
    (rather than single returns) so short-range autocorrelation and
    volatility clustering survive the resample. Each draw re-estimates mu
    and sigma from a reassembled synthetic history the same length as the
    original, giving a distribution of plausible parameters instead of one
    point estimate. mu's spread is typically enormous relative to its mean
    over a 5y window — that spread is the point, not noise to average away.
    """
    r = log_returns_of(prices).values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    starts_max = n - block_size
    if starts_max < 0:
        raise ValueError("block_size larger than the available return history")

    rng = np.random.default_rng(seed)
    mus = np.empty(n_boot)
    sigmas = np.empty(n_boot)
    for i in range(n_boot):
        block_starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([r[s : s + block_size] for s in block_starts])[:n]
        mus[i] = sample.mean() * TRADING_DAYS_PER_YEAR
        sigmas[i] = sample.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    return pd.DataFrame({"mu": mus, "sigma": sigmas})
