"""Fetch historical prices (cached on disk) and estimate volatility, drift and jumps from them."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.signal import lfilter

SILVER_TICKER = "SI=F"  # COMEX silver front-month futures. Use "SLV" for the ETF instead.
TRADING_DAYS_PER_YEAR = 252

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

Frame = TypeVar("Frame", pd.Series, pd.DataFrame)


# --- fetching ---------------------------------------------------------------

def resolve_as_of(as_of: str | pd.Timestamp | None) -> pd.Timestamp:
    """The last date fetched data may come from (default: today)."""
    return pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()


def period_start(end: pd.Timestamp, period: str) -> pd.Timestamp:
    """Start date `period` ('5y', '18mo', '90d') before `end`."""
    match = re.fullmatch(r"(\d+)(y|mo|d)", period)
    if match is None:
        raise ValueError(f"period must look like '5y', '18mo' or '90d', got {period!r}")
    n, unit = int(match.group(1)), match.group(2)
    offset = {"y": pd.DateOffset(years=n), "mo": pd.DateOffset(months=n), "d": pd.DateOffset(days=n)}[unit]
    return end - offset


def cached(key: str, fetch: Callable[[], Frame]) -> Frame:
    """Load `key` from the on-disk cache, or fetch it and store it.

    Every key includes the as-of date, so a live run re-downloads once per
    day and a run pinned with --as-of reads exactly the same data every
    time. Delete .cache/ to force a fresh download.
    """
    path = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".pkl")
    if path.exists():
        return pd.read_pickle(path)
    value = fetch()
    CACHE_DIR.mkdir(exist_ok=True)
    value.to_pickle(path)
    return value


def fetch_yf_close(ticker: str, period: str = "5y", as_of: str | None = None) -> pd.Series:
    """Daily closes for `ticker` over `period`, ending at `as_of` (inclusive)."""
    end = resolve_as_of(as_of)
    start = period_start(end, period)

    def fetch() -> pd.Series:
        # yfinance's `end` is exclusive, hence +1 day to keep the as-of date's own bar.
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        close = df["Close"].dropna().squeeze()
        close.name = ticker
        return close

    return cached(f"yf_{ticker}_{start.date()}_{end.date()}", fetch)


def fetch_silver_prices(period: str = "5y", as_of: str | None = None, ticker: str = SILVER_TICKER) -> pd.Series:
    return fetch_yf_close(ticker, period=period, as_of=as_of)


def log_returns_of(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


# --- drift ------------------------------------------------------------------

@dataclass
class GBMParams:
    s0: float
    mu: float | np.ndarray  # annualized expected (arithmetic) return; the log-price drift is mu - sigma^2/2
    sigma: float | np.ndarray  # annualized volatility (scalar, or one value per sim path)


@dataclass
class JumpParams:
    intensity: float  # expected jumps per year
    mean: float  # mean log jump size
    std: float  # std of log jump size
    n_detected: int

    @property
    def compensator(self) -> float:
        """k = E[e^Y - 1], the expected simple return of a single jump."""
        return float(np.exp(self.mean + 0.5 * self.std**2) - 1)


def log_drift_to_mu(mean_log_return: float | np.ndarray, sigma: float | np.ndarray,
                    jumps: JumpParams | None = None) -> float | np.ndarray:
    """Annualized mean log return -> the arithmetic drift mu the simulators expect.

    A sample mean of log returns estimates E[log return], which under GBM is
    mu - sigma^2/2, not mu. The simulators subtract sigma^2/2 themselves, so
    passing the raw mean in as mu would make every simulated path drift
    sigma^2/2 per year (~7% at silver's vol) below the history it was
    calibrated on. With Merton jumps, E[log return] is
    mu - sigma^2/2 - intensity*k + intensity*jump_mean.

    mu is still estimated from a few years of data, which is notoriously
    noisy -- see bootstrap_estimates for how uncertain it really is.
    """
    mu = mean_log_return + 0.5 * np.square(sigma)
    if jumps is not None:
        mu = mu + jumps.intensity * (jumps.compensator - jumps.mean)
    return mu


# --- volatility -------------------------------------------------------------

def ewma_volatility(log_returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics-style EWMA annualized volatility.

    Weights recent squared returns most heavily (halflife ~11 trading days
    at lam=0.94), so it tracks the current vol regime instead of averaging
    over years of mostly irrelevant history.
    """
    r2 = np.asarray(log_returns, dtype=float) ** 2
    x = (1 - lam) * r2
    x[0] = r2[0]
    var = lfilter([1.0], [1.0, -lam], x)  # var[t] = lam * var[t-1] + (1 - lam) * r[t]^2
    return float(np.sqrt(var[-1] * TRADING_DAYS_PER_YEAR))


@dataclass
class Garch11Params:
    omega: float  # in raw daily-return^2 units
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1 - self.persistence)


def _garch11_variance(omega: float, alpha: float, beta: float, r: np.ndarray) -> np.ndarray:
    """sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1], with sigma2[0] = var(r).

    A first-order linear recursion, so lfilter runs it in C rather than a
    Python loop -- fast enough to refit inside a rolling backtest.
    """
    x = np.empty_like(r)
    x[0] = np.var(r)
    x[1:] = omega + alpha * r[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], x)


def _garch11_neg_loglik(x: np.ndarray, r: np.ndarray) -> float:
    sigma2 = np.maximum(_garch11_variance(x[0], x[1], x[2], r), 1e-14)
    return 0.5 * float(np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2))


def fit_garch11(log_returns: np.ndarray, n_starts: int = 8, seed: int = 0) -> Garch11Params:
    """MLE fit of a GARCH(1,1) model to daily log returns (normal innovations).

    Fits on returns scaled to "percent" units (x100). Raw daily log returns
    are ~1e-2 while alpha/beta are O(0.1-1), and omega ends up ~1e-5 in raw
    units -- that four-order-of-magnitude spread badly conditions SLSQP's
    finite-difference gradient and it silently returns the initial guess.
    Percent-scaling brings omega to a comparable O(0.1-1) scale; alpha/beta
    are scale-invariant so only omega needs rescaling back afterwards.
    Also multi-starts, since the likelihood surface is flat enough that a
    single start can land on a poor local optimum.
    """
    r_raw = np.asarray(log_returns, dtype=float)
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
    return Garch11Params(omega=float(omega_pct / 100**2), alpha=float(alpha), beta=float(beta))


def garch11_term_volatility(log_returns: np.ndarray, fit: Garch11Params, horizon_days: int = 1) -> float:
    """Annualized vol over the next `horizon_days`, from a GARCH(1,1) fit.

    The one-step-ahead variance only describes tomorrow. GARCH variance
    mean-reverts toward V_L = omega / (1 - alpha - beta) at rate
    alpha + beta per day, so the k-step forecast is
        E[sigma2_{t+k}] = V_L + (alpha + beta)^(k-1) * (sigma2_{t+1} - V_L)
    and a horizon_days return's variance is the sum of those. Using the
    one-step vol for a 63-day option would carry today's elevated (or
    depressed) vol across the whole three months.
    """
    r = np.asarray(log_returns, dtype=float)
    r = r - r.mean()
    sigma2 = _garch11_variance(fit.omega, fit.alpha, fit.beta, r)
    next_var = fit.omega + fit.alpha * r[-1] ** 2 + fit.beta * sigma2[-1]
    k = np.arange(horizon_days)
    term = fit.long_run_variance + fit.persistence**k * (next_var - fit.long_run_variance)
    return float(np.sqrt(term.mean() * TRADING_DAYS_PER_YEAR))


# --- parameter uncertainty --------------------------------------------------

def bootstrap_estimates(
    log_returns: np.ndarray,
    n_boot: int = 10_000,
    block_size: int = 21,
    seed: int | None = None,
    sigma_returns: np.ndarray | None = None,
) -> pd.DataFrame:
    """Moving-block bootstrap distribution of annualized (mean log return, sigma).

    Resamples overlapping blocks of `block_size` consecutive log-returns
    (rather than single returns) so short-range autocorrelation and
    volatility clustering survive the resample. Each draw re-estimates the
    mean and sigma from a reassembled synthetic history the same length as
    the original, giving a distribution of plausible parameters instead of
    one point estimate. The mean's spread is typically enormous relative to
    its value over a 5y window -- that spread is the point, not noise to
    average away.

    sigma_returns: estimate sigma from this series instead, using the same
    resampled blocks (the jump-cleaned returns, for a jump-diffusion model).
    """
    r = np.asarray(log_returns, dtype=float)
    s = r if sigma_returns is None else np.asarray(sigma_returns, dtype=float)
    n = len(r)
    if block_size > n:
        raise ValueError("block_size larger than the available return history")
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    sigmas = np.empty(n_boot)
    chunk = max(1, 2_000_000 // n)  # keeps each gathered index array around 16MB
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        starts = rng.integers(0, n - block_size + 1, size=(hi - lo, n_blocks))
        idx = (starts[:, :, None] + offsets).reshape(hi - lo, -1)[:, :n]
        means[lo:hi] = r[idx].mean(axis=1) * TRADING_DAYS_PER_YEAR
        sigmas[lo:hi] = s[idx].std(axis=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return pd.DataFrame({"mean_log": means, "sigma": sigmas})


# --- jumps ------------------------------------------------------------------

def detect_jumps(log_returns: np.ndarray, threshold: float = 4.0, window: int = 63) -> np.ndarray:
    """Flag returns more than `threshold` local standard deviations from zero (Lee-Mykland style).

    Local vol at day t is bipower variation over the preceding `window`
    returns, (pi/2) * mean(|r_j| * |r_j-1|). A single huge return barely
    moves it, because it only ever enters multiplied by an ordinary-sized
    neighbour -- unlike a plain std, which the jumps themselves inflate.
    Using a trailing local window rather than one full-sample vol stops a
    high-vol regime from being misread as a run of jumps.
    """
    r = np.asarray(log_returns, dtype=float)
    products = np.abs(r[1:]) * np.abs(r[:-1])  # products[j] pairs r[j+1] with r[j]
    rolling = pd.Series(products).rolling(window - 1, min_periods=min(20, len(products))).mean().to_numpy()
    local = np.full(len(r), np.nan)
    local[2:] = rolling[: len(r) - 2]  # day t only sees pairs ending at t-1 or earlier
    first_valid = np.flatnonzero(~np.isnan(local))
    fallback = np.mean(products) if len(products) else np.nan
    local[np.isnan(local)] = local[first_valid[0]] if len(first_valid) else fallback
    return np.abs(r) > threshold * np.sqrt(np.pi / 2 * local)


def estimate_jumps(log_returns: np.ndarray, threshold: float = 4.0) -> tuple[JumpParams, np.ndarray]:
    """Split daily log returns into jumps and diffusion.

    Returns (JumpParams, diffusion_returns), where diffusion_returns has each
    jump day replaced by the mean non-jump return. Diffusion vol must be
    estimated from that series: a sigma measured on all returns already
    contains the jumps' variance, and the jump component would then add it
    a second time.
    """
    r = np.asarray(log_returns, dtype=float)
    is_jump = detect_jumps(r, threshold)
    jump_sizes = r[is_jump]
    years = len(r) / TRADING_DAYS_PER_YEAR
    params = JumpParams(
        intensity=len(jump_sizes) / years,
        mean=float(jump_sizes.mean()) if len(jump_sizes) else 0.0,
        std=float(jump_sizes.std()) if len(jump_sizes) > 1 else 0.0,
        n_detected=int(len(jump_sizes)),
    )
    diffusion = r.copy()
    diffusion[is_jump] = r[~is_jump].mean()
    return params, diffusion
