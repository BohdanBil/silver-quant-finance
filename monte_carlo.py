"""Monte Carlo simulation of future silver prices under GBM (+ fat tails / jumps)."""

from typing import Literal

import numpy as np

from data import GBMParams, TRADING_DAYS_PER_YEAR

Dist = Literal["normal", "t"]


def _draw_base(n_sims: int, horizon_days: int, dist: Dist, t_df: float,
                rng: np.random.Generator) -> np.ndarray:
    if dist == "normal":
        return rng.standard_normal((n_sims, horizon_days))
    if dist == "t":
        if t_df <= 2:
            raise ValueError("t_df must be > 2 (variance is undefined otherwise)")
        raw = rng.standard_t(t_df, size=(n_sims, horizon_days))
        return raw / np.sqrt(t_df / (t_df - 2))  # rescale to unit variance
    raise ValueError(f"unknown dist: {dist!r}")


def _draw_innovations(n_sims: int, horizon_days: int, dist: Dist, t_df: float,
                       antithetic: bool, rng: np.random.Generator) -> np.ndarray:
    """Standardized (mean 0, var 1) innovations, shape (n_sims, horizon_days)."""
    if not antithetic:
        return _draw_base(n_sims, horizon_days, dist, t_df, rng)

    half = (n_sims + 1) // 2
    z_half = _draw_base(half, horizon_days, dist, t_df, rng)
    return np.concatenate([z_half, -z_half], axis=0)[:n_sims]


def _as_column(x: float | np.ndarray) -> np.ndarray | float:
    """Reshape a per-path parameter array to (n_sims, 1) so it broadcasts against
    (n_sims, horizon_days). Scalars pass through untouched."""
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def simulate_gbm_paths(
    params: GBMParams,
    horizon_days: int,
    n_sims: int,
    dist: Dist = "normal",
    t_df: float = 5.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths.

    params.mu / params.sigma may be scalars (all paths share one parameter
    set) or arrays of length n_sims (e.g. one draw per path from
    calibrate_gbm_bootstrap), in which case parameter uncertainty is baked
    into the simulation alongside the usual random-shock uncertainty.

    dist="t" swaps the standard normal shocks for standardized Student-t
    ones (fatter tails than GBM's normal increments; silver realistically
    jumps more than a lognormal process admits). antithetic=True pairs each
    draw with its mirror image to cut Monte Carlo variance on VaR/CVaR
    estimates for the same n_sims.

    Returns an array of shape (n_sims, horizon_days + 1), column 0 is s0.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS_PER_YEAR
    mu = _as_column(params.mu)
    sigma = _as_column(params.sigma)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    z = _draw_innovations(n_sims, horizon_days, dist, t_df, antithetic, rng)
    log_returns = drift + diffusion * z
    log_paths = np.cumsum(log_returns, axis=1)

    paths = np.empty((n_sims, horizon_days + 1))
    paths[:, 0] = params.s0
    paths[:, 1:] = params.s0 * np.exp(log_paths)
    return paths


def simulate_gbm_jump_paths(
    params: GBMParams,
    horizon_days: int,
    n_sims: int,
    jump_intensity: float = 2.0,
    jump_mean: float = -0.02,
    jump_std: float = 0.05,
    dist: Dist = "normal",
    t_df: float = 5.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Merton jump-diffusion: GBM plus a compound Poisson jump component.

    Each trading day, the number of jumps N ~ Poisson(jump_intensity * dt)
    and jump sizes are iid log-normal shocks ~ Normal(jump_mean, jump_std).
    The drift is compensated by k = E[e^Y - 1] so mu still equals the
    expected annualized return with jumps included, not just the diffusion.

    jump_intensity: expected number of jumps per year (e.g. 2 = a couple of
    sharp moves/year, roughly matching how often silver gaps on macro
    surprises). jump_mean/jump_std: log-size distribution of a single jump
    (default centered slightly negative — crash risk is usually the part a
    pure GBM misses, not the occasional spike up).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS_PER_YEAR
    mu = _as_column(params.mu)
    sigma = _as_column(params.sigma)
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
    drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt
    diffusion = sigma * np.sqrt(dt)

    z = _draw_innovations(n_sims, horizon_days, dist, t_df, antithetic, rng)

    n_jumps = rng.poisson(jump_intensity * dt, size=(n_sims, horizon_days))
    # sum of n_jumps iid Normal(jump_mean, jump_std^2) draws is
    # Normal(n_jumps*jump_mean, n_jumps*jump_std^2); handles 0, 1, or several
    # jumps landing in the same step without an inner loop.
    jump_component = (
        rng.standard_normal(n_jumps.shape) * jump_std * np.sqrt(n_jumps)
        + n_jumps * jump_mean
    )

    log_returns = drift + diffusion * z + jump_component
    log_paths = np.cumsum(log_returns, axis=1)

    paths = np.empty((n_sims, horizon_days + 1))
    paths[:, 0] = params.s0
    paths[:, 1:] = params.s0 * np.exp(log_paths)
    return paths
