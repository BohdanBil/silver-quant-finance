"""Monte Carlo simulation of future silver prices under GBM (+ fat tails / jumps)."""

from typing import Literal

import numpy as np

from data import GBMParams, JumpParams, TRADING_DAYS_PER_YEAR

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


def _paths_from_log_returns(s0: float, log_returns: np.ndarray) -> np.ndarray:
    paths = np.empty((log_returns.shape[0], log_returns.shape[1] + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


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
    data.bootstrap_estimates), in which case parameter uncertainty is baked
    into the simulation alongside the usual random-shock uncertainty.

    dist="t" swaps the standard normal shocks for standardized Student-t
    ones (fatter tails than GBM's normal increments). antithetic=True pairs
    each draw with its mirror image to cut Monte Carlo variance on VaR/CVaR
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
    return _paths_from_log_returns(params.s0, drift + diffusion * z)


def simulate_gbm_jump_paths(
    params: GBMParams,
    jumps: JumpParams,
    horizon_days: int,
    n_sims: int,
    dist: Dist = "normal",
    t_df: float = 5.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Merton jump-diffusion: GBM plus a compound Poisson jump component.

    Each trading day, the number of jumps N ~ Poisson(intensity * dt) and
    jump sizes are iid log-normal shocks ~ Normal(jumps.mean, jumps.std).
    The drift is compensated by k = E[e^Y - 1] so mu still equals the
    expected annualized return with jumps included, not just the diffusion.

    params.sigma should be the *diffusion* vol, estimated with the jump days
    removed (data.estimate_jumps); otherwise the jumps' variance is counted
    twice.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS_PER_YEAR
    mu = _as_column(params.mu)
    sigma = _as_column(params.sigma)
    drift = (mu - 0.5 * sigma**2 - jumps.intensity * jumps.compensator) * dt
    diffusion = sigma * np.sqrt(dt)

    z = _draw_innovations(n_sims, horizon_days, dist, t_df, antithetic, rng)

    n_jumps = rng.poisson(jumps.intensity * dt, size=(n_sims, horizon_days))
    # sum of n_jumps iid Normal(mean, std^2) draws is Normal(n*mean, n*std^2);
    # handles 0, 1, or several jumps landing in the same step without an inner loop.
    jump_component = rng.standard_normal(n_jumps.shape) * jumps.std * np.sqrt(n_jumps) + n_jumps * jumps.mean

    return _paths_from_log_returns(params.s0, drift + diffusion * z + jump_component)
