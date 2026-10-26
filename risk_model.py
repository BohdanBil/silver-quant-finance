"""Risk model = calibration method x price process x shock distribution.

The one place that turns a price history into simulated forward prices,
shared by main.py (one forecast from today) and the rolling VaR backtest
(one forecast per historical date, each from only the data available then).
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from data import (
    TRADING_DAYS_PER_YEAR,
    Garch11Params,
    GBMParams,
    JumpParams,
    bootstrap_estimates,
    estimate_jumps,
    ewma_volatility,
    fit_garch11,
    garch11_term_volatility,
    log_drift_to_mu,
    log_returns_of,
)
from monte_carlo import Dist, simulate_gbm_jump_paths, simulate_gbm_paths

Calibration = Literal["point", "ewma", "garch", "bootstrap"]
ProcessModel = Literal["gbm", "gbm-jump"]
CALIBRATIONS: tuple[Calibration, ...] = ("point", "ewma", "garch", "bootstrap")
PROCESS_MODELS: tuple[ProcessModel, ...] = ("gbm", "gbm-jump")
DISTS: tuple[Dist, ...] = ("normal", "t")


@dataclass(frozen=True)
class RiskModelSpec:
    calibration: Calibration = "point"
    model: ProcessModel = "gbm"
    dist: Dist = "normal"
    t_df: float = 5.0
    ewma_lambda: float = 0.94
    block_size: int = 21
    jump_threshold: float = 4.0

    @property
    def label(self) -> str:
        dist = "normal" if self.dist == "normal" else f"t({self.t_df:g})"
        return f"{self.calibration} / {self.model} / {dist}"


@dataclass
class SlowFit:
    """The expensive estimates: GARCH MLE and bootstrap draws.

    Both move slowly as a calibration window rolls forward, so a rolling
    backtest refits them every few weeks rather than daily. The cheap parts
    (spot, drift, EWMA/GARCH conditional variance, jumps) are always
    recomputed from the latest window.
    """
    garch: Garch11Params | None = None
    boot: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CalibratedModel:
    spec: RiskModelSpec
    params: GBMParams
    jumps: JumpParams | None
    point_sigma: float  # flat whole-window sigma, for comparison when printing
    notes: list[str]


def _split_jumps(log_returns: np.ndarray, spec: RiskModelSpec) -> tuple[JumpParams | None, np.ndarray]:
    if spec.model == "gbm-jump":
        return estimate_jumps(log_returns, spec.jump_threshold)
    return None, log_returns


def fit_slow(prices: pd.Series, spec: RiskModelSpec, n_sims: int, seed: int | None = None) -> SlowFit:
    r = log_returns_of(prices).to_numpy()
    _, diffusion = _split_jumps(r, spec)
    slow = SlowFit()
    if spec.calibration == "garch":
        try:
            slow.garch = fit_garch11(diffusion)
        except RuntimeError as e:
            slow.notes.append(f"GARCH(1,1) fit failed ({e}); fell back to EWMA sigma")
    elif spec.calibration == "bootstrap":
        slow.boot = bootstrap_estimates(r, n_boot=n_sims, block_size=spec.block_size, seed=seed,
                                        sigma_returns=diffusion)
    return slow


def calibrate(
    prices: pd.Series,
    spec: RiskModelSpec,
    horizon_days: int,
    n_sims: int,
    seed: int | None = None,
    slow: SlowFit | None = None,
) -> CalibratedModel:
    """Fit `spec` to `prices` (every price in it is treated as history).

    For a jump-diffusion every sigma below is the diffusion vol, estimated
    with the detected jump days removed. mu is always the arithmetic drift
    implied by the window's mean log return (data.log_drift_to_mu).
    """
    r = log_returns_of(prices).to_numpy()
    jumps, diffusion = _split_jumps(r, spec)
    if slow is None:
        slow = fit_slow(prices, spec, n_sims, seed)

    point_sigma = float(np.std(diffusion, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    mean_log = float(r.mean() * TRADING_DAYS_PER_YEAR)
    sigma: float | np.ndarray = point_sigma
    mu = log_drift_to_mu(mean_log, point_sigma, jumps)

    if spec.calibration == "ewma" or (spec.calibration == "garch" and slow.garch is None):
        sigma = ewma_volatility(diffusion, spec.ewma_lambda)
    elif spec.calibration == "garch":
        sigma = garch11_term_volatility(diffusion, slow.garch, horizon_days)
    elif spec.calibration == "bootstrap":
        # one (mu, sigma) draw per simulated path, so parameter uncertainty
        # (mu especially) is baked into the terminal distribution instead of
        # collapsed to a single number.
        sigma = slow.boot["sigma"].to_numpy()
        mu = log_drift_to_mu(slow.boot["mean_log"].to_numpy(), sigma, jumps)

    return CalibratedModel(
        spec=spec,
        params=GBMParams(s0=float(prices.iloc[-1]), mu=mu, sigma=sigma),
        jumps=jumps,
        point_sigma=point_sigma,
        notes=list(slow.notes),
    )


def simulate_paths(model: CalibratedModel, horizon_days: int, n_sims: int,
                   antithetic: bool = False, seed: int | None = None) -> np.ndarray:
    spec = model.spec
    if model.jumps is None:
        return simulate_gbm_paths(model.params, horizon_days, n_sims, dist=spec.dist, t_df=spec.t_df,
                                  antithetic=antithetic, seed=seed)
    return simulate_gbm_jump_paths(model.params, model.jumps, horizon_days, n_sims, dist=spec.dist,
                                   t_df=spec.t_df, antithetic=antithetic, seed=seed)


class RollingForecaster:
    """history -> simulated horizon_days simple returns, for risk.rolling_var_backtest.

    Each call calibrates on the last `window` returns of the history it is
    handed and nothing later. The slow estimates are refit once every
    `refit_every` trading days.
    """

    def __init__(self, spec: RiskModelSpec, horizon_days: int, n_sims: int, window: int,
                 refit_every: int = 21, seed: int = 0) -> None:
        self.spec = spec
        self.horizon_days = horizon_days
        self.n_sims = n_sims
        self.window = window
        self.refit_every = refit_every
        self.seed = seed
        self._slow: SlowFit | None = None
        self._fitted_at = -refit_every
        self._calls = 0

    def __call__(self, history: pd.Series) -> np.ndarray:
        prices = history.iloc[-(self.window + 1):]
        if len(history) - self._fitted_at >= self.refit_every:
            self._slow = fit_slow(prices, self.spec, self.n_sims, seed=self.seed + self._calls)
            self._fitted_at = len(history)
        model = calibrate(prices, self.spec, self.horizon_days, self.n_sims, slow=self._slow)
        paths = simulate_paths(model, self.horizon_days, self.n_sims, seed=self.seed + self._calls)
        self._calls += 1
        return paths[:, -1] / paths[:, 0] - 1
