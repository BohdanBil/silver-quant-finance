"""Value-at-Risk and Conditional VaR from a distribution of simulated P&L."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass
class RiskMetrics:
    confidence: float
    var: float  # loss threshold, positive number = loss
    cvar: float  # expected loss beyond VaR, positive number = loss


def var_cvar(pnl: np.ndarray, confidence: float = 0.95) -> RiskMetrics:
    """Historical VaR/CVaR from simulated P&L (or return) samples.

    pnl: array of simulated profit/loss outcomes (negative = loss).
    confidence: e.g. 0.95 for 95% VaR.
    """
    losses = -pnl
    alpha = 1 - confidence
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return RiskMetrics(confidence=confidence, var=var, cvar=cvar)


@dataclass
class BacktestResult:
    confidence: float
    n_obs: int
    n_exceptions: int
    exception_rate: float
    expected_rate: float
    lr_stat: float
    p_value: float
    reject_at_5pct: bool  # True => model's exception rate is statistically off


def backtest_var(
    prices: pd.Series,
    var_threshold: float,
    position_oz: float,
    horizon_days: int,
    confidence: float = 0.95,
) -> BacktestResult:
    """Kupiec (1995) proportion-of-failures test for a VaR threshold.

    Walks the actual price history in overlapping horizon_days windows,
    turns each into a realized P&L for the given position, and counts how
    often the real loss would have exceeded var_threshold. If the model is
    calibrated correctly that should happen at rate (1 - confidence); the
    Kupiec likelihood-ratio test checks whether the observed exception rate
    is statistically consistent with that, rather than just eyeballing it.

    Caveat: horizon_days-ahead windows overlap (day t+1's window shares all
    but one return with day t's), so exceptions are autocorrelated and the
    effective sample size is smaller than n_obs suggests. That inflates the
    test's apparent power — treat a rejection as a strong signal, a
    non-rejection as weaker reassurance than the p-value implies.
    """
    p = prices.values
    realized_pnl = (p[horizon_days:] - p[:-horizon_days]) * position_oz
    losses = -realized_pnl
    n = len(losses)
    exceptions = int(np.sum(losses > var_threshold))
    rate = exceptions / n
    alpha = 1 - confidence

    p_hat = min(max(rate, 1e-9), 1 - 1e-9)
    a = min(max(alpha, 1e-9), 1 - 1e-9)
    ll_null = (n - exceptions) * np.log(1 - a) + exceptions * np.log(a)
    ll_alt = (n - exceptions) * np.log(1 - p_hat) + exceptions * np.log(p_hat)
    lr_stat = float(-2 * (ll_null - ll_alt))
    p_value = float(1 - chi2.cdf(lr_stat, df=1))

    return BacktestResult(
        confidence=confidence,
        n_obs=n,
        n_exceptions=exceptions,
        exception_rate=rate,
        expected_rate=alpha,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
    )
