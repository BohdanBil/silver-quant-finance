"""VaR / CVaR from simulated P&L, and out-of-sample VaR backtests (Kupiec, Christoffersen)."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
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
    losses = -np.asarray(pnl)
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return RiskMetrics(confidence=confidence, var=var, cvar=cvar)


@dataclass
class CoverageTest:
    n_obs: int
    n_exceptions: int
    exception_rate: float
    expected_rate: float
    kupiec_pvalue: float  # right number of exceptions? (unconditional coverage)
    independence_pvalue: float  # do exceptions cluster? (Christoffersen independence)

    @property
    def rejected(self) -> bool:
        return self.kupiec_pvalue < 0.05 or self.independence_pvalue < 0.05

    @property
    def verdict(self) -> str:
        if self.kupiec_pvalue < 0.05:
            side = "too many" if self.exception_rate > self.expected_rate else "too few"
            return f"REJECTED ({side} exceptions)"
        if self.independence_pvalue < 0.05:
            return "REJECTED (exceptions cluster)"
        return "not rejected"


def coverage_test(exceptions: np.ndarray, expected_rate: float) -> CoverageTest:
    """Kupiec (1995) and Christoffersen (1998) tests on a sequence of VaR exceptions.

    Kupiec's likelihood ratio asks whether the exception *rate* matches
    expected_rate. A model can pass it and still be wrong in a way that
    matters: exceptions arriving in bunches (a constant-vol model in a
    volatile month) rather than independently. Christoffersen's
    independence test compares P(exception | exception yesterday) with
    P(exception | none yesterday). Each LR is chi2 with 1 degree of freedom.

    Both assume the windows don't overlap -- rolling_var_backtest enforces it.
    """
    hits = np.asarray(exceptions, dtype=bool)
    n = len(hits)
    x = int(hits.sum())
    rate = x / n if n else float("nan")

    ll_null = xlogy(n - x, 1 - expected_rate) + xlogy(x, expected_rate)
    ll_alt = xlogy(n - x, 1 - rate) + xlogy(x, rate)
    lr_uc = max(float(-2 * (ll_null - ll_alt)), 0.0)

    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else 0.0
    ll_indep = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_markov = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = max(float(-2 * (ll_indep - ll_markov)), 0.0)

    return CoverageTest(
        n_obs=n,
        n_exceptions=x,
        exception_rate=rate,
        expected_rate=expected_rate,
        kupiec_pvalue=float(chi2.sf(lr_uc, df=1)),
        independence_pvalue=float(chi2.sf(lr_ind, df=1)),
    )


def rolling_var_backtest(
    prices: pd.Series,
    forecast_returns: Callable[[pd.Series], np.ndarray],
    horizon_days: int,
    confidence_levels: Sequence[float],
    min_history: int,
    step: int | None = None,
) -> dict[float, CoverageTest]:
    """Out-of-sample VaR backtest, in return space.

    At each test date t (every `step` days from `min_history` on),
    forecast_returns sees only prices[:t+1] and returns simulated
    horizon_days simple returns. VaR at each confidence is the loss
    quantile of those; an exception is a realized loss
    1 - prices[t+h]/prices[t] worse than it.

    Returns, not dollars: a fixed-oz position's dollar P&L scales with the
    price level, so testing today's dollar VaR against dollar losses from
    years when silver cost a third as much would make nearly every early
    window look safe.

    step defaults to horizon_days so windows never overlap; overlapping
    windows share most of their returns, so their exceptions arrive in
    runs by construction and both coverage tests would be invalid.
    """
    step = horizon_days if step is None else step
    if step < horizon_days:
        raise ValueError("step must be >= horizon_days, or test windows overlap")

    p = prices.to_numpy()
    hits: dict[float, list[bool]] = {c: [] for c in confidence_levels}
    for t in range(min_history, len(p) - horizon_days, step):
        simulated_losses = -forecast_returns(prices.iloc[: t + 1])
        realized_loss = 1 - p[t + horizon_days] / p[t]
        for c in confidence_levels:
            hits[c].append(realized_loss > np.quantile(simulated_losses, c))

    return {c: coverage_test(np.array(h), 1 - c) for c, h in hits.items()}
