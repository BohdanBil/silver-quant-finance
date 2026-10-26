"""Forecast-evaluation tests: Clark-West (forecast error), Pesaran-Timmermann (direction),
and multiple-testing adjustments."""

import numpy as np
from scipy.stats import norm


def clark_west(actuals: np.ndarray, preds: np.ndarray, bench_preds: np.ndarray) -> float:
    """One-sided p-value of Clark & West (2007): does the model's MSPE beat a nested benchmark's?

    A raw MSPE comparison (and so OOS R^2) is biased against the larger
    model: even when its extra predictors are useless, it still pays for
    estimating their noise. CW adds back the (bench - model)^2 term that
    noise contributes and t-tests the adjusted loss differential
        f_t = e_bench^2 - (e_model^2 - (bench - model)^2).
    H1: the model is better. NaN when the model is the benchmark (f_t = 0).

    Assumes non-overlapping forecast targets, as walk_forward_validate
    produces; overlapping ones would need a HAC standard error.
    """
    y, f_model, f_bench = (np.asarray(a, dtype=float) for a in (actuals, preds, bench_preds))
    f = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    n = len(f)
    sd = f.std(ddof=1) if n > 1 else 0.0
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(norm.sf(f.mean() / (sd / np.sqrt(n))))


def pesaran_timmermann(actuals: np.ndarray, preds: np.ndarray) -> float:
    """One-sided p-value of Pesaran & Timmermann (1992): is there real directional skill?

    Compares the hit rate with what independent guessing would score given
    how often the predictions and the outcomes are each "up":
    P* = Py*Px + (1-Py)*(1-Px). A binomial test against 50% instead credits
    an always-long model in a bull market with "significant" accuracy it
    did nothing to earn. A prediction of exactly 0 counts as "not up".
    NaN when every prediction (or every outcome) has the same sign -- there
    is no timing to test.
    """
    up_pred = np.asarray(preds) > 0
    up_act = np.asarray(actuals) > 0
    n = len(up_act)
    px, py = up_pred.mean(), up_act.mean()
    if n == 0 or px in (0.0, 1.0) or py in (0.0, 1.0):
        return float("nan")

    hit_rate = np.mean(up_pred == up_act)
    p_star = py * px + (1 - py) * (1 - px)
    var_hit = p_star * (1 - p_star) / n
    var_p_star = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n \
        + 4 * py * px * (1 - py) * (1 - px) / n**2
    denom = var_hit - var_p_star
    if denom <= 0:
        return float("nan")
    return float(norm.sf((hit_rate - p_star) / np.sqrt(denom)))


def _adjust(pvalues: np.ndarray, adjust_sorted) -> np.ndarray:
    """Apply a sorted-p-value adjustment to the non-NaN entries; NaNs are outside the family and stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return out
    q = p[valid]
    order = np.argsort(q)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjust_sorted(q[order], m), 1.0)
    out[valid] = adjusted
    return out


def bonferroni(pvalues: np.ndarray) -> np.ndarray:
    return _adjust(pvalues, lambda q, m: q * m)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values: family-wise error control, never less powerful than Bonferroni."""
    return _adjust(pvalues, lambda q, m: np.maximum.accumulate((m - np.arange(m)) * q))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values: controls the false discovery rate rather than
    the chance of any false positive, so it is less conservative across many correlated tests."""
    return _adjust(pvalues, lambda q, m: np.minimum.accumulate((q * m / np.arange(1, m + 1))[::-1])[::-1])
