import math

import numpy as np
from scipy.stats import binomtest

from significance import benjamini_hochberg, bonferroni, clark_west, holm, pesaran_timmermann


def test_pesaran_timmermann_is_undefined_for_a_constant_prediction():
    actuals = np.random.default_rng(0).normal(0.01, 0.05, 200)
    assert math.isnan(pesaran_timmermann(actuals, np.ones(200)))
    assert math.isnan(pesaran_timmermann(actuals, np.zeros(200)))


def test_pesaran_timmermann_does_not_reward_riding_the_base_rate():
    """In a 70%-up market, a mostly-long guesser with no timing looks great to a 50% binomial test."""
    rng = np.random.default_rng(1)
    binomial_rejects, pt_rejects = [], []
    for _ in range(500):
        actuals = np.where(rng.random(400) < 0.7, 1.0, -1.0)
        preds = np.where(rng.random(400) < 0.9, 1.0, -1.0)  # independent of actuals
        hits = int(np.sum(np.sign(preds) == np.sign(actuals)))
        binomial_rejects.append(binomtest(hits, 400, 0.5, alternative="greater").pvalue < 0.05)
        pt_rejects.append(pesaran_timmermann(actuals, preds) < 0.05)
    assert np.mean(binomial_rejects) > 0.99
    assert np.mean(pt_rejects) < 0.08  # ~5% false positives, as a 5% test should have


def test_pesaran_timmermann_detects_real_skill():
    rng = np.random.default_rng(2)
    actuals = rng.normal(0, 1, 300)
    preds = actuals + rng.normal(0, 1, 300)
    assert pesaran_timmermann(actuals, preds) < 1e-6


def test_clark_west():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 300)
    actuals = 0.5 * x + rng.normal(0, 1, 300)
    zeros = np.zeros(300)
    assert math.isnan(clark_west(actuals, zeros, zeros))
    assert clark_west(actuals, 0.5 * x, zeros) < 1e-4
    assert clark_west(actuals, 0.05 * rng.normal(0, 1, 300), zeros) > 0.01


P = np.array([0.01, 0.04, 0.03, 0.005, np.nan])


def test_bonferroni():
    np.testing.assert_allclose(bonferroni(P), [0.04, 0.16, 0.12, 0.02, np.nan])


def test_holm():
    np.testing.assert_allclose(holm(P), [0.03, 0.06, 0.06, 0.02, np.nan])


def test_benjamini_hochberg():
    np.testing.assert_allclose(benjamini_hochberg(P), [0.02, 0.04, 0.04, 0.02, np.nan])


def test_adjusted_pvalues_are_capped_and_ordered():
    p = np.random.default_rng(4).random(50)
    for adjust in (bonferroni, holm, benjamini_hochberg):
        a = adjust(p)
        assert np.all((a >= p) & (a <= 1))
        assert np.all(np.diff(a[np.argsort(p)]) >= -1e-15)
    assert np.all(holm(p) <= bonferroni(p))
    assert np.all(benjamini_hochberg(p) <= holm(p) + 1e-15)
