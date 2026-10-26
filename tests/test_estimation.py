import numpy as np
import pandas as pd
import pytest

import data
from data import (
    TRADING_DAYS_PER_YEAR,
    GBMParams,
    Garch11Params,
    JumpParams,
    _garch11_variance,
    bootstrap_estimates,
    detect_jumps,
    estimate_jumps,
    ewma_volatility,
    fit_garch11,
    garch11_term_volatility,
    log_drift_to_mu,
    period_start,
)
from monte_carlo import simulate_gbm_jump_paths, simulate_gbm_paths


def simulate_garch(n: int, params: Garch11Params, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    var = params.long_run_variance
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = params.omega + params.alpha * r[t] ** 2 + params.beta * var
    return r


def test_ewma_matches_reference_loop():
    r = np.random.default_rng(0).normal(0, 0.02, 500)
    var = r[0] ** 2
    for ret in r[1:]:
        var = 0.94 * var + 0.06 * ret**2
    assert ewma_volatility(r, 0.94) == pytest.approx(np.sqrt(var * TRADING_DAYS_PER_YEAR), rel=1e-12)


def test_garch_variance_filter_matches_reference_loop():
    r = np.random.default_rng(1).normal(0, 0.02, 300)
    omega, alpha, beta = 3e-6, 0.07, 0.9
    expected = np.empty_like(r)
    expected[0] = np.var(r)
    for t in range(1, len(r)):
        expected[t] = omega + alpha * r[t - 1] ** 2 + beta * expected[t - 1]
    np.testing.assert_allclose(_garch11_variance(omega, alpha, beta, r), expected, rtol=1e-12)


def test_garch_fit_recovers_known_parameters():
    true = Garch11Params(omega=4e-6, alpha=0.08, beta=0.90)
    fit = fit_garch11(simulate_garch(6_000, true, seed=3))
    assert fit.alpha == pytest.approx(true.alpha, abs=0.03)
    assert fit.beta == pytest.approx(true.beta, abs=0.04)
    assert fit.long_run_variance == pytest.approx(true.long_run_variance, rel=0.3)


def test_garch_term_volatility_reverts_to_long_run():
    fit = Garch11Params(omega=4e-6, alpha=0.08, beta=0.90)
    calm = np.full(300, 0.001)
    calm_one_day = garch11_term_volatility(calm, fit, horizon_days=1)
    long_run = np.sqrt(fit.long_run_variance * TRADING_DAYS_PER_YEAR)
    assert calm_one_day < garch11_term_volatility(calm, fit, horizon_days=63) < long_run
    assert garch11_term_volatility(calm, fit, horizon_days=5_000) == pytest.approx(long_run, rel=0.02)


def test_simulated_drift_matches_the_history_it_was_calibrated_on():
    """mu from log_drift_to_mu must make simulated mean log returns equal the historical ones."""
    mean_log, sigma = 0.10, 0.40
    params = GBMParams(s0=50.0, mu=log_drift_to_mu(mean_log, sigma), sigma=sigma)
    paths = simulate_gbm_paths(params, horizon_days=252, n_sims=40_000, seed=0)
    assert np.mean(np.log(paths[:, -1] / paths[:, 0])) == pytest.approx(mean_log, abs=0.01)


def test_jump_diffusion_drift_and_compensation():
    jumps = JumpParams(intensity=3.0, mean=-0.05, std=0.08, n_detected=0)
    mean_log, sigma = 0.08, 0.30
    params = GBMParams(s0=50.0, mu=log_drift_to_mu(mean_log, sigma, jumps), sigma=sigma)
    paths = simulate_gbm_jump_paths(params, jumps, horizon_days=252, n_sims=60_000, seed=0)
    terminal = paths[:, -1] / paths[:, 0]
    assert np.mean(np.log(terminal)) == pytest.approx(mean_log, abs=0.01)
    assert np.mean(terminal) == pytest.approx(np.exp(params.mu), rel=0.01)  # E[S_T] = s0 * e^{mu T}


def test_detect_jumps_finds_planted_jumps_without_flagging_a_vol_regime():
    rng = np.random.default_rng(7)
    vol = np.where(np.arange(1_500) < 750, 0.01, 0.03)  # calm half, then a volatile half
    r = rng.standard_normal(1_500) * vol
    planted = [200, 600, 900, 1_300]
    r[planted] = [-0.15, 0.12, -0.25, 0.20]
    flagged = np.flatnonzero(detect_jumps(r, threshold=4.0))
    assert set(planted) <= set(flagged)
    assert len(flagged) <= len(planted) + 3


def test_estimate_jumps_separates_diffusion_vol():
    rng = np.random.default_rng(11)
    r = rng.normal(0, 0.015, 2_520)
    r[[100, 800, 1_500, 2_200]] = [-0.2, -0.15, 0.18, -0.25]
    jumps, diffusion = estimate_jumps(r)
    assert jumps.n_detected >= 4
    assert jumps.intensity == pytest.approx(jumps.n_detected / 10, rel=1e-9)
    assert np.std(diffusion) == pytest.approx(0.015, rel=0.05)
    assert np.std(r) > 1.1 * np.std(diffusion)  # the raw std is inflated by the jumps


def test_bootstrap_estimates_centered_on_sample_and_uses_sigma_returns():
    r = np.random.default_rng(2).normal(0.0004, 0.02, 1_000)
    boot = bootstrap_estimates(r, n_boot=3_000, block_size=21, seed=0)
    assert boot["mean_log"].mean() == pytest.approx(r.mean() * TRADING_DAYS_PER_YEAR, abs=0.03)
    assert boot["sigma"].mean() == pytest.approx(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR), rel=0.02)
    halved = bootstrap_estimates(r, n_boot=500, block_size=21, seed=0, sigma_returns=r / 2)
    plain = bootstrap_estimates(r, n_boot=500, block_size=21, seed=0)
    np.testing.assert_allclose(halved["sigma"], plain["sigma"] / 2)
    np.testing.assert_allclose(halved["mean_log"], plain["mean_log"])


def test_period_start():
    end = pd.Timestamp("2026-10-23")
    assert period_start(end, "5y") == pd.Timestamp("2021-10-23")
    assert period_start(end, "18mo") == pd.Timestamp("2025-04-23")
    with pytest.raises(ValueError):
        period_start(end, "5 years")


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]), name="x")

    first = data.cached("yf_SI=F_2021-01-01_2026-01-02", fetch)
    second = data.cached("yf_SI=F_2021-01-01_2026-01-02", fetch)
    pd.testing.assert_series_equal(first, second)
    assert len(calls) == 1
