import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk import coverage_test, rolling_var_backtest, var_cvar
from risk_model import RiskModelSpec, RollingForecaster, calibrate


def gbm_prices(n: int, sigma_daily: float = 0.02, s0: float = 25.0, drift: float = 0.0005,
               seed: int = 0) -> pd.Series:
    r = np.random.default_rng(seed).normal(drift - 0.5 * sigma_daily**2, sigma_daily, n)
    return pd.Series(s0 * np.exp(np.cumsum(r)), index=pd.bdate_range("2015-01-01", periods=n))


def test_var_cvar_of_a_standard_normal():
    pnl = np.random.default_rng(0).standard_normal(400_000)
    m = var_cvar(pnl, 0.95)
    assert m.var == pytest.approx(norm.ppf(0.95), abs=0.01)
    assert m.cvar == pytest.approx(norm.pdf(norm.ppf(0.95)) / 0.05, abs=0.01)


def test_coverage_test_accepts_the_right_rate_and_rejects_the_wrong_one():
    rng = np.random.default_rng(0)
    good = coverage_test(rng.random(2_000) < 0.01, 0.01)
    bad = coverage_test(rng.random(2_000) < 0.03, 0.01)
    assert not good.rejected
    assert bad.kupiec_pvalue < 1e-4 and bad.verdict.startswith("REJECTED (too many")


def test_coverage_test_detects_clustering_at_the_right_rate():
    hits = np.zeros(1_000, dtype=bool)
    for start in range(50, 1_000, 100):  # 10 bursts of 5 consecutive exceptions = 5% overall
        hits[start:start + 5] = True
    result = coverage_test(hits, 0.05)
    assert result.kupiec_pvalue > 0.5
    assert result.independence_pvalue < 1e-6
    assert result.verdict == "REJECTED (exceptions cluster)"


def test_coverage_test_with_no_exceptions():
    result = coverage_test(np.zeros(100, dtype=bool), 0.01)
    assert result.n_exceptions == 0
    assert 0 < result.kupiec_pvalue < 1


def test_rolling_backtest_passes_a_correct_model():
    sigma = 0.02
    prices = gbm_prices(2_600, sigma_daily=sigma, seed=4)
    rng = np.random.default_rng(1)

    def oracle(history: pd.Series) -> np.ndarray:  # the true 1-day return distribution
        return np.exp(rng.normal(0.0005 - 0.5 * sigma**2, sigma, 20_000)) - 1

    results = rolling_var_backtest(prices, oracle, horizon_days=1, confidence_levels=[0.95, 0.99], min_history=500)
    for c, bt in results.items():
        assert bt.n_obs == 2_099
        assert bt.kupiec_pvalue > 0.01, c


def test_rolling_backtest_is_price_level_free():
    """Same returns at 3x the price level must give identical exceptions (the old dollar backtest didn't)."""
    prices = gbm_prices(900, seed=5)
    spec = RiskModelSpec()
    run = lambda p: rolling_var_backtest(p, RollingForecaster(spec, 1, 2_000, window=250, seed=0),
                                         horizon_days=1, confidence_levels=[0.99], min_history=250)[0.99]
    assert run(prices).n_exceptions == run(prices * 3).n_exceptions


def test_rolling_backtest_only_shows_the_forecaster_the_past():
    prices = gbm_prices(400, seed=6)
    seen_last_dates = []

    def spy(history: pd.Series) -> np.ndarray:
        seen_last_dates.append(history.index[-1])
        return np.random.default_rng(0).normal(0, 0.02, 1_000)

    rolling_var_backtest(prices, spy, horizon_days=5, confidence_levels=[0.95], min_history=100)
    expected = prices.index[list(range(100, 400 - 5, 5))]
    assert list(seen_last_dates) == list(expected)


def test_rolling_backtest_rejects_overlapping_windows():
    with pytest.raises(ValueError):
        rolling_var_backtest(gbm_prices(300), lambda h: np.zeros(10), horizon_days=5,
                             confidence_levels=[0.95], min_history=100, step=1)


def test_rolling_forecaster_ignores_data_before_its_window():
    prices = gbm_prices(600, seed=7)
    altered = prices.copy()
    altered.iloc[:300] *= np.random.default_rng(0).uniform(0.5, 1.5, 300)  # only pre-window history changes
    for calibration in ("point", "ewma", "garch", "bootstrap"):
        spec = RiskModelSpec(calibration=calibration, model="gbm-jump")
        a = RollingForecaster(spec, 1, 1_000, window=250, seed=0)(prices)
        b = RollingForecaster(spec, 1, 1_000, window=250, seed=0)(altered)
        np.testing.assert_allclose(a, b, err_msg=calibration)


def test_garch_calibration_uses_horizon_averaged_vol():
    """Right after a shock, tomorrow's GARCH vol is high but a long horizon's average reverts toward normal."""
    rng = np.random.default_rng(8)
    r, var = np.empty(1_200), 1e-4
    for t in range(1_200):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = 5e-6 + 0.1 * r[t] ** 2 + 0.85 * var
    r[-3:] = [-0.07, 0.06, -0.08]
    prices = pd.Series(30 * np.exp(np.cumsum(r)), index=pd.bdate_range("2020-01-01", periods=1_200))
    spec = RiskModelSpec(calibration="garch")
    short = calibrate(prices, spec, horizon_days=1, n_sims=100)
    long = calibrate(prices, spec, horizon_days=250, n_sims=100)
    assert short.params.sigma > 1.2 * long.params.sigma
