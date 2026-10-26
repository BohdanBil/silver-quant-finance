import math

import numpy as np
import pandas as pd
import pytest

import forecast_model
from features import engineer_features
from forecast_data import BREAKEVEN_INFLATION_10Y, USD_INDEX, _asof_align, as_published
from forecast_model import (
    WalkForwardResult,
    build_model,
    economic_significance,
    feature_stability_summary,
    forecast_metrics,
    walk_forward_coefficients,
    walk_forward_validate,
)


def synthetic_frame(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    walk = lambda scale: np.exp(np.cumsum(rng.normal(0, scale, n)))
    return pd.DataFrame({
        "silver": 20 * walk(0.02),
        "gold": 1500 * walk(0.01),
        "real_rate": np.cumsum(rng.normal(0, 0.03, n)),
        "breakeven_inflation": 2 + np.cumsum(rng.normal(0, 0.02, n)),
        "usd_index": 100 * walk(0.004),
        "cot_mm_net_pct_oi": np.clip(np.cumsum(rng.normal(0, 0.01, n)), -0.5, 0.5),
        "cot_comm_net_pct_oi": np.clip(np.cumsum(rng.normal(0, 0.01, n)), -0.5, 0.5),
    }, index=idx)


def test_features_never_look_ahead():
    frame = synthetic_frame()
    X_full, _ = engineer_features(frame, horizon_days=21)
    X_cut, _ = engineer_features(frame.iloc[:600], horizon_days=21)
    common = X_cut.index.intersection(X_full.index)
    assert len(common) > 400
    pd.testing.assert_frame_equal(X_full.loc[common], X_cut.loc[common])


def test_target_is_the_forward_log_return():
    frame = synthetic_frame()
    _, y = engineer_features(frame, horizon_days=5)
    t = y.index[100]
    pos = frame.index.get_loc(t)
    assert y.loc[t] == pytest.approx(np.log(frame["silver"].iloc[pos + 5] / frame["silver"].iloc[pos]))


class SpyModel:
    calls: list[tuple[pd.Timestamp, int]] = []

    def fit(self, X, y):
        self.last_train_date = X.index[-1]
        return self

    def predict(self, X):
        SpyModel.calls.append((self.last_train_date, X.index[0]))
        return np.zeros(len(X))


def test_walk_forward_purges_training_labels_that_overlap_the_test_window(monkeypatch):
    frame = synthetic_frame()
    X, y = engineer_features(frame, horizon_days=21)
    monkeypatch.setitem(forecast_model.MODEL_FACTORIES, "spy", lambda h: SpyModel())
    SpyModel.calls = []
    walk_forward_validate(X, y, horizon_days=21, model_name="spy", min_train_size=300)
    assert SpyModel.calls
    for last_train, test_date in SpyModel.calls:
        # the last training label covers last_train .. last_train + 21 rows; it must end before the test date
        assert X.index.get_loc(last_train) + 21 < X.index.get_loc(test_date)


def test_internal_cv_has_a_horizon_gap():
    assert build_model("lasso", 21)[-1].cv.gap == 21
    assert build_model("elasticnet", 63)[-1].cv.gap == 63


def test_coefficient_tracking_refuses_unsigned_tree_importances():
    frame = synthetic_frame()
    X, y = engineer_features(frame, horizon_days=21)
    with pytest.raises(ValueError):
        walk_forward_coefficients(X, y, horizon_days=21, model_name="random_forest", min_train_size=300)
    coefs = walk_forward_coefficients(X, y, horizon_days=21, model_name="ridge", min_train_size=300)
    assert list(coefs.columns) == list(X.columns)


def test_feature_stability_summary():
    coefs = pd.DataFrame({
        "stable": [1.0, 2.0, 1.5, 0.5],
        "flips": [1.0, -1.0, 1.2, -1.1],
        "never_selected": [0.0, 0.0, 0.0, 0.0],
        "sometimes_dropped": [1.0, 0.0, 1.0, 0.0],
    })
    s = feature_stability_summary(coefs)["sign_consistency"]
    assert s["stable"] == 1.0
    assert s["flips"] == 0.5
    assert math.isnan(s["never_selected"])
    assert s["sometimes_dropped"] == 0.5


def make_result(preds, actuals) -> WalkForwardResult:
    preds, actuals = np.asarray(preds, float), np.asarray(actuals, float)
    return WalkForwardResult("m", pd.bdate_range("2020-01-01", periods=len(preds)), actuals, preds,
                             np.zeros(len(preds)), forecast_metrics(preds, actuals, np.zeros(len(preds))))


def test_economic_significance_costs():
    actuals = np.array([0.02, -0.01, 0.03, 0.01])
    always_long = economic_significance(make_result(np.ones(4), actuals), horizon_days=21, cost_bps=0)
    assert always_long.strategy_total_return == pytest.approx(always_long.buy_hold_total_return)

    with_cost = economic_significance(make_result(np.ones(4), actuals), horizon_days=21, cost_bps=10)
    assert with_cost.n_trades == 1  # one entry, then held
    assert with_cost.strategy_total_return == pytest.approx(np.expm1(actuals.sum() - 0.001))

    flip = economic_significance(make_result([1, -1, 1, -1], actuals), horizon_days=21, cost_bps=10)
    gross = np.array([1, -1, 1, -1]) * actuals
    assert flip.strategy_total_return == pytest.approx(np.expm1(gross.sum() - 0.001 - 3 * 0.002))


def test_max_drawdown_counts_losses_from_the_start():
    result = economic_significance(make_result(np.ones(3), [-0.10, 0.02, 0.01]), horizon_days=21, cost_bps=0)
    assert result.strategy_max_drawdown == pytest.approx(np.expm1(-0.10))


def test_forecast_metrics():
    actuals = np.random.default_rng(0).normal(0, 1, 100)
    perfect = forecast_metrics(actuals, actuals, np.zeros(100))
    assert perfect["oos_r2_vs_zero"] == pytest.approx(1.0)
    zero = forecast_metrics(np.zeros(100), actuals, np.zeros(100))
    assert zero["oos_r2_vs_zero"] == 0.0
    assert math.isnan(zero["cw_pvalue"]) and math.isnan(zero["pt_pvalue"])


def test_publication_dates():
    obs = pd.DatetimeIndex(["2026-10-19", "2026-10-23"])  # a Monday and a Friday
    usd = as_published(pd.Series([1.0, 2.0], index=obs, name=USD_INDEX))
    assert list(usd.index) == [pd.Timestamp("2026-10-27")]  # both land in the next Monday's release
    assert usd.iloc[0] == 2.0  # and the later observation in it wins
    daily = as_published(pd.Series([1.0, 2.0], index=obs, name=BREAKEVEN_INFLATION_10Y))
    assert list(daily.index) == [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-26")]


def test_asof_alignment_never_uses_a_later_value():
    published = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-10-20", "2026-10-26"]))
    target = pd.bdate_range("2026-10-19", "2026-10-27")
    aligned = _asof_align(published, target)
    assert math.isnan(aligned.loc["2026-10-19"])
    assert aligned.loc["2026-10-23"] == 1.0
    assert aligned.loc["2026-10-26"] == 2.0
