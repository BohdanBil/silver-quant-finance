"""Feature engineering: raw price/macro levels -> stationary predictors + forward-return target."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _log_ret(s: pd.Series, periods: int) -> pd.Series:
    return np.log(s / s.shift(periods))


def engineer_features(frame: pd.DataFrame, horizon_days: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    """Build stationary features from build_feature_frame()'s raw levels, plus
    the horizon_days-ahead silver log-return target.

    Every feature at row t uses only information available at or before t
    (rolling/diff/pct_change windows all look backward). The target at row
    t is log(silver[t+horizon_days] / silver[t]) -- the one deliberately
    forward-looking column, and it's what's being predicted, not a feature.
    Raw price/rate *levels* are skipped on purpose: they're non-stationary
    (a linear model would partly just be relearning "silver was ~$65
    lately"), so only returns/changes/ratios go into X.
    """
    silver = frame["silver"]
    gold = frame["gold"]

    X = pd.DataFrame(index=frame.index)
    X["silver_ret_1d"] = _log_ret(silver, 1)
    X["silver_ret_5d"] = _log_ret(silver, 5)
    X["silver_ret_21d"] = _log_ret(silver, 21)
    X["silver_vol_21d"] = _log_ret(silver, 1).rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    X["gold_ret_21d"] = _log_ret(gold, 21)
    X["gold_silver_ratio_chg_21d"] = (gold / silver).pct_change(21)
    X["real_rate_chg_21d"] = frame["real_rate"].diff(21)
    X["breakeven_inflation_chg_21d"] = frame["breakeven_inflation"].diff(21)
    X["usd_index_ret_21d"] = _log_ret(frame["usd_index"], 21)
    # net position / open interest is already a bounded ratio (roughly
    # stationary), so the level itself is informative -- unlike price/rate
    # levels, it's not just relearning "where things happened to be lately".
    X["cot_mm_net_pct_oi"] = frame["cot_mm_net_pct_oi"]
    X["cot_mm_net_pct_oi_chg_21d"] = frame["cot_mm_net_pct_oi"].diff(21)
    X["cot_comm_net_pct_oi"] = frame["cot_comm_net_pct_oi"]

    y = _log_ret(silver, horizon_days).shift(-horizon_days)
    y.name = f"fwd_ret_{horizon_days}d"

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]
