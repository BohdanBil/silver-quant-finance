"""Silver forward-return regression models + purged walk-forward out-of-sample validation."""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data import TRADING_DAYS_PER_YEAR

ModelName = str

# LassoCV/ElasticNetCV's auto alpha path includes small-alpha candidates
# that sometimes don't fully converge within max_iter -- worse at short
# horizons, where the target scale (e.g. 1-day returns) shifts the whole
# regularization path. eps=1e-2 below was tuned against 21d-scale targets
# and doesn't fully clear it at 1d/5d. Not independently verified that this
# leaves the *selected* alpha's coefficients unbiased (unlike the Monte
# Carlo option pricer, which was checked this way) -- the practical
# argument is that CV rarely picks the smallest, least-regularized alphas
# on the path anyway, so the unconverged candidates are usually the ones
# losing the internal CV comparison, not the one that ships. Silenced
# because at 1d/5d horizons it produced enough warning spam to bury the
# actual walk-forward results in one run.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Internal CV for the *CV models must respect time order too -- a random
# K-fold would let a training fold sit chronologically after a validation
# fold, leaking future information into the alpha selection itself.
_TSCV = TimeSeriesSplit(n_splits=5)

MOMENTUM_COLUMNS = ["silver_ret_21d"]
AR_COLUMNS = ["silver_ret_1d", "silver_ret_5d", "silver_ret_21d"]
COT_COLUMNS = ["cot_mm_net_pct_oi", "cot_mm_net_pct_oi_chg_21d", "cot_comm_net_pct_oi"]


class ZeroReturnModel:
    """Baseline: always predicts 0 -- the pure random-walk call. This is
    also exactly what oos_r2_vs_naive uses as its denominator, so this
    model's own OOS R^2 is 0.0 by construction; it's included as an
    explicit row so that's a visible anchor in the comparison table, not
    just an invisible constant baked into a formula.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZeroReturnModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class MeanReturnModel:
    """Baseline: always predicts the training set's historical mean return
    (unconditional drift, no timing) -- distinct from ZeroReturnModel
    whenever the sample mean return isn't ~0, e.g. silver's multi-year bull
    run.
    """

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanReturnModel":
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)


class ColumnSubsetLinearModel:
    """Plain OLS restricted to a fixed subset of columns.

    Used for baselines that should only see a narrow, classic signal --
    momentum (last 21d return only) or AR (silver's own recent lags) --
    rather than the full macro/COT feature set the ML models get.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._reg = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnSubsetLinearModel":
        self._reg.fit(X[self.columns], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._reg.predict(X[self.columns])

    @property
    def coef_(self) -> np.ndarray:
        return self._reg.coef_


BASELINE_MODELS = {"zero_return", "historical_mean", "momentum", "ar"}
LINEAR_MODELS = {"ridge", "lasso", "elasticnet", "momentum", "ar"}

MODEL_FACTORIES: dict[ModelName, Callable] = {
    "zero_return": lambda: ZeroReturnModel(),
    "historical_mean": lambda: MeanReturnModel(),
    "momentum": lambda: ColumnSubsetLinearModel(MOMENTUM_COLUMNS),
    "ar": lambda: ColumnSubsetLinearModel(AR_COLUMNS),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "lasso": lambda: make_pipeline(StandardScaler(), LassoCV(cv=_TSCV, alphas=50, eps=1e-2, max_iter=10_000)),
    "elasticnet": lambda: make_pipeline(
        StandardScaler(),
        ElasticNetCV(cv=_TSCV, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50, eps=1e-2, max_iter=10_000),
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=0
    ),
    "gbm": lambda: GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0),
    "xgboost": lambda: XGBRegressor(
        max_depth=3, n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=0,
    ),
}


@dataclass
class WalkForwardResult:
    model_name: str
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    oos_r2_vs_naive: float
    directional_accuracy: float
    directional_pvalue: float
    n_predictions: int


def _compute_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Shared OOS R^2 / directional-accuracy / binomial-p-value calc, used by
    both a full walk-forward run and any post-hoc slice of its predictions
    (e.g. regime_conditional_metrics) -- one definition, so a regime slice
    and the full-sample number are never computed two subtly different ways.
    """
    n = len(actuals)
    ss_res_model = np.sum((actuals - preds) ** 2)
    ss_res_naive = np.sum(actuals**2)
    oos_r2 = 1 - ss_res_model / ss_res_naive if ss_res_naive > 0 else float("nan")

    n_correct = int(np.sum(np.sign(preds) == np.sign(actuals)))
    directional_accuracy = n_correct / n if n > 0 else float("nan")
    p_value = binomtest(n_correct, n, 0.5, alternative="two-sided").pvalue if n > 0 else float("nan")

    return {
        "n_predictions": n,
        "oos_r2_vs_naive": float(oos_r2),
        "directional_accuracy": float(directional_accuracy),
        "directional_pvalue": float(p_value),
    }


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    model_name: ModelName = "ridge",
    min_train_size: int = 504,
    test_step: int | None = None,
) -> WalkForwardResult:
    """Expanding-window walk-forward backtest with a purge gap.

    y[t] already encodes information through t+horizon_days (it's a forward
    return). Refitting naively at every row would let training labels near
    the test point leak information from inside the test point's own
    window. So each step only trains on rows up to (test_index -
    horizon_days).

    test_step controls how far the test point advances each iteration;
    it defaults to horizon_days, which gives non-overlapping test targets
    (the standard, safest choice). A larger test_step is sometimes used to
    subsample test points and cap the number of refits -- e.g. horizon=1
    would otherwise refit at every single row -- while a smaller test_step
    would make consecutive targets overlap and isn't allowed here for the
    same reason the VaR backtest flags overlapping windows as
    autocorrelated: it would inflate the apparent sample size without
    adding real information.

    oos_r2_vs_naive: 1 - SS_res(model) / SS_res(naive), where naive always
    predicts 0 return (pure random walk). This is the standard
    Campbell-Thompson-style out-of-sample R^2 used in the return-
    predictability literature -- positive means the model beat a coin flip
    at guessing the *level* of the return, not just its sign.
    """
    if horizon_days >= min_train_size:
        raise ValueError("min_train_size must exceed horizon_days")
    if test_step is None:
        test_step = horizon_days
    elif test_step < horizon_days:
        raise ValueError("test_step must be >= horizon_days, or targets overlap between test points")

    build_model = MODEL_FACTORIES[model_name]
    n = len(X)

    preds, actuals, dates = [], [], []
    i = min_train_size
    while i < n:
        train_end = i - horizon_days
        model = build_model()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        preds.append(model.predict(X.iloc[[i]])[0])
        actuals.append(y.iloc[i])
        dates.append(X.index[i])
        i += test_step

    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    metrics = _compute_metrics(preds, actuals)

    return WalkForwardResult(
        model_name=model_name,
        dates=pd.DatetimeIndex(dates),
        y_true=actuals,
        y_pred=preds,
        oos_r2_vs_naive=metrics["oos_r2_vs_naive"],
        directional_accuracy=metrics["directional_accuracy"],
        directional_pvalue=metrics["directional_pvalue"],
        n_predictions=metrics["n_predictions"],
    )


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    min_train_size: int = 504,
    model_names: list[ModelName] | None = None,
    test_step: int | None = None,
) -> tuple[pd.DataFrame, dict[ModelName, WalkForwardResult]]:
    """Run walk_forward_validate for every model and rank by OOS R^2.

    Returns (summary_df, results_by_model) so callers can both print/inspect
    the ranking and pull the winner's raw predictions (e.g. for
    economic_significance) without rerunning anything.
    """
    model_names = model_names or list(MODEL_FACTORIES.keys())
    results = {
        name: walk_forward_validate(X, y, horizon_days=horizon_days, model_name=name,
                                     min_train_size=min_train_size, test_step=test_step)
        for name in model_names
    }
    summary = pd.DataFrame(
        {
            "model": name,
            "n_predictions": r.n_predictions,
            "oos_r2_vs_naive": r.oos_r2_vs_naive,
            "directional_accuracy": r.directional_accuracy,
            "p_value": r.directional_pvalue,
        }
        for name, r in results.items()
    ).sort_values("oos_r2_vs_naive", ascending=False).reset_index(drop=True)
    return summary, results


def fit_full_and_explain(X: pd.DataFrame, y: pd.Series, model_name: ModelName = "ridge") -> pd.Series | None:
    """Fit on the full dataset and return per-feature coefficients/importances.

    This is an in-sample fit purely for interpretability (which factors the
    model leans on) -- it says nothing about out-of-sample skill, that's
    what walk_forward_validate is for. Don't read predictive power into it.
    Returns None for zero_return/historical_mean, which don't use features
    at all.
    """
    if model_name in ("zero_return", "historical_mean"):
        return None

    model = MODEL_FACTORIES[model_name]()
    model.fit(X, y)

    if model_name in ("momentum", "ar"):
        return pd.Series(model.coef_, index=model.columns, name="coef").sort_values(key=np.abs, ascending=False)
    if model_name in LINEAR_MODELS:
        coef = model[-1].coef_  # last pipeline step is the linear model
        return pd.Series(coef, index=X.columns, name="standardized_coef").sort_values(key=np.abs, ascending=False)
    return pd.Series(model.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)


def walk_forward_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    model_name: ModelName = "ridge",
    min_train_size: int = 504,
) -> pd.DataFrame:
    """Refit at each walk-forward step (same schedule as walk_forward_validate)
    and record *that step's* feature coefficients/importances, not just one
    full-sample fit.

    A feature that looks important in a single full-sample fit_full_and_explain
    call can still be pure noise: it just happened to correlate with the
    target over that one particular stretch of history. Refitting at every
    step and looking at how much a feature's weight jumps around (or
    flips sign) across sub-periods is how you tell a time-stable
    relationship from an artifact of which years happened to be in the
    training set.
    """
    if model_name in ("zero_return", "historical_mean"):
        raise ValueError(f"{model_name} has no per-feature coefficients to track")

    build_model = MODEL_FACTORIES[model_name]
    columns = MODEL_FACTORIES[model_name]().columns if model_name in ("momentum", "ar") else X.columns
    n = len(X)

    rows, dates = [], []
    i = min_train_size
    while i < n:
        train_end = i - horizon_days
        model = build_model()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        if model_name in ("momentum", "ar"):
            rows.append(model.coef_)
        elif model_name in LINEAR_MODELS:
            rows.append(model[-1].coef_)
        else:
            rows.append(model.feature_importances_)
        dates.append(X.index[i])
        i += horizon_days

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=columns)


def feature_stability_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature stability across walk-forward refits.

    sign_consistency: fraction of refits where the feature's coefficient
    agrees in sign with its own across-refit mean. A feature with a large
    mean importance but sign_consistency near 50% flipped direction
    constantly -- that's the signature of fitting noise in whichever
    period a given refit happened to see, not a real, usable effect.
    """
    mean = importance_df.mean()
    std = importance_df.std()
    majority_sign = np.sign(mean)
    sign_consistency = np.sign(importance_df).eq(majority_sign, axis=1).mean()
    out = pd.DataFrame({"mean": mean, "std": std, "sign_consistency": sign_consistency})
    return out.reindex(out["mean"].abs().sort_values(ascending=False).index)


def ablation_compare(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    drop_columns: list[str],
    model_names: list[ModelName] | None = None,
    min_train_size: int = 504,
) -> pd.DataFrame:
    """Compare OOS R^2 with vs without `drop_columns`, per model.

    This is the direct test of whether a feature group actually helps --
    a nonzero coefficient in one fit only shows the model *used* a
    feature, not that using it made predictions any better out of sample.
    Baselines are excluded by default: zero/historical_mean ignore
    features entirely and momentum/ar never see COT columns to begin
    with, so dropping them can't change those results.
    """
    model_names = model_names or [n for n in MODEL_FACTORIES if n not in BASELINE_MODELS]

    with_summary, _ = compare_models(X, y, horizon_days, min_train_size, model_names)
    without_summary, _ = compare_models(X.drop(columns=drop_columns), y, horizon_days, min_train_size, model_names)

    merged = with_summary.set_index("model")[["oos_r2_vs_naive", "directional_accuracy"]].join(
        without_summary.set_index("model")[["oos_r2_vs_naive", "directional_accuracy"]],
        lsuffix="_with", rsuffix="_without",
    )
    merged["oos_r2_delta"] = merged["oos_r2_vs_naive_with"] - merged["oos_r2_vs_naive_without"]
    return merged.sort_values("oos_r2_delta", ascending=False)


@dataclass
class EconomicSignificanceResult:
    model_name: str
    n_periods: int
    n_position_changes: int
    cost_bps: float
    strategy_total_return: float
    strategy_annualized_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    buy_hold_total_return: float
    buy_hold_annualized_return: float
    buy_hold_sharpe: float


def economic_significance(
    result: WalkForwardResult, horizon_days: int, cost_bps: float = 5.0
) -> EconomicSignificanceResult:
    """Turn walk-forward predictions into a P&L, not just a p-value.

    Rule: go long for the horizon_days block when the model predicts a
    positive return, short when negative (sign(y_pred)); each position flip
    pays a round-trip cost of 2*cost_bps (exit the old side, enter the new
    one). A model can hit a "significant" directional-accuracy p-value and
    still lose money once realistic costs eat the edge -- that's what this
    checks, benchmarked against plain buy-and-hold over the same
    out-of-sample stretch.

    Returns are log-returns, additive across the non-overlapping periods,
    so cumulative P&L is just their running sum; expm1 converts a summed
    log-return back to an arithmetic (%) return for reporting.
    """
    position = np.sign(result.y_pred)
    gross_period_return = position * result.y_true

    position_prev = np.concatenate([[0.0], position[:-1]])
    flipped = position != position_prev
    cost_per_period = flipped * (cost_bps / 10_000) * 2
    net_period_return = gross_period_return - cost_per_period

    periods_per_year = TRADING_DAYS_PER_YEAR / horizon_days

    def _annualize(period_returns: np.ndarray) -> tuple[float, float, float]:
        total = float(np.expm1(np.sum(period_returns)))
        annualized = float(np.expm1(np.mean(period_returns) * periods_per_year))
        std = np.std(period_returns)
        sharpe = float(np.mean(period_returns) / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
        return total, annualized, sharpe

    strategy_total, strategy_annualized, strategy_sharpe = _annualize(net_period_return)
    buy_hold_total, buy_hold_annualized, buy_hold_sharpe = _annualize(result.y_true)

    cum_log = np.cumsum(net_period_return)
    running_max = np.maximum.accumulate(cum_log)
    max_drawdown = float(np.expm1(np.min(cum_log - running_max)))

    return EconomicSignificanceResult(
        model_name=result.model_name,
        n_periods=result.n_predictions,
        n_position_changes=int(np.sum(flipped)),
        cost_bps=cost_bps,
        strategy_total_return=strategy_total,
        strategy_annualized_return=strategy_annualized,
        strategy_sharpe=strategy_sharpe,
        strategy_max_drawdown=max_drawdown,
        buy_hold_total_return=buy_hold_total,
        buy_hold_annualized_return=buy_hold_annualized,
        buy_hold_sharpe=buy_hold_sharpe,
    )


def label_regimes(prices: pd.Series, trend_window: int = 200, vol_window: int = 21) -> pd.DataFrame:
    """Classify each trading day into a trend regime (bull/bear) and a
    volatility regime (high_vol/low_vol).

    trend: price vs. its own trailing trend_window-day SMA -- the standard,
    widely-used bull/bear split (this is literally the "200-day moving
    average" rule).
    vol: trailing vol_window-day realized vol vs. its own full-sample
    median.

    This is a post-hoc label for slicing *already-made* walk-forward
    predictions by market condition, not a feature fed into any model, so
    using the full-sample vol median (rather than only-the-past-up-to-t)
    doesn't leak anything into a forecast -- it only decides which bucket a
    past prediction gets reported under, after the fact.
    """
    sma = prices.rolling(trend_window).mean()
    trend_regime = pd.Series(np.where(prices > sma, "bull", "bear"), index=prices.index)
    trend_regime[sma.isna()] = np.nan

    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol_regime = pd.Series(np.where(realized_vol > realized_vol.median(), "high_vol", "low_vol"), index=prices.index)
    vol_regime[realized_vol.isna()] = np.nan

    return pd.DataFrame({"trend_regime": trend_regime, "vol_regime": vol_regime}, index=prices.index)


def regime_conditional_metrics(result: WalkForwardResult, regimes: pd.DataFrame, regime_column: str,
                                min_n: int = 8) -> pd.DataFrame:
    """Slice an existing WalkForwardResult's predictions by regime label and
    recompute OOS R^2 / directional accuracy / p-value within each bucket --
    no refitting, just partitioning predictions that were already made.

    Regimes with fewer than min_n test points are dropped rather than
    reported with a misleadingly precise-looking (and tiny-sample) metric.
    """
    labels = regimes.loc[result.dates, regime_column]
    rows = []
    for label in sorted(labels.dropna().unique()):
        mask = (labels == label).values
        if mask.sum() < min_n:
            continue
        metrics = _compute_metrics(result.y_pred[mask], result.y_true[mask])
        rows.append({"model": result.model_name, "regime": label, **metrics})
    return pd.DataFrame(rows)
