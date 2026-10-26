"""Silver forward-return regression models + purged walk-forward out-of-sample validation."""

import warnings
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data import TRADING_DAYS_PER_YEAR
from significance import clark_west, pesaran_timmermann

ModelName = str

MOMENTUM_COLUMNS = ["silver_ret_21d"]
AR_COLUMNS = ["silver_ret_1d", "silver_ret_5d", "silver_ret_21d"]
COT_COLUMNS = ["cot_mm_net_pct_oi", "cot_mm_net_pct_oi_chg_21d", "cot_comm_net_pct_oi"]


class ZeroReturnModel:
    """Baseline: always predicts 0 -- the pure random-walk call. This is
    also exactly what oos_r2_vs_zero uses as its denominator, so this
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
    run. Also the benchmark behind oos_r2_vs_mean.
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


def _time_series_cv(horizon_days: int) -> TimeSeriesSplit:
    """Internal CV for the *CV models' alpha selection.

    Folds must respect time order, and need a horizon_days gap: training
    labels are forward returns, so without one the last rows of each
    training fold have targets that overlap the validation fold's period,
    leaking it into the choice of alpha.
    """
    return TimeSeriesSplit(n_splits=5, gap=horizon_days)


BASELINE_MODELS = {"zero_return", "historical_mean", "momentum", "ar"}
LINEAR_MODELS = {"ridge", "lasso", "elasticnet", "momentum", "ar"}
SIGNED_ML_MODELS = ["ridge", "lasso", "elasticnet"]  # non-baseline models with signed coefficients

MODEL_FACTORIES: dict[ModelName, Callable[[int], object]] = {
    "zero_return": lambda h: ZeroReturnModel(),
    "historical_mean": lambda h: MeanReturnModel(),
    "momentum": lambda h: ColumnSubsetLinearModel(MOMENTUM_COLUMNS),
    "ar": lambda h: ColumnSubsetLinearModel(AR_COLUMNS),
    "ridge": lambda h: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "lasso": lambda h: make_pipeline(
        StandardScaler(), LassoCV(cv=_time_series_cv(h), alphas=50, eps=1e-2, max_iter=10_000)
    ),
    "elasticnet": lambda h: make_pipeline(
        StandardScaler(),
        ElasticNetCV(cv=_time_series_cv(h), l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50, eps=1e-2, max_iter=10_000),
    ),
    "random_forest": lambda h: RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=0
    ),
    "gbm": lambda h: GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0),
    "xgboost": lambda h: XGBRegressor(
        max_depth=3, n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=0,
    ),
}


def build_model(model_name: ModelName, horizon_days: int):
    return MODEL_FACTORIES[model_name](horizon_days)


def _fit(model, X: pd.DataFrame, y: pd.Series):
    """Fit with sklearn's ConvergenceWarning silenced, for this call only.

    LassoCV/ElasticNetCV's alpha path includes small-alpha candidates that
    don't always converge within max_iter, mostly at 1d/5d horizons where
    the target's scale shifts the path. Those barely-regularized candidates
    are rarely the ones internal CV picks, and the warnings otherwise bury
    the walk-forward output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


def _coefficients(model, model_name: ModelName) -> np.ndarray:
    if model_name in ("momentum", "ar"):
        return model.coef_
    return model[-1].coef_  # last pipeline step is the linear model (standardized coefficients)


def _walk_forward_steps(n: int, horizon_days: int, min_train_size: int,
                        test_step: int | None) -> Iterator[tuple[int, int]]:
    """(test_index, train_end) pairs for an expanding-window walk-forward with a purge gap.

    y[t] already encodes information through t+horizon_days (it's a forward
    return), so each step only trains on rows before test_index -
    horizon_days: any later training label would overlap the test point's
    own window.

    test_step controls how far the test point advances each iteration; it
    defaults to horizon_days, which gives non-overlapping test targets (the
    standard, safest choice). A larger test_step subsamples test points to
    cap the number of refits -- horizon=1 would otherwise refit at every
    single row. A smaller one would make consecutive targets overlap,
    inflating the apparent sample size without adding information, so it
    isn't allowed.
    """
    if horizon_days >= min_train_size:
        raise ValueError("min_train_size must exceed horizon_days")
    if test_step is None:
        test_step = horizon_days
    elif test_step < horizon_days:
        raise ValueError("test_step must be >= horizon_days, or targets overlap between test points")
    for i in range(min_train_size, n, test_step):
        yield i, i - horizon_days


def forecast_metrics(preds: np.ndarray, actuals: np.ndarray, bench_mean: np.ndarray) -> dict:
    """Out-of-sample metrics shared by a full walk-forward run and any post-hoc
    slice of its predictions (e.g. regime_conditional_metrics), so the two are
    never computed two subtly different ways.

    oos_r2_vs_zero: 1 - SSE(model) / SSE(predict 0), the random-walk
    benchmark (as in Gu, Kelly & Xiu 2020). oos_r2_vs_mean: against the
    expanding historical mean instead (Campbell & Thompson 2008) -- a
    tougher bar whenever the sample has a strong drift. cw_pvalue:
    Clark-West test of the model against the random walk. pt_pvalue:
    Pesaran-Timmermann test of directional skill. Both one-sided.
    """
    n = len(actuals)
    sse_model = np.sum((actuals - preds) ** 2)
    sse_zero = np.sum(actuals**2)
    sse_mean = np.sum((actuals - bench_mean) ** 2)
    return {
        "n_predictions": n,
        "oos_r2_vs_zero": float(1 - sse_model / sse_zero) if sse_zero > 0 else float("nan"),
        "oos_r2_vs_mean": float(1 - sse_model / sse_mean) if sse_mean > 0 else float("nan"),
        "cw_pvalue": clark_west(actuals, preds, np.zeros(n)),
        "directional_accuracy": float(np.mean((preds > 0) == (actuals > 0))) if n else float("nan"),
        "pt_pvalue": pesaran_timmermann(actuals, preds),
    }


@dataclass
class WalkForwardResult:
    model_name: str
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    y_bench_mean: np.ndarray  # the historical-mean benchmark's prediction at each test point
    metrics: dict


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    model_name: ModelName = "ridge",
    min_train_size: int = 504,
    test_step: int | None = None,
) -> WalkForwardResult:
    """Expanding-window walk-forward backtest with a purge gap (see _walk_forward_steps)."""
    preds, actuals, bench, dates = [], [], [], []
    for i, train_end in _walk_forward_steps(len(X), horizon_days, min_train_size, test_step):
        model = _fit(build_model(model_name, horizon_days), X.iloc[:train_end], y.iloc[:train_end])
        preds.append(model.predict(X.iloc[[i]])[0])
        actuals.append(y.iloc[i])
        bench.append(y.iloc[:train_end].mean())
        dates.append(X.index[i])

    preds, actuals, bench = np.asarray(preds), np.asarray(actuals), np.asarray(bench)
    return WalkForwardResult(
        model_name=model_name,
        dates=pd.DatetimeIndex(dates),
        y_true=actuals,
        y_pred=preds,
        y_bench_mean=bench,
        metrics=forecast_metrics(preds, actuals, bench),
    )


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    min_train_size: int = 504,
    model_names: list[ModelName] | None = None,
    test_step: int | None = None,
) -> tuple[pd.DataFrame, dict[ModelName, WalkForwardResult]]:
    """Run walk_forward_validate for every model and rank by OOS R^2 vs the random walk.

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
        {"model": name, **r.metrics} for name, r in results.items()
    ).sort_values("oos_r2_vs_zero", ascending=False).reset_index(drop=True)
    return summary, results


def fit_full_and_explain(X: pd.DataFrame, y: pd.Series, horizon_days: int,
                         model_name: ModelName = "ridge") -> pd.Series | None:
    """Fit on the full dataset and return per-feature coefficients/importances.

    This is an in-sample fit purely for interpretability (which factors the
    model leans on) -- it says nothing about out-of-sample skill, that's
    what walk_forward_validate is for. Don't read predictive power into it.
    Returns None for zero_return/historical_mean, which don't use features
    at all.
    """
    if model_name in ("zero_return", "historical_mean"):
        return None

    model = _fit(build_model(model_name, horizon_days), X, y)
    if model_name in ("momentum", "ar"):
        return pd.Series(model.coef_, index=model.columns, name="coef").sort_values(key=np.abs, ascending=False)
    if model_name in LINEAR_MODELS:
        return pd.Series(_coefficients(model, model_name), index=X.columns,
                         name="standardized_coef").sort_values(key=np.abs, ascending=False)
    return pd.Series(model.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)


def walk_forward_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_days: int,
    model_name: ModelName = "ridge",
    min_train_size: int = 504,
    test_step: int | None = None,
) -> pd.DataFrame:
    """Refit on the walk-forward schedule and record *each refit's*
    coefficients, not just one full-sample fit.

    A feature that looks important in a single full-sample
    fit_full_and_explain call can still be pure noise: it just happened to
    correlate with the target over that one particular stretch of history.
    How much its coefficient jumps around (or flips sign) across refits is
    how you tell a time-stable relationship from an artifact of which years
    happened to be in the training set.

    Linear models only: tree models' feature_importances_ are unsigned
    (always >= 0), so a sign-stability check on them would report a
    perfectly stable direction no matter what.
    """
    if model_name not in LINEAR_MODELS:
        raise ValueError(f"{model_name} has no signed per-feature coefficients to track")

    if model_name in ("momentum", "ar"):
        columns = list(build_model(model_name, horizon_days).columns)
    else:
        columns = list(X.columns)
    rows, dates = [], []
    for i, train_end in _walk_forward_steps(len(X), horizon_days, min_train_size, test_step):
        model = _fit(build_model(model_name, horizon_days), X.iloc[:train_end], y.iloc[:train_end])
        rows.append(_coefficients(model, model_name))
        dates.append(X.index[i])

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=columns)


def feature_stability_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Per-feature stability across walk-forward refits.

    sign_consistency: fraction of refits where the coefficient is nonzero
    and agrees in sign with its own across-refit mean. A refit that zeroed
    it (Lasso dropping the feature) counts as not agreeing, and a feature
    that was never selected gets NaN rather than a vacuous 100%. A large
    mean with sign_consistency near 50% means the feature flipped direction
    constantly -- the signature of fitting noise in whichever period a
    given refit happened to see, not a real, usable effect.
    """
    mean = coefficients.mean()
    std = coefficients.std()
    majority_sign = np.sign(mean)
    agrees = np.sign(coefficients).eq(majority_sign, axis=1) & coefficients.ne(0)
    sign_consistency = agrees.mean().where(majority_sign != 0)
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

    with_summary, _ = compare_models(X, y, horizon_days=horizon_days, min_train_size=min_train_size,
                                     model_names=model_names)
    without_summary, _ = compare_models(X.drop(columns=drop_columns), y, horizon_days=horizon_days,
                                        min_train_size=min_train_size, model_names=model_names)

    cols = ["oos_r2_vs_zero", "directional_accuracy"]
    merged = with_summary.set_index("model")[cols].join(
        without_summary.set_index("model")[cols], lsuffix="_with", rsuffix="_without",
    )
    merged["oos_r2_delta"] = merged["oos_r2_vs_zero_with"] - merged["oos_r2_vs_zero_without"]
    return merged.sort_values("oos_r2_delta", ascending=False)


@dataclass
class EconomicSignificanceResult:
    model_name: str
    n_periods: int
    n_trades: int
    cost_bps: float
    strategy_total_return: float
    strategy_annualized_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    buy_hold_total_return: float
    buy_hold_annualized_return: float
    buy_hold_sharpe: float


def strategy_returns(result: WalkForwardResult, cost_bps: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """(net log return per period, units traded per period) for the sign(y_pred) strategy
    described in economic_significance."""
    position = np.sign(result.y_pred)
    position_prev = np.concatenate([[0.0], position[:-1]])
    units_traded = np.abs(position - position_prev)
    return position * result.y_true - units_traded * cost_bps / 10_000, units_traded


def economic_significance(
    result: WalkForwardResult, horizon_days: int, cost_bps: float = 5.0
) -> EconomicSignificanceResult:
    """Turn walk-forward predictions into a P&L, not just a p-value.

    Rule: go long for the horizon_days block when the model predicts a
    positive return, short when negative (sign(y_pred)). Each change in
    position pays cost_bps per unit traded: entering from flat trades one
    unit, flipping long <-> short trades two (exit the old side, enter the
    new one). A model can hit a "significant" directional p-value and still
    lose money once realistic costs eat the edge -- that's what this checks,
    benchmarked against plain buy-and-hold over the same out-of-sample
    stretch.

    Returns are log-returns, additive across the non-overlapping periods,
    so cumulative P&L is just their running sum; expm1 converts a summed
    log-return back to an arithmetic (%) return for reporting.
    """
    net_period_return, units_traded = strategy_returns(result, cost_bps)
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
    running_max = np.maximum.accumulate(np.concatenate([[0.0], cum_log]))[1:]
    max_drawdown = float(np.expm1(np.min(cum_log - running_max)))

    return EconomicSignificanceResult(
        model_name=result.model_name,
        n_periods=len(result.y_true),
        n_trades=int(np.sum(units_traded > 0)),
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
    recompute the forecast metrics within each bucket -- no refitting, just
    partitioning predictions that were already made.

    Regimes with fewer than min_n test points are dropped rather than
    reported with a misleadingly precise-looking (and tiny-sample) metric.
    """
    labels = regimes.loc[result.dates, regime_column]
    rows = []
    for label in sorted(labels.dropna().unique()):
        mask = (labels == label).to_numpy()
        if mask.sum() < min_n:
            continue
        metrics = forecast_metrics(result.y_pred[mask], result.y_true[mask], result.y_bench_mean[mask])
        rows.append({"model": result.model_name, "regime": label, **metrics})
    return pd.DataFrame(rows)


def largest_contribution(result: WalkForwardResult) -> dict:
    """How much of a model's edge over the random walk comes from its single best test point.

    OOS R^2 sums squared errors, so one extreme day can carry it: a model that
    happened to lean the right way into a crash looks skilful overall. Reports
    that day, its share of the total squared-error reduction (can exceed 100%
    when the other points net out negative), and the R^2 with it removed.
    """
    gain = result.y_true**2 - (result.y_true - result.y_pred) ** 2
    top = int(np.argmax(gain))
    keep = np.arange(len(gain)) != top
    total = float(gain.sum())
    return {
        "date": result.dates[top],
        "actual": float(result.y_true[top]),
        "predicted": float(result.y_pred[top]),
        "share_of_gain": float(gain[top] / total) if total > 0 else float("nan"),
        "oos_r2_without": forecast_metrics(result.y_pred[keep], result.y_true[keep],
                                           result.y_bench_mean[keep])["oos_r2_vs_zero"],
    }
