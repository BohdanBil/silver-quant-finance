"""CLI: is the weak forecasting signal smeared across horizon, or is there really none?

Two checks the single-horizon forecast_main.py can't answer on its own:
1. Sweep horizons (1d/5d/21d/63d) -- a signal invisible at 21d could show up
   at 1d (short-horizon microstructure) or 63d (slower macro-driven
   mean reversion). Flat-near-zero at every horizon is much stronger
   evidence of "no signal" than a single horizon ever could be.
2. Slice the 21d walk-forward predictions by market regime (bull/bear via
   200d SMA, high/low realized vol) -- an average OOS R^2 near zero can
   still hide real skill that's confined to, say, high-vol regimes and
   cancelled out by noise elsewhere.

Every cell is tested twice (Clark-West on forecast error, Pesaran-Timmermann
on direction), and with this many cells some will clear p<0.05 by chance
alone, so each family of tests is also reported after Bonferroni, Holm
(family-wise error) and Benjamini-Hochberg (false discovery rate) adjustment.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import fetch_silver_prices
from features import engineer_features
from forecast_data import build_feature_frame
from forecast_model import (
    MODEL_FACTORIES,
    WalkForwardResult,
    compare_models,
    label_regimes,
    largest_contribution,
    regime_conditional_metrics,
)
from plot_style import DIVERGING, INK, MUTED, SURFACE, apply_style
from significance import benjamini_hochberg, bonferroni, holm

HORIZONS = [1, 5, 21, 63]
TARGET_TEST_POINTS = 100  # cap refits for short horizons; leaves 21d/63d at their natural (larger) step
TESTS = {"cw_pvalue": "Clark-West (beats random walk)", "pt_pvalue": "Pesaran-Timmermann (direction)"}
ADJUSTMENTS = {"Bonferroni": bonferroni, "Holm": holm, "BH": benjamini_hochberg}


def horizon_sweep(frame: pd.DataFrame, horizons: list[int], min_train_size: int = 504,
                  model_names: list[str] | None = None,
                  ) -> tuple[pd.DataFrame, dict[int, dict[str, WalkForwardResult]]]:
    """compare_models at every horizon; returns (one row per model x horizon, results by horizon)."""
    model_names = model_names or list(MODEL_FACTORIES.keys())
    rows, results_by_horizon = [], {}
    for h in horizons:
        X, y = engineer_features(frame, horizon_days=h)
        test_step = max(h, -(-(len(X) - min_train_size) // TARGET_TEST_POINTS))  # ceil div
        summary, results = compare_models(X, y, horizon_days=h, min_train_size=min_train_size,
                                          model_names=model_names, test_step=test_step)
        results_by_horizon[h] = results
        summary.insert(0, "horizon_days", h)
        summary["test_step"] = test_step
        rows.append(summary)
    return pd.concat(rows, ignore_index=True), results_by_horizon


def regime_breakdown(prices: pd.Series, results: dict[str, WalkForwardResult], min_n: int = 8) -> pd.DataFrame:
    """Per-model metrics within each trend and vol regime, reusing already-made predictions."""
    regimes = label_regimes(prices)
    rows = [regime_conditional_metrics(result, regimes, col, min_n=min_n)
            for result in results.values() for col in ("trend_regime", "vol_regime")]
    return pd.concat([r for r in rows if not r.empty], ignore_index=True)


def adjusted_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """Add <test>_<adjustment> columns, each test adjusted across all the rows of `df` as one family."""
    out = df.copy()
    for col in TESTS:
        for name, adjust in ADJUSTMENTS.items():
            out[f"{col}_{name.lower()}"] = adjust(df[col].to_numpy())
    return out


def significance_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """For each test: testable cells, raw p<0.05, and p<0.05 after each adjustment (df from adjusted_pvalues)."""
    counts = {}
    for col in TESTS:
        p = df[col].to_numpy()
        counts[col] = {"testable": int(np.sum(~np.isnan(p))), "raw": int(np.sum(p < 0.05))}
        for name in ADJUSTMENTS:
            counts[col][name] = int(np.sum(df[f"{col}_{name.lower()}"].to_numpy() < 0.05))
    return counts


def report_family(df: pd.DataFrame, id_columns: list[str]) -> int:
    """Print raw and adjusted significance counts for one family of cells (df from adjusted_pvalues).

    Returns how many (cell, test) pairs survive Holm or Benjamini-Hochberg.
    """
    counts = significance_counts(df)
    survivors = 0
    for col, test_name in TESTS.items():
        c = counts[col]
        adjusted = "  ".join(f"{name} {c[name]}" for name in ADJUSTMENTS)
        print(f"  {test_name}: {c['testable']} testable cells -- p<0.05 raw {c['raw']}  |  "
              f"after adjustment: {adjusted}")
        raw_hits = df[df[col] < 0.05]
        if not raw_hits.empty:
            shown = raw_hits[id_columns + [col, f"{col}_holm", f"{col}_bh"]]
            print("    " + shown.to_string(index=False, float_format="{:.4f}".format).replace("\n", "\n    "))
        survivors += int(np.sum((df[f"{col}_holm"] < 0.05) | (df[f"{col}_bh"] < 0.05)))
    return survivors


def plot_heatmap(pivot: pd.DataFrame, path: str) -> None:
    apply_style()
    data = pivot.to_numpy()
    limit = float(np.nanmax(np.abs(data))) or 1.0
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(pivot) + 1.4))
    ax.imshow(data, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(pivot.shape[1]), [f"{h}d" for h in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            strong = abs(value) > 0.55 * limit
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if strong else INK)
    ax.set_xlabel("forecast horizon")
    ax.set_title("Out-of-sample R² vs a random walk (blue = better, red = worse)", loc="left")
    fig.text(0.01, 0.01, "Walk-forward with a purge gap; 0 = no better than predicting a zero return.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    print(f"\nSaved heatmap to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="10y", help="history window -- longer is better here for regime variety")
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--regime-horizon", type=int, default=21, help="which horizon's predictions to slice by regime")
    p.add_argument("--min-train-size", type=int, default=504)
    p.add_argument("--min-regime-n", type=int, default=8, help="drop regime buckets with fewer test points than this")
    p.add_argument("--plot", default=None, help="save a heatmap of OOS R^2 by model x horizon to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    frame = build_feature_frame(prices, period=args.period, as_of=args.as_of)
    print(f"Fetched {len(prices)} silver prices over {args.period} "
          f"({prices.index[0].date()} to {prices.index[-1].date()}), merged feature frame: {frame.shape[0]} rows")

    model_names = list(MODEL_FACTORIES.keys())
    print(f"\n=== Horizon sweep: {args.horizons} ===")
    sweep, results_by_horizon = horizon_sweep(frame, args.horizons, args.min_train_size, model_names)
    sweep = adjusted_pvalues(sweep)
    for h in args.horizons:
        row = sweep[sweep["horizon_days"] == h].iloc[0]
        print(f"  horizon={h:>3d}d: {row['n_predictions']} test points (test_step={row['test_step']})")

    print("\nOOS R^2 vs random walk, by model x horizon:")
    pivot_r2 = sweep.pivot(index="model", columns="horizon_days", values="oos_r2_vs_zero").loc[model_names]
    print(pivot_r2.to_string(float_format="{:+.4f}".format))

    print("\nOOS R^2 vs historical mean, by model x horizon:")
    print(sweep.pivot(index="model", columns="horizon_days", values="oos_r2_vs_mean").loc[model_names]
          .to_string(float_format="{:+.4f}".format))

    print("\nDirectional accuracy, by model x horizon:")
    print(sweep.pivot(index="model", columns="horizon_days", values="directional_accuracy").loc[model_names]
          .to_string(float_format="{:.1%}".format))

    print("\nBest OOS R^2 per horizon (across all models):")
    for h in args.horizons:
        sub = sweep[sweep["horizon_days"] == h]
        best = sub.loc[sub["oos_r2_vs_zero"].idxmax()]
        print(f"  {h:>3d}d: {best['model']:<16} R2={best['oos_r2_vs_zero']:+.4f}  "
              f"dir_acc={best['directional_accuracy']:.1%}  CW p={best['cw_pvalue']:.3f}  "
              f"PT p={best['pt_pvalue']:.3f}")

    print("\nHow much of each positive OOS R^2 comes from a single test point:")
    for _, row in sweep[sweep["oos_r2_vs_zero"] > 0.005].iterrows():
        top = largest_contribution(results_by_horizon[row["horizon_days"]][row["model"]])
        print(f"  {row['horizon_days']:>3d}d {row['model']:<16} R2={row['oos_r2_vs_zero']:+.4f}: "
              f"{top['share_of_gain']:.0%} from {top['date'].date()} (actual {np.expm1(top['actual']):+.2%}); "
              f"R2 without it {top['oos_r2_without']:+.4f}")

    print(f"\nSignificance across the {len(sweep)} (model, horizon) cells:")
    sweep_survivors = report_family(sweep, ["horizon_days", "model"])

    # --- regime breakdown, reusing the already-computed regime_horizon results (no refitting) ---
    print(f"\n=== Regime breakdown at horizon={args.regime_horizon}d (reusing the sweep's predictions) ===")
    if args.regime_horizon in results_by_horizon:
        regime_results = results_by_horizon[args.regime_horizon]
    else:
        X, y = engineer_features(frame, horizon_days=args.regime_horizon)
        _, regime_results = compare_models(X, y, horizon_days=args.regime_horizon,
                                           min_train_size=args.min_train_size, model_names=model_names)
    regime_df = adjusted_pvalues(regime_breakdown(prices, regime_results, min_n=args.min_regime_n))

    print("\nPer-regime metrics (only buckets with "
          f">= {args.min_regime_n} test points shown):")
    print(regime_df[["model", "regime", "n_predictions", "oos_r2_vs_zero", "oos_r2_vs_mean", "cw_pvalue",
                     "directional_accuracy", "pt_pvalue"]].to_string(index=False, na_rep="n/a", formatters={
        "oos_r2_vs_zero": "{:+.4f}".format,
        "oos_r2_vs_mean": "{:+.4f}".format,
        "cw_pvalue": "{:.4f}".format,
        "directional_accuracy": "{:.1%}".format,
        "pt_pvalue": "{:.4f}".format,
    }))

    beats_in_regime = regime_df[regime_df["oos_r2_vs_zero"] > 0]
    print(f"\n  -> {len(beats_in_regime)}/{len(regime_df)} (model, regime) combos beat random walk within that regime")
    print(f"Significance across the {len(regime_df)} (model, regime) cells:")
    regime_survivors = report_family(regime_df, ["regime", "model"])

    print("\n=== Verdict ===")
    if sweep_survivors == 0 and regime_survivors == 0:
        print("No signal survives a multiple-comparison adjustment: some (model, horizon) and (model, regime) "
              "cells have positive OOS R^2 or a raw p<0.05, which is expected by chance alone across this many "
              "tests, but none remain significant after Holm or Benjamini-Hochberg adjustment. That's consistent "
              "with no real, exploitable signal in these features for silver at any horizon or regime tested.")
    else:
        print("Some cells survive adjustment -- see the tables above. Treat them with caution: the cells are "
              "correlated (same data, overlapping models), and a survivor still needs an economic check "
              "(forecast_main.py's cost-adjusted backtest) before it means anything tradable.")

    if args.plot:
        plot_heatmap(pivot_r2, args.plot)


if __name__ == "__main__":
    main()
