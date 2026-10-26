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
from forecast_model import MODEL_FACTORIES, compare_models, label_regimes, regime_conditional_metrics
from plot_style import DIVERGING, INK, MUTED, SURFACE, apply_style
from significance import benjamini_hochberg, bonferroni, holm

HORIZONS = [1, 5, 21, 63]
TARGET_TEST_POINTS = 100  # cap refits for short horizons; leaves 21d/63d at their natural (larger) step
TESTS = {"cw_pvalue": "Clark-West (beats random walk)", "pt_pvalue": "Pesaran-Timmermann (direction)"}


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


def report_family(df: pd.DataFrame, id_columns: list[str], family_name: str) -> int:
    """Print raw and adjusted significance counts for each test over one family of cells.

    Returns how many (cell, test) pairs survive Holm or Benjamini-Hochberg.
    """
    survivors = 0
    for col, test_name in TESTS.items():
        p = df[col].to_numpy()
        m = int(np.sum(~np.isnan(p)))
        adjusted = {"Bonferroni": bonferroni(p), "Holm": holm(p), "BH": benjamini_hochberg(p)}
        counts = "  ".join(f"{name} {int(np.sum(a < 0.05))}" for name, a in adjusted.items())
        print(f"  {test_name}: {m} testable {family_name} -- p<0.05 raw {int(np.sum(p < 0.05))}  |  "
              f"after adjustment: {counts}")
        raw_hits = df[p < 0.05]
        if not raw_hits.empty:
            shown = raw_hits[id_columns + [col]].copy()
            shown["holm"] = adjusted["Holm"][p < 0.05]
            shown["bh"] = adjusted["BH"][p < 0.05]
            print("    " + shown.to_string(index=False, float_format="{:.4f}".format).replace("\n", "\n    "))
        survivors += int(np.sum((adjusted["Holm"] < 0.05) | (adjusted["BH"] < 0.05)))
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


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    frame = build_feature_frame(prices, period=args.period, as_of=args.as_of)
    print(f"Fetched {len(prices)} silver prices over {args.period} "
          f"({prices.index[0].date()} to {prices.index[-1].date()}), merged feature frame: {frame.shape[0]} rows")

    model_names = list(MODEL_FACTORIES.keys())

    print(f"\n=== Horizon sweep: {args.horizons} ===")
    sweep_rows = []
    results_by_horizon = {}
    for h in args.horizons:
        X, y = engineer_features(frame, horizon_days=h)
        test_step = max(h, -(-(len(X) - args.min_train_size) // TARGET_TEST_POINTS))  # ceil div
        summary, results = compare_models(
            X, y, horizon_days=h, min_train_size=args.min_train_size,
            model_names=model_names, test_step=test_step,
        )
        results_by_horizon[h] = results
        summary.insert(0, "horizon_days", h)
        sweep_rows.append(summary)
        print(f"  horizon={h:>3d}d: {summary['n_predictions'].iloc[0]} test points (test_step={test_step})")

    sweep = pd.concat(sweep_rows, ignore_index=True)

    print("\nOOS R^2 vs random walk, by model x horizon:")
    pivot_r2 = sweep.pivot(index="model", columns="horizon_days", values="oos_r2_vs_zero")
    pivot_r2 = pivot_r2.loc[model_names]
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

    print(f"\nSignificance across the {len(sweep)} (model, horizon) cells:")
    sweep_survivors = report_family(sweep, ["horizon_days", "model"], "cells")

    # --- regime breakdown, reusing the already-computed regime_horizon results (no refitting) ---
    print(f"\n=== Regime breakdown at horizon={args.regime_horizon}d (reusing the sweep's predictions) ===")
    if args.regime_horizon not in results_by_horizon:
        X, y = engineer_features(frame, horizon_days=args.regime_horizon)
        _, regime_results = compare_models(X, y, horizon_days=args.regime_horizon,
                                           min_train_size=args.min_train_size, model_names=model_names)
    else:
        regime_results = results_by_horizon[args.regime_horizon]

    regimes = label_regimes(prices)
    regime_rows = []
    for result in regime_results.values():
        for col in ("trend_regime", "vol_regime"):
            regime_rows.append(regime_conditional_metrics(result, regimes, col, min_n=args.min_regime_n))
    regime_df = pd.concat([r for r in regime_rows if not r.empty], ignore_index=True)

    print("\nPer-regime metrics (only buckets with "
          f">= {args.min_regime_n} test points shown):")
    print(regime_df.to_string(index=False, na_rep="n/a", formatters={
        "oos_r2_vs_zero": "{:+.4f}".format,
        "oos_r2_vs_mean": "{:+.4f}".format,
        "cw_pvalue": "{:.4f}".format,
        "directional_accuracy": "{:.1%}".format,
        "pt_pvalue": "{:.4f}".format,
    }))

    beats_in_regime = regime_df[regime_df["oos_r2_vs_zero"] > 0]
    print(f"\n  -> {len(beats_in_regime)}/{len(regime_df)} (model, regime) combos beat random walk within that regime")
    print(f"Significance across the {len(regime_df)} (model, regime) cells:")
    regime_survivors = report_family(regime_df, ["regime", "model"], "cells")

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
