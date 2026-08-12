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
"""

import argparse

import numpy as np
import pandas as pd

from data import fetch_silver_prices
from features import engineer_features
from forecast_data import build_feature_frame
from forecast_model import (
    BASELINE_MODELS,
    MODEL_FACTORIES,
    compare_models,
    label_regimes,
    regime_conditional_metrics,
)

HORIZONS = [1, 5, 21, 63]
TARGET_TEST_POINTS = 100  # cap refits for short horizons; leaves 21d/63d at their natural (larger) step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--period", default="10y", help="history window -- longer is better here for regime variety")
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--regime-horizon", type=int, default=21, help="which horizon's predictions to slice by regime")
    p.add_argument("--min-train-size", type=int, default=504)
    p.add_argument("--min-regime-n", type=int, default=8, help="drop regime buckets with fewer test points than this")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period)
    frame = build_feature_frame(prices, period=args.period)
    print(f"Fetched {len(prices)} silver prices over {args.period}, merged feature frame: {frame.shape[0]} rows")

    model_names = list(MODEL_FACTORIES.keys())

    print(f"\n=== Horizon sweep: {args.horizons} ===")
    sweep_rows = []
    results_by_horizon = {}
    for h in args.horizons:
        X, y = engineer_features(frame, horizon_days=h)
        n_usable = len(X)
        test_step = max(h, -(-(n_usable - args.min_train_size) // TARGET_TEST_POINTS))  # ceil div
        summary, results = compare_models(
            X, y, horizon_days=h, min_train_size=args.min_train_size,
            model_names=model_names, test_step=test_step,
        )
        results_by_horizon[h] = results
        summary.insert(0, "horizon_days", h)
        sweep_rows.append(summary)
        print(f"  horizon={h:>3d}d: {summary['n_predictions'].iloc[0]} test points (test_step={test_step})")

    sweep = pd.concat(sweep_rows, ignore_index=True)

    print("\nOOS R^2 vs naive, by model x horizon:")
    pivot_r2 = sweep.pivot(index="model", columns="horizon_days", values="oos_r2_vs_naive")
    print(pivot_r2.to_string(float_format="{:+.4f}".format))

    print("\nDirectional accuracy, by model x horizon:")
    pivot_acc = sweep.pivot(index="model", columns="horizon_days", values="directional_accuracy")
    print(pivot_acc.to_string(float_format="{:.1%}".format))

    print("\nBest OOS R^2 per horizon (across all models):")
    for h in args.horizons:
        sub = sweep[sweep["horizon_days"] == h]
        best = sub.loc[sub["oos_r2_vs_naive"].idxmax()]
        sig = " (p<0.05)" if best["p_value"] < 0.05 else ""
        print(f"  {h:>3d}d: {best['model']:<16} R2={best['oos_r2_vs_naive']:+.4f}  "
              f"dir_acc={best['directional_accuracy']:.1%}  p={best['p_value']:.3f}{sig}")

    # A low p-value only means "significantly better OR worse than 50/50".
    # zero_return can never match a sign (it always predicts exactly 0), so
    # its directional_accuracy is always 0% -- reliably, "significantly"
    # wrong, not a signal. Require accuracy > 50% too so that artifact
    # doesn't get counted as a win.
    bonferroni_alpha = 0.05 / len(sweep)
    any_positive = sweep["oos_r2_vs_naive"] > 0
    real_significant = (sweep["p_value"] < 0.05) & (sweep["directional_accuracy"] > 0.5)
    survives_bonferroni = (sweep["p_value"] < bonferroni_alpha) & (sweep["directional_accuracy"] > 0.5)
    print(f"\n  -> {any_positive.sum()}/{len(sweep)} (model, horizon) combos beat random walk on OOS R^2")
    print(f"  -> {real_significant.sum()}/{len(sweep)} beat 50/50 with p<0.05 "
          f"(excluding zero_return's always-0%-accuracy artifact)")
    if real_significant.any():
        print(sweep[real_significant][["horizon_days", "model", "oos_r2_vs_naive",
                                        "directional_accuracy", "p_value"]].to_string(index=False))
    print(f"  -> with {len(sweep)} combinations tested, a Bonferroni-corrected threshold is "
          f"p<{bonferroni_alpha:.5f}: {survives_bonferroni.sum()}/{len(sweep)} survive that")

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
    for name, result in regime_results.items():
        for col in ("trend_regime", "vol_regime"):
            regime_rows.append(regime_conditional_metrics(result, regimes, col, min_n=args.min_regime_n))
    regime_df = pd.concat([r for r in regime_rows if not r.empty], ignore_index=True)

    print("\nOOS R^2 / directional accuracy by regime (only buckets with "
          f">= {args.min_regime_n} test points shown):")
    print(regime_df.to_string(index=False, formatters={
        "oos_r2_vs_naive": "{:+.4f}".format,
        "directional_accuracy": "{:.1%}".format,
        "directional_pvalue": "{:.4f}".format,
    }))

    beats_in_regime = regime_df[regime_df["oos_r2_vs_naive"] > 0]
    # Same fix as the horizon sweep: exclude the always-0%-accuracy artifact.
    sig_in_regime = regime_df[(regime_df["directional_pvalue"] < 0.05) & (regime_df["directional_accuracy"] > 0.5)]
    print(f"\n  -> {len(beats_in_regime)}/{len(regime_df)} (model, regime) combos beat random walk within that regime")
    if not sig_in_regime.empty:
        print("  -> statistically significant (p<0.05, and actually >50% accuracy) in-regime results:")
        print(sig_in_regime.to_string(index=False, formatters={
            "oos_r2_vs_naive": "{:+.4f}".format, "directional_accuracy": "{:.1%}".format,
        }))
    else:
        print("  -> none statistically significant at the 5% level in any single regime "
              "(excluding zero_return's always-0%-accuracy artifact)")

    print("\n=== Verdict ===")
    if not real_significant.any() and sig_in_regime.empty:
        print("No statistically real signal anywhere: some (model, horizon) and (model, regime) cells have "
              "positive OOS R^2 (expected by chance alone with this many comparisons), but none reach "
              "p<0.05 with genuine >50% directional accuracy, and none survive a Bonferroni-corrected "
              "threshold either. The weak signal seen at 21d unconditionally isn't hiding at another "
              "horizon or inside a specific regime -- it's consistent with there being no real, "
              "exploitable signal in these features for silver, not a horizon or regime that happened "
              "to be tested wrong.")
    else:
        print("Not uniformly flat -- see the rows printed above under 'beat 50/50 with p<0.05' and "
              "'statistically significant ... in-regime results'. Treat these with real caution: with "
              f"{len(sweep)} (model, horizon) combinations alone, ~{len(sweep) * 0.05:.0f} would clear "
              "p<0.05 by chance even with zero true signal, and only entries surviving the "
              "Bonferroni-corrected threshold printed above are close to safe from that.")


if __name__ == "__main__":
    main()
