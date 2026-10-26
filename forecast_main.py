"""CLI: silver forward-return forecasting -- simple baselines vs ML models, OOS
comparison, economic significance, and a COT-specific ablation/stability check."""

import argparse

from data import fetch_silver_prices
from features import engineer_features
from forecast_data import build_feature_frame
from forecast_model import (
    COT_COLUMNS,
    MODEL_FACTORIES,
    SIGNED_ML_MODELS,
    ablation_compare,
    compare_models,
    economic_significance,
    feature_stability_summary,
    fit_full_and_explain,
    walk_forward_coefficients,
)
from significance import holm

SUMMARY_FORMATTERS = {
    "oos_r2_vs_zero": "{:+.4f}".format,
    "oos_r2_vs_mean": "{:+.4f}".format,
    "cw_pvalue": "{:.4f}".format,
    "directional_accuracy": "{:.1%}".format,
    "pt_pvalue": "{:.4f}".format,
    "cw_holm": "{:.4f}".format,
    "pt_holm": "{:.4f}".format,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--period", default="5y", help="history window, e.g. 5y, 2y, 10y")
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--horizon-days", type=int, default=21, help="forecast horizon in trading days")
    p.add_argument("--models", nargs="+", choices=list(MODEL_FACTORIES), default=None,
                   help="subset of models to compare (default: all, baselines + ML)")
    p.add_argument("--min-train-size", type=int, default=504, help="trading days before the first prediction")
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="transaction cost in bps per unit traded, for economic significance")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = fetch_silver_prices(period=args.period, as_of=args.as_of)
    print(f"Fetched {len(prices)} silver prices over {args.period}, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    frame = build_feature_frame(prices, period=args.period, as_of=args.as_of)
    print(f"Merged feature frame: {frame.shape[0]} rows "
          f"(silver, gold, real_rate, breakeven_inflation, usd_index, COT positioning)")

    X, y = engineer_features(frame, horizon_days=args.horizon_days)
    print(f"Engineered {X.shape[1]} features, {X.shape[0]} usable rows "
          f"(target: {args.horizon_days}-day forward log return)")

    summary, results = compare_models(
        X, y, horizon_days=args.horizon_days, min_train_size=args.min_train_size, model_names=args.models,
    )
    summary["cw_holm"] = holm(summary["cw_pvalue"])
    summary["pt_holm"] = holm(summary["pt_pvalue"])
    n_predictions = int(summary["n_predictions"].iloc[0])

    print(f"\nModel comparison (walk-forward OOS, {n_predictions} non-overlapping "
          f"{args.horizon_days}-day predictions each; simple baselines + ML side by side):")
    print("  oos_r2_vs_zero / vs_mean: against a random walk / against the expanding historical mean")
    print("  cw: Clark-West p (beats the random walk?)   pt: Pesaran-Timmermann p (real directional skill?)")
    print("  *_holm: Holm-adjusted across the models in this table; n/a = nothing to test (e.g. constant sign)")
    print(summary.to_string(index=False, formatters=SUMMARY_FORMATTERS, na_rep="n/a"))

    beats_rw = summary[summary["oos_r2_vs_zero"] > 0]
    if beats_rw.empty:
        print("  -> nothing beats a pure random-walk baseline out of sample")
    else:
        print(f"  -> positive OOS R^2 vs random walk: {', '.join(beats_rw['model'])}")
    significant = summary[(summary["cw_holm"] < 0.05) | (summary["pt_holm"] < 0.05)]
    if significant.empty:
        print("  -> none is significant after Holm adjustment (Clark-West or Pesaran-Timmermann, 5% level)")
    else:
        print(f"  -> significant after Holm adjustment: {', '.join(significant['model'])}")
    if n_predictions < 50:
        print(f"  -> caveat: only {n_predictions} non-overlapping test points in {args.period}; "
              f"treat rankings as a rough read, not a settled verdict")

    best_name = summary.iloc[0]["model"]
    best_result = results[best_name]
    print(f"\nBest by OOS R^2: {best_name} -- checking whether it's actually worth trading:")

    econ = economic_significance(best_result, horizon_days=args.horizon_days, cost_bps=args.cost_bps)
    print(f"  {econ.n_trades} trades over {econ.n_periods} periods, {econ.cost_bps:g}bps per unit traded")
    print(f"  strategy:  total return {econ.strategy_total_return:+.1%}  "
          f"annualized {econ.strategy_annualized_return:+.1%}  "
          f"Sharpe {econ.strategy_sharpe:.2f}  max drawdown {econ.strategy_max_drawdown:.1%}")
    print(f"  buy&hold:  total return {econ.buy_hold_total_return:+.1%}  "
          f"annualized {econ.buy_hold_annualized_return:+.1%}  "
          f"Sharpe {econ.buy_hold_sharpe:.2f}")
    verdict = "beats" if econ.strategy_total_return > econ.buy_hold_total_return else "loses to"
    print(f"    -> {best_name}-driven strategy {verdict} plain buy-and-hold over this window")

    importances = fit_full_and_explain(X, y, horizon_days=args.horizon_days, model_name=best_name)
    if importances is None:
        print(f"\n{best_name} has no per-feature weights to show (it ignores features entirely).")
    else:
        print(f"\nFull-sample fit of {best_name} (interpretability only -- NOT an out-of-sample result):")
        for name, val in importances.items():
            print(f"  {name:<30} {val:+.4f}")

    # --- COT deep-dive: does it actually predict, or does it just show up in one fit? ---
    linear_summary = summary[summary["model"].isin(SIGNED_ML_MODELS)]
    if linear_summary.empty:
        return
    best_linear = linear_summary.iloc[0]["model"]
    print("\n--- COT deep-dive ---")
    print(f"Coefficient stability uses {best_linear}, the best ML model with signed coefficients "
          f"(tree importances are always >= 0, so their 'sign' can't flip and says nothing).")

    coefficients = walk_forward_coefficients(X, y, horizon_days=args.horizon_days, model_name=best_linear,
                                             min_train_size=args.min_train_size)
    stability = feature_stability_summary(coefficients)
    print(f"\nFeature stability across the {len(coefficients)} walk-forward refits "
          f"(sign_consistency near 100% = stable direction; near 50% or below = flips sign or drops out):")
    print(stability.to_string(na_rep="never used", formatters={
        "mean": "{:+.5f}".format, "std": "{:.5f}".format, "sign_consistency": "{:.0%}".format,
    }))
    cot_rows = stability.loc[stability.index.isin(COT_COLUMNS), "sign_consistency"]
    if cot_rows.notna().any():
        avg_cot_consistency = cot_rows.fillna(0).mean()
        print(f"  -> COT features' average sign consistency: {avg_cot_consistency:.0%} "
              f"({'looks stable' if avg_cot_consistency >= 0.8 else 'not a stable relationship'})")

    ablation = ablation_compare(X, y, horizon_days=args.horizon_days, drop_columns=COT_COLUMNS,
                                min_train_size=args.min_train_size)
    print("\nAblation: OOS R^2 with COT vs with COT dropped, per ML model "
          "(negative delta = COT made that model worse):")
    print(ablation.to_string(formatters={
        c: ("{:.1%}" if c.startswith("directional_accuracy") else "{:+.4f}").format for c in ablation.columns
    }))

    n_helped = int((ablation["oos_r2_delta"] > 0).sum())
    print(f"\nVerdict: COT helped {n_helped}/{len(ablation)} models' OOS R^2.")
    if n_helped == 0:
        print("  -> Does COT really predict? No -- dropping it improved every single model here.")
    elif n_helped == len(ablation):
        print("  -> Does COT really predict? Looks that way -- it helped every model tested; "
              "worth checking sign_consistency above before trusting it, though.")
    else:
        print("  -> Does COT really predict? Mixed -- helps some models, hurts others; "
              "not a clean, model-agnostic signal.")


if __name__ == "__main__":
    main()
