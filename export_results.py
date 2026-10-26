"""Run every analysis once and write the results the website in site/ displays.

    python export_results.py --as-of 2026-10-23
    open site/index.html          # or: python -m http.server -d site 8000

Writes site/data/results.json, plus site/data/results.js (the same data as a
script, so index.html also works opened straight from disk). A full run takes
around ten minutes, most of it the forecasting walk-forwards; --only re-runs
some sections and keeps the rest from the previous export.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import TRADING_DAYS_PER_YEAR, fetch_silver_prices, log_returns_of, resolve_as_of
from features import engineer_features
from forecast_data import build_feature_frame
from forecast_main import ForecastReport, run_forecast_report
from forecast_model import BASELINE_MODELS, MODEL_FACTORIES, largest_contribution, strategy_returns
from forecast_robustness import HORIZONS, adjusted_pvalues, horizon_sweep, regime_breakdown, significance_counts
from option_main import default_inputs, price_silver_option
from option_pricing import monte_carlo_option_price, option_price
from risk import CoverageTest, var_cvar
from risk_model import all_specs, backtest_spec, backtest_variants, calibrate, simulate_paths

SITE_DATA = Path(__file__).resolve().parent / "site" / "data"
SECTIONS = ("prices", "risk", "risk_backtest", "forecast", "robustness", "options")
CONFIDENCE = [0.95, 0.99]


def clean(obj):
    """Make results JSON-safe: numpy scalars -> Python, NaN/inf -> null, timestamps -> dates."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.ndarray, pd.Index, pd.Series)):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(obj) else round(float(obj), 8)
    return obj


def records(df: pd.DataFrame) -> list[dict]:
    return df.reset_index(drop=df.index.name is None).to_dict(orient="records")


def coverage(results: dict[float, CoverageTest]) -> dict:
    return {f"{c:.2f}": {
        "n_obs": bt.n_obs, "n_exceptions": bt.n_exceptions, "rate": bt.exception_rate,
        "expected": bt.expected_rate, "kupiec_p": bt.kupiec_pvalue, "cluster_p": bt.independence_pvalue,
        "rejected": bt.rejected, "verdict": bt.verdict,
    } for c, bt in results.items()}


# --- sections -----------------------------------------------------------------

def export_prices(as_of: str) -> dict:
    prices = fetch_silver_prices("10y", as_of=as_of)
    return {"dates": prices.index, "close": prices.to_numpy()}


def export_risk(as_of: str, period: str = "5y", horizon: int = 21, n_sims: int = 20_000,
                position_oz: float = 1_000.0, seed: int = 42, window: int = 504, backtest_sims: int = 5_000) -> dict:
    prices = fetch_silver_prices(period, as_of=as_of)
    r = log_returns_of(prices)
    variants = []
    for spec in all_specs():
        started = time.perf_counter()
        model = calibrate(prices, spec, horizon_days=horizon, n_sims=n_sims, seed=seed)
        paths = simulate_paths(model, horizon, n_sims, seed=seed)
        s0 = model.params.s0
        pnl = (paths[:, -1] - s0) * position_oz
        counts, edges = np.histogram(pnl, bins=60)
        mu, sigma = np.asarray(model.params.mu), np.asarray(model.params.sigma)
        jumps = model.jumps
        variants.append({
            "id": f"{spec.calibration}|{spec.model}|{spec.dist}",
            "label": spec.label, "calibration": spec.calibration, "process": spec.model, "dist": spec.dist,
            "sigma": float(sigma.mean()), "sigma_flat": model.point_sigma,
            "mu": float(mu.mean()),
            "mu_range": [float(np.quantile(mu, 0.05)), float(np.quantile(mu, 0.95))] if mu.ndim else None,
            "jumps": None if jumps is None else {"per_year": jumps.intensity, "mean": jumps.mean, "std": jumps.std,
                                                 "n": jumps.n_detected},
            "notes": model.notes,
            "fan": {f"p{q:02d}": np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95)},
            "terminal": {"mean": paths[:, -1].mean(), "p05": np.percentile(paths[:, -1], 5),
                         "p95": np.percentile(paths[:, -1], 95)},
            "hist": {"edges": edges, "counts": counts},
            "metrics": [{"confidence": c, "var": m.var, "cvar": m.cvar, "var_pct": m.var / (s0 * position_oz),
                         "cvar_pct": m.cvar / (s0 * position_oz)}
                        for c in CONFIDENCE for m in [var_cvar(pnl, c)]],
            "backtest": {
                "1": coverage(backtest_spec(prices, spec, 1, CONFIDENCE, window, backtest_sims, seed=seed)),
                str(horizon): coverage(backtest_spec(prices, spec, horizon, CONFIDENCE, window, backtest_sims,
                                                     seed=seed)),
            },
        })
        print(f"  risk: {spec.label} ({time.perf_counter() - started:.1f}s)")

    worst = r.idxmin()
    return {
        "settings": {"period": period, "horizon_days": horizon, "n_sims": n_sims, "position_oz": position_oz,
                     "seed": seed, "backtest_window": window, "backtest_sims": backtest_sims},
        "start": prices.index[0], "end": prices.index[-1], "n_prices": len(prices),
        "s0": float(prices.iloc[-1]),
        "worst_day": {"date": worst, "return": float(np.expm1(r.min()))},
        "variants": variants,
    }


def export_risk_backtest(as_of: str, period: str = "10y", window: int = 504, n_sims: int = 5_000) -> dict:
    prices = fetch_silver_prices(period, as_of=as_of)
    table = backtest_variants(prices, all_specs(), 1, CONFIDENCE, window=window, n_sims=n_sims,
                              on_result=lambda spec, _: print(f"  backtest: {spec.label}"))
    n_tests = len(range(window, len(prices) - 1))
    return {"period": period, "window": window, "horizon_days": 1, "n_tests": n_tests,
            "start": prices.index[window], "end": prices.index[-1], "rows": records(table)}


def forecast_payload(report: ForecastReport, period: str, start, end) -> dict:
    summary = report.summary.copy()
    summary["kind"] = np.where(summary["model"].isin(BASELINE_MODELS), "baseline", "ml")
    best = report.results[report.best_name]
    net, _ = strategy_returns(best, report.cost_bps)
    econ = report.econ
    stability = None
    if report.stability is not None:
        stability = records(report.stability.rename_axis("feature"))
    return {
        "period": period, "start": start, "end": end, "horizon_days": report.horizon_days,
        "n_predictions": int(summary["n_predictions"].iloc[0]), "cost_bps": report.cost_bps,
        "summary": records(summary),
        "best_model": report.best_name,
        "economics": vars(econ),
        "equity": {"dates": best.dates, "strategy": np.expm1(np.cumsum(net)),
                   "buy_hold": np.expm1(np.cumsum(best.y_true))},
        "stability_model": report.stability_model, "n_refits": report.n_refits, "stability": stability,
        "ablation": records(report.ablation.rename_axis("model")),
    }


def export_forecast(as_of: str, horizon: int = 21) -> dict:
    out = {}
    for period in ("5y", "10y"):
        started = time.perf_counter()
        prices = fetch_silver_prices(period, as_of=as_of)
        frame = build_feature_frame(prices, period=period, as_of=as_of)
        X, y = engineer_features(frame, horizon_days=horizon)
        report = run_forecast_report(X, y, horizon_days=horizon)
        out[period] = forecast_payload(report, period, X.index[504], X.index[-1])
        print(f"  forecast {period}: {time.perf_counter() - started:.0f}s")
    return out


def export_robustness(as_of: str, period: str = "10y", regime_horizon: int = 21) -> dict:
    prices = fetch_silver_prices(period, as_of=as_of)
    frame = build_feature_frame(prices, period=period, as_of=as_of)
    sweep, results_by_horizon = horizon_sweep(frame, HORIZONS)
    sweep = adjusted_pvalues(sweep)
    regimes = adjusted_pvalues(regime_breakdown(prices, results_by_horizon[regime_horizon]))
    contributions = []
    for _, row in sweep[sweep["oos_r2_vs_zero"] > 0.005].iterrows():
        top = largest_contribution(results_by_horizon[row["horizon_days"]][row["model"]])
        contributions.append({"model": row["model"], "horizon_days": row["horizon_days"],
                              "oos_r2_vs_zero": row["oos_r2_vs_zero"], "date": top["date"],
                              "actual_return": np.expm1(top["actual"]), "share_of_gain": top["share_of_gain"],
                              "oos_r2_without": top["oos_r2_without"]})
    return {
        "period": period, "horizons": HORIZONS, "regime_horizon": regime_horizon,
        "models": list(MODEL_FACTORIES), "cells": records(sweep), "regimes": records(regimes),
        "counts": {"horizon": significance_counts(sweep), "regime": significance_counts(regimes)},
        "contributions": contributions,
    }


def export_options(as_of: str, maturity_days: int = 63, n_sims: int = 200_000, seed: int = 42,
                   bias_runs: int = 400) -> dict:
    prices = fetch_silver_prices("5y", as_of=as_of)
    f0 = float(prices.iloc[-1])
    sigma, rate = default_inputs(prices, maturity_days, as_of)
    report = price_silver_option(f0, f0, rate, sigma, maturity_days, "call", n_sims=n_sims, seed=seed)
    T, g = report.T, report.greeks

    strikes = np.linspace(0.7 * f0, 1.3 * f0, 41)
    curve = {
        "strikes": strikes,
        "black76_call": [option_price(f0, k, rate, sigma, T, "call", "futures") for k in strikes],
        "black76_put": [option_price(f0, k, rate, sigma, T, "put", "futures") for k in strikes],
        "bs_spot_call": [option_price(f0, k, rate, sigma, T, "call", "spot") for k in strikes],
    }

    convergence = []
    for n in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000):
        for label, anti, cv in (("plain", False, False), ("both", True, True)):
            res = monte_carlo_option_price(f0, f0, rate, sigma, T, n_sims=n, underlying="futures",
                                           antithetic=anti, control_variate=cv, seed=seed)
            convergence.append({"n_paths": n, "setting": label, "price": res.price,
                                "low": res.ci_95[0], "high": res.ci_95[1], "std_error": res.std_error})

    runs = [monte_carlo_option_price(f0, f0, rate, sigma, T, n_sims=n_sims, underlying="futures", seed=s)
            for s in range(bias_runs)]
    errors = np.array([run.price for run in runs]) - report.price
    ses = np.array([run.std_error for run in runs])

    plain_se = report.variance_reduction["plain"].std_error
    return {
        "inputs": {"futures": f0, "strike": f0, "rate": rate, "sigma": sigma, "maturity_days": maturity_days,
                   "T": T, "option_type": "call", "n_sims": n_sims, "seed": seed},
        "black76": report.price, "bs_as_spot": report.bs_as_spot,
        "greeks": {"delta": g.delta, "gamma": g.gamma, "vega_per_vol_pt": g.vega / 100,
                   "theta_per_trading_day": g.theta / TRADING_DAYS_PER_YEAR, "rho_per_pct": g.rho / 100},
        "mc": {"price": report.mc.price, "low": report.mc.ci_95[0], "high": report.mc.ci_95[1],
               "std_error": report.mc.std_error, "n_paths": report.mc.n_paths},
        "variance_reduction": [{"setting": label, "std_error": res.std_error,
                                "paths_saved": (plain_se / res.std_error) ** 2}
                               for label, res in report.variance_reduction.items()],
        "curve": curve,
        "convergence": convergence,
        "bias_check": {"runs": bias_runs, "n_sims": n_sims, "mean_error": errors.mean(),
                       "z": errors.mean() / (errors.std(ddof=1) / np.sqrt(bias_runs)),
                       "empirical_sd": errors.std(ddof=1), "mean_reported_se": ses.mean(),
                       "outside_ci": float(np.mean(np.abs(errors) > 1.96 * ses))},
    }


EXPORTERS = {
    "prices": export_prices,
    "risk": export_risk,
    "risk_backtest": export_risk_backtest,
    "forecast": export_forecast,
    "robustness": export_robustness,
    "options": export_options,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", default=None, help="last date of data to use (YYYY-MM-DD); default: today")
    p.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                   help="re-run only these sections, keeping the others from the existing export")
    args = p.parse_args()
    as_of = resolve_as_of(args.as_of).strftime("%Y-%m-%d")

    json_path = SITE_DATA / "results.json"
    results = json.loads(json_path.read_text()) if json_path.exists() and set(args.only) != set(SECTIONS) else {}
    if results and results.get("meta", {}).get("as_of") != as_of:
        raise SystemExit(f"existing export is as of {results['meta']['as_of']}, not {as_of}; re-run every section")

    for section in args.only:
        started = time.perf_counter()
        print(f"[{section}]")
        results[section] = clean(EXPORTERS[section](as_of))
        print(f"[{section}] done in {time.perf_counter() - started:.0f}s")

    results["meta"] = {"as_of": as_of, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "silver_ticker": "SI=F"}
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, separators=(",", ":"), allow_nan=False)
    json_path.write_text(payload)
    (SITE_DATA / "results.js").write_text(f"window.RESULTS = {payload};\n")
    print(f"Wrote {json_path} ({len(payload) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
