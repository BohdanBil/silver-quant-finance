# Monte Carlo Silver: Risk, Forecasting, and Option Pricing

A silver-price research project built around three genuinely different questions, kept in
separate modules on purpose because they answer different things:

1. **Risk simulation** ([main.py](main.py)) — given the price is roughly a random walk, how bad
   could a position get? (Monte Carlo, VaR/CVaR)
2. **Forecasting** ([forecast_main.py](forecast_main.py), [forecast_robustness.py](forecast_robustness.py)) —
   is there any actual predictive signal in gold/macro/COT data? (walk-forward ML, tested honestly)
3. **Option pricing** ([option_main.py](option_main.py)) — what is a silver option worth?
   (Black-Scholes closed form, cross-checked against risk-neutral Monte Carlo)

The headline result of the forecasting work is a **negative one**: after testing 10 models
across 4 horizons and market regimes, nothing beats a naive random-walk baseline in a way that
survives multiple-comparison correction. That's not a bug — it's the actual, load-bearing
finding, detailed in [Key findings](#key-findings) below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

XGBoost needs the OpenMP runtime on macOS, which isn't installed by default:

```bash
brew install libomp
```

No API keys required. Silver/gold prices come from Yahoo Finance (`yfinance`); real rates,
breakeven inflation, and the dollar index come from FRED's public CSV export (no key needed);
CFTC Commitments-of-Traders positioning comes from the CFTC's public Socrata API.

## Project layout

| File | Role |
|---|---|
| [data.py](data.py) | Fetch silver prices; calibrate a GBM model four ways (flat historical, EWMA vol, GARCH(1,1), block bootstrap) |
| [monte_carlo.py](monte_carlo.py) | GBM path simulation — Student-t innovations, Merton jump-diffusion, antithetic variates |
| [risk.py](risk.py) | VaR / CVaR from simulated P&L, plus a Kupiec proportion-of-failures backtest against real history |
| [main.py](main.py) | CLI tying the above together: simulate, plot a fan chart + P&L histogram, print risk metrics |
| [forecast_data.py](forecast_data.py) | Macro/gold/COT data ingestion (FRED, yfinance, CFTC), all forward-filled with no lookahead |
| [features.py](features.py) | Turns raw levels into stationary features + a leakage-free N-day forward-return target |
| [forecast_model.py](forecast_model.py) | 10 models (4 naive baselines + 6 ML), purged walk-forward validation, economic-significance backtest, feature-stability tracking, regime labeling |
| [forecast_main.py](forecast_main.py) | CLI: compare all models OOS at one horizon, check economic significance, COT ablation + stability deep-dive |
| [forecast_robustness.py](forecast_robustness.py) | CLI: sweep 1d/5d/21d/63d horizons and bull/bear/high-vol/low-vol regimes, with multiple-comparison correction |
| [option_pricing.py](option_pricing.py) | Black-Scholes price + Greeks (closed form); risk-neutral Monte Carlo price with antithetic + control-variate variance reduction |
| [option_main.py](option_main.py) | CLI: price a silver option both ways, using live calibrated inputs, and check they agree |

## Usage

**Risk simulation** — Monte Carlo price paths, VaR/CVaR, with a choice of calibration and
process model:

```bash
python main.py --calibration bootstrap --model gbm-jump --dist t --antithetic
```

**Forecasting, one horizon** — every model's out-of-sample R² vs. a random walk, economic
significance net of transaction costs, and whether COT positioning actually helps:

```bash
python forecast_main.py --period 10y --horizon-days 21
```

**Forecasting, robustness sweep** — the same question across 4 horizons and 4 market regimes at
once, with a Bonferroni-corrected significance bar so the multiple-comparisons problem can't
quietly produce a false "win":

```bash
python forecast_robustness.py --period 10y
```

**Option pricing** — Black-Scholes vs. Monte Carlo, using live spot + a GARCH vol estimate + the
FRED 3-month T-bill rate by default:

```bash
python option_main.py --option-type put --strike 60 --maturity-days 90
```

All four scripts are self-contained CLIs; run any of them with `--help` for the full flag list.

## Key findings

These are the specific, reproducible results from this project's own runs — not general claims
about markets.

### Risk model
- A plain GBM with historical drift/vol **fails its own Kupiec backtest**: at the 99% level it's
  breached ~1.8% of the time historically (should be ~1%) — it understates tail risk. Block
  bootstrap calibration (propagating parameter *uncertainty*, not just process noise) combined
  with a Merton jump-diffusion component fixes this in testing.
- Silver's worst single day in the 5-year sample (Jan 30, 2026, roughly -31%) is a real historical
  event — the worst day since March 1980 — not a data artifact. It was cross-checked against news
  coverage after initially (wrongly) suspecting a futures-roll splice.

### Forecasting
- Ridge/Lasso/ElasticNet/Random Forest/GBM/XGBoost, using gold, real rates, breakeven inflation,
  the dollar index, and CFTC Commitments-of-Traders positioning (Managed Money and Producer/
  Merchant net positioning) as features, were walk-forward validated against 21-day-forward
  silver log returns.
- **A naive "predict the historical mean return" baseline outperformed every ML model** on a 5-year
  window — and that "win" reversed to a loss on a 10-year window, which is itself informative: it
  was riding a trend that happened to continue, not skill.
- **COT positioning was ablation-tested** (with vs. without, per model): removing it improved
  every model on the 5-year window; on 10 years the result was a mixed, marginal, noise-level
  effect. Its coefficient's sign was unstable across walk-forward refits in both windows
  (~20-43% sign consistency — closer to a coin flip than a real, stable relationship).
- **The horizon/regime sweep** ([forecast_robustness.py](forecast_robustness.py)) tested 10
  models × 4 horizons (1d/5d/21d/63d) × regime slices (bull/bear via 200-day SMA, high/low
  realized vol). Out of 40 (model, horizon) combinations, exactly one — momentum at a 1-day
  horizon (p≈0.02) — was nominally significant, which is within the ~2 false positives expected
  by chance at α=0.05 across 40 tests, and it doesn't survive a Bonferroni-corrected threshold.
  No regime slice produced a real (non-artifact) significant result either.
- **Conclusion**: this project did not find an exploitable forecasting signal in gold/macro/COT
  data for silver at any horizon from 1 to 63 trading days, or within any bull/bear/vol regime
  tested. That's a real result, arrived at by actually checking rather than assuming — most of
  the engineering effort above (purged walk-forward, ablation, regime slicing, multiple-comparison
  correction) exists specifically to make sure a "signal" isn't just noise dressed up as a finding.

### Option pricing
- Black-Scholes closed-form and Greeks were checked against a standard textbook example (Hull:
  S=K=100, r=5%, σ=20%, T=1y) and match to 4 decimal places; put-call parity holds exactly.
  The Monte Carlo pricer converges to the closed-form price (verified via 200 independent
  replications, not just eyeballing one run's confidence interval), with antithetic variates +
  a control variate cutting the standard error by roughly 2-5x for the same path count.

## A note on methodology

A few corrections happened along the way and are left visible in the git history rather than
squashed away: a GBM time-step bug that produced explosive prices, a Black-Scholes/Monte-Carlo
"mismatch" that turned out to be a single unlucky random seed rather than a bug (verified via
200-replication bias test before accepting it as fine), an `sklearn` regularization-path scaling
issue that silently returned unconverged coefficients, and a first read of silver's -31% crash day
as a likely data artifact that turned out, on checking, to be a real and significant historical
event. Included here because the failure modes are as informative as the successes, and because
"looks wrong, turned out right" and "looks right, turned out wrong" both happened in this project
and are worth being honest about.
