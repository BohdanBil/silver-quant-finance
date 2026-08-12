"""Macro/fundamental data ingestion for silver return forecasting (free sources only).

Silver's own price history lives in data.py. This module pulls the external
series a forecast would lean on: gold (cross-asset), real rates and
breakeven inflation and the dollar index (macro, via FRED), and CFTC
Commitments-of-Traders positioning.
"""

import io

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REAL_RATE_10Y = "DFII10"  # 10y TIPS yield
BREAKEVEN_INFLATION_10Y = "T10YIE"  # 10y breakeven inflation
USD_INDEX = "DTWEXBGS"  # trade-weighted USD, broad
GOLD_TICKER = "GC=F"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# CFTC Socrata API: "Disaggregated Futures Only" report, silver on COMEX.
COT_DISAGG_FUTONLY_ENDPOINT = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_SILVER_MARKET = "SILVER - COMMODITY EXCHANGE INC."
COT_PUBLICATION_LAG_DAYS = 3  # report is "as of" Tuesday, published the following Friday


def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch one FRED series with no API key, via its public CSV export.

    Uses `requests` rather than pandas.read_csv(url) directly: on this
    machine the stdlib ssl module can't find a cert store (python.org's
    macOS build needs Install Certificates.command, which wasn't run), so
    urllib-based fetches fail SSL verification. requests/yfinance don't hit
    this because they bundle certifi as their default CA source.
    """
    resp = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["value"].dropna()
    series.name = series_id
    return series


def fetch_gold_price(period: str = "5y", ticker: str = GOLD_TICKER) -> pd.Series:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    close = df["Close"].dropna()
    return close.squeeze()


def fetch_cot_silver() -> pd.DataFrame:
    """Weekly CFTC Disaggregated Futures-Only positioning for COMEX silver.

    Managed Money is the closest COT category to speculative/trend
    positioning; Producer/Merchant is commercial hedgers -- the classic
    "smart money vs. crowd" pairing COT-based signals are built on. Both
    are expressed as net position (long - short) scaled by open interest,
    since raw contract counts trend upward with OI growth over a 5-10y
    window and aren't comparable across time on their own.

    The returned index is report_date + COT_PUBLICATION_LAG_DAYS, not the
    report's own "as of" date: that date is a Tuesday, but nobody could
    actually see that snapshot until the Commission published it the
    following Friday. Indexing by the as-of date directly would leak
    information into a walk-forward backtest three days early.
    """
    resp = requests.get(
        COT_DISAGG_FUTONLY_ENDPOINT,
        params={
            "$where": f"market_and_exchange_names = '{COT_SILVER_MARKET}'",
            "$select": (
                "report_date_as_yyyy_mm_dd,open_interest_all,"
                "m_money_positions_long_all,m_money_positions_short_all,"
                "prod_merc_positions_long,prod_merc_positions_short"
            ),
            "$order": "report_date_as_yyyy_mm_dd",
            "$limit": 5000,
        },
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError("CFTC API returned no COT rows for silver")

    df = pd.DataFrame(rows)
    numeric_cols = [
        "open_interest_all", "m_money_positions_long_all", "m_money_positions_short_all",
        "prod_merc_positions_long", "prod_merc_positions_short",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    report_date = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

    oi = df["open_interest_all"].replace(0, np.nan)
    mm_net_pct_oi = (df["m_money_positions_long_all"] - df["m_money_positions_short_all"]) / oi
    comm_net_pct_oi = (df["prod_merc_positions_long"] - df["prod_merc_positions_short"]) / oi

    out = pd.DataFrame(
        {"cot_mm_net_pct_oi": mm_net_pct_oi.values, "cot_comm_net_pct_oi": comm_net_pct_oi.values},
        index=report_date + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS),
    )
    out.index.name = "date"
    return out.sort_index().dropna()


def _asof_align(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill `series` onto `target_index`: each date gets the most
    recent value AT OR BEFORE it, never a later one. Never backfilled, so a
    walk-forward backtest over the result can't accidentally see the future.
    """
    combined = series.reindex(target_index.union(series.index)).sort_index().ffill()
    return combined.reindex(target_index)


def build_feature_frame(silver_prices: pd.Series, period: str = "5y") -> pd.DataFrame:
    """Merge silver against gold + macro series + COT positioning on silver's trading calendar."""
    gold = fetch_gold_price(period=period)
    real_rate = fetch_fred_series(REAL_RATE_10Y)
    breakeven = fetch_fred_series(BREAKEVEN_INFLATION_10Y)
    usd_index = fetch_fred_series(USD_INDEX)
    cot = fetch_cot_silver()

    idx = silver_prices.index
    frame = pd.DataFrame(index=idx)
    frame["silver"] = silver_prices
    frame["gold"] = _asof_align(gold, idx)
    frame["real_rate"] = _asof_align(real_rate, idx)
    frame["breakeven_inflation"] = _asof_align(breakeven, idx)
    frame["usd_index"] = _asof_align(usd_index, idx)
    frame["cot_mm_net_pct_oi"] = _asof_align(cot["cot_mm_net_pct_oi"], idx)
    frame["cot_comm_net_pct_oi"] = _asof_align(cot["cot_comm_net_pct_oi"], idx)
    return frame.dropna()
