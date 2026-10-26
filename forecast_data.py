"""Macro/fundamental data ingestion for silver return forecasting (free sources only).

Silver's own price history lives in data.py. This module pulls the external
series a forecast would lean on: gold (cross-asset), real rates and
breakeven inflation and the dollar index (macro, via FRED), and CFTC
Commitments-of-Traders positioning.
"""

import io
from typing import Callable

import numpy as np
import pandas as pd
import requests

from data import cached, fetch_yf_close, resolve_as_of

REAL_RATE_10Y = "DFII10"  # 10y TIPS yield
BREAKEVEN_INFLATION_10Y = "T10YIE"  # 10y breakeven inflation
USD_INDEX = "DTWEXBGS"  # trade-weighted USD, broad
GOLD_TICKER = "GC=F"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# The first date a value observed on date d could have been used. Feature rows
# are dated at silver's close, so anything published after that close only
# counts from the next day on.
PUBLICATION_DATE: dict[str, Callable[[pd.DatetimeIndex], pd.DatetimeIndex]] = {
    # Daily H.15 / Treasury yields: out the following business day.
    REAL_RATE_10Y: lambda d: d + pd.offsets.BDay(1),
    BREAKEVEN_INFLATION_10Y: lambda d: d + pd.offsets.BDay(1),
    # H.10 dollar index: a weekly release on Monday afternoon covering the
    # previous week, so Monday-Friday values become usable the Tuesday after.
    USD_INDEX: lambda d: d + pd.offsets.Week(weekday=0) + pd.Timedelta(days=1),
}

# CFTC Socrata API: "Disaggregated Futures Only" report, silver on COMEX.
COT_DISAGG_FUTONLY_ENDPOINT = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_SILVER_MARKET = "SILVER - COMMODITY EXCHANGE INC."
# Report is "as of" Tuesday and released Friday 3:30pm ET, after the COMEX
# close, so it's first usable at the next session (Tuesday + 4 days lands on
# Saturday, which as-of alignment carries into Monday).
COT_PUBLICATION_LAG_DAYS = 4


def fetch_fred_series(series_id: str, as_of: str | None = None) -> pd.Series:
    """Fetch one FRED series (observations up to as_of) with no API key, via its public CSV export.

    Uses `requests` rather than pandas.read_csv(url): requests verifies TLS
    against certifi's CA bundle, which avoids SSL failures on Python builds
    that have no system certificate store configured.
    """
    end = resolve_as_of(as_of)

    def fetch() -> pd.Series:
        resp = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        series = pd.to_numeric(df.set_index("date")["value"], errors="coerce").dropna()
        series.name = series_id
        return series.loc[:end]

    return cached(f"fred_{series_id}_{end.date()}", fetch)


def fetch_cot_silver(as_of: str | None = None) -> pd.DataFrame:
    """Weekly CFTC Disaggregated Futures-Only positioning for COMEX silver.

    Managed Money is the closest COT category to speculative/trend
    positioning; Producer/Merchant is commercial hedgers -- the classic
    "smart money vs. crowd" pairing COT-based signals are built on. Both
    are expressed as net position (long - short) scaled by open interest,
    since raw contract counts trend upward with OI growth over a 5-10y
    window and aren't comparable across time on their own.

    The returned index is report_date + COT_PUBLICATION_LAG_DAYS, not the
    report's own "as of" date: nobody could see a Tuesday snapshot until it
    was published after Friday's close. Indexing by the as-of date directly
    would leak information into a walk-forward backtest days early.
    """
    end = resolve_as_of(as_of)

    def fetch() -> pd.DataFrame:
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
            index=pd.DatetimeIndex(report_date + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS), name="date"),
        )
        return out.loc[out.index <= end].sort_index().dropna()

    return cached(f"cot_silver_{end.date()}", fetch)


def as_published(series: pd.Series) -> pd.Series:
    """Re-index a FRED series by the date each value became public (see PUBLICATION_DATE)."""
    if series.name not in PUBLICATION_DATE:
        return series
    out = series.copy()
    out.index = PUBLICATION_DATE[series.name](series.index)
    # several observations can land on the same release date; the latest one wins
    return out[~out.index.duplicated(keep="last")]


def _asof_align(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill `series` onto `target_index`: each date gets the most
    recent value AT OR BEFORE it, never a later one. Never backfilled, so a
    walk-forward backtest over the result can't accidentally see the future.
    """
    combined = series.reindex(target_index.union(series.index)).sort_index().ffill()
    return combined.reindex(target_index)


def build_feature_frame(silver_prices: pd.Series, period: str = "5y", as_of: str | None = None) -> pd.DataFrame:
    """Merge silver against gold + macro series + COT positioning on silver's trading calendar,
    each aligned by when it was published rather than the date it describes."""
    gold = fetch_yf_close(GOLD_TICKER, period=period, as_of=as_of)
    real_rate = as_published(fetch_fred_series(REAL_RATE_10Y, as_of=as_of))
    breakeven = as_published(fetch_fred_series(BREAKEVEN_INFLATION_10Y, as_of=as_of))
    usd_index = as_published(fetch_fred_series(USD_INDEX, as_of=as_of))
    cot = fetch_cot_silver(as_of=as_of)

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
