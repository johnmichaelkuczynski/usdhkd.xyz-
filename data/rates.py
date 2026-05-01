"""US and Japan 3-month government bond yields.

Primary source: FRED (no API key required for the JSON observations endpoint via fredgraph CSV).
We use the public CSV download which does not require an API key. Falls back to cached data on failure.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 60 * 60 * 12

# US 3-month T-bill secondary market rate
US_SERIES = "DTB3"
# Japan 3-month interbank rate (close proxy when JGB 3m is sparse)
JP_SERIES = "IRSTCB01JPM156N"

FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _cache_path(series: str) -> Path:
    return CACHE_DIR / f"fred_{series}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _fetch_fred_series(series: str) -> pd.Series:
    cache_file = _cache_path(series)
    if _is_fresh(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            return df.iloc[:, 0]
        except Exception:
            pass

    try:
        resp = requests.get(FREDGRAPH, params={"id": series}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # FRED CSV is "DATE" + series column; values may be "." for missing
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna()
        s = df.set_index(date_col)[val_col].sort_index()
        s.name = series
        try:
            s.to_frame().to_parquet(cache_file)
        except Exception:
            pass
        return s
    except Exception:
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df.iloc[:, 0]
        raise


def fetch_us_3m_yield() -> pd.Series:
    return _fetch_fred_series(US_SERIES)


def fetch_jp_3m_yield() -> pd.Series:
    """Japan short-term rate. The IRSTCB01JPM156N series is monthly; we forward-fill to daily."""
    s = _fetch_fred_series(JP_SERIES)
    # forward-fill to daily so it can join cleanly with daily US/FX data
    full_index = pd.date_range(s.index.min(), pd.Timestamp.utcnow().normalize(), freq="D")
    s_daily = s.reindex(full_index).ffill()
    s_daily.name = JP_SERIES
    return s_daily


def build_rate_differential() -> pd.DataFrame:
    """Return DataFrame with daily us_yield, jp_yield, diff (us - jp)."""
    us = fetch_us_3m_yield().rename("us_yield")
    jp = fetch_jp_3m_yield().rename("jp_yield")
    df = pd.concat([us, jp], axis=1)
    df = df.dropna(subset=["us_yield"]).ffill().dropna()
    df["diff"] = df["us_yield"] - df["jp_yield"]
    return df
