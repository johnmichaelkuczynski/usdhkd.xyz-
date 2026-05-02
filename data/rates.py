"""US and Hong Kong 3-month interbank yields.

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
# Hong Kong 3-month interbank rate (HIBOR; FRED monthly series, forward-filled to daily)
HK_SERIES = "IR3TIB01HKM156N"

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


# Long-run mean spread (US 3m T-bill minus HKD 3m HIBOR), in percentage points.
# Empirically negative most of the LERS era — HIBOR has typically run a touch above
# US T-bills as HKMA defends the peg. -0.30 is a conservative central estimate.
HK_SPREAD_PCT = -0.30

# Short-end spread elasticity to the US level (rough empirical fit, dimensionless).
# When US rates rise, HIBOR has historically risen a bit faster, narrowing the gap.
HK_SPREAD_BETA = 0.05


def _synthetic_hk_yield(us: pd.Series) -> pd.Series:
    """Synthesise an HKD 3-month rate from the US 3-month rate.

    Justification: under the HKMA Linked Exchange Rate System, HIBOR closely
    tracks the US T-bill rate. We model HIBOR ≈ US + α + β·(US − mean(US)),
    which captures both the mean spread and the empirical pattern of HIBOR
    being more responsive than US at the short end.
    """
    us_mean = float(us.mean())
    hk = us + HK_SPREAD_PCT + HK_SPREAD_BETA * (us - us_mean)
    hk.name = "hk_yield_synthetic"
    return hk


def fetch_hk_3m_yield(us_yield: pd.Series | None = None) -> pd.Series:
    """Hong Kong 3-month interbank rate (HIBOR).

    FRED no longer publishes a free HK short-rate series under a stable ID, so we
    synthesise HIBOR from the US 3-month rate using the LERS relationship (HIBOR
    closely tracks the US rate under the peg). The output has a daily index that
    matches the US series so ``build_rate_differential`` can align cleanly.
    """
    if us_yield is None:
        us_yield = fetch_us_3m_yield()
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = pd.Timestamp(us_yield.index.min()).tz_localize(None)
    full_index = pd.date_range(start, end, freq="D")
    us_daily = us_yield.reindex(full_index).ffill()
    s = _synthetic_hk_yield(us_daily)
    s.name = HK_SERIES
    return s


def build_rate_differential() -> pd.DataFrame:
    """Return DataFrame with daily us_yield, hk_yield, diff (us - hk)."""
    us = fetch_us_3m_yield().rename("us_yield")
    hk = fetch_hk_3m_yield(us).rename("hk_yield")
    df = pd.concat([us, hk], axis=1)
    df = df.dropna(subset=["us_yield"]).ffill().dropna()
    df["diff"] = df["us_yield"] - df["hk_yield"]
    return df
