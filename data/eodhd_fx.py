"""USD/JPY daily history from EODHD."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 60 * 60 * 12  # 12 hours

EODHD_BASE = "https://eodhistoricaldata.com/api"


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"eodhd_{safe}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def fetch_usdjpy_history(years: int = 5, symbol: str = "USDJPY.FOREX") -> pd.DataFrame:
    """Fetch daily USD/JPY history from EODHD.

    Returns a DataFrame with a DatetimeIndex (date) and columns:
      open, high, low, close, adjusted_close, volume
    """
    cache_file = _cache_path(symbol)
    if _is_fresh(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    api_key = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD")
    if not api_key:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise RuntimeError("EODHD_API_KEY is not set and no cached data is available.")

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=int(365.25 * years) + 30)

    url = f"{EODHD_BASE}/eod/{symbol}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "period": "d",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"EODHD returned no data for {symbol}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "adjusted_close", "volume"] if c in df.columns]
    df = df[keep].astype(float)
    df = df.dropna(subset=["close"])

    try:
        df.to_parquet(cache_file)
    except Exception:
        pass
    return df


def latest_close(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    import numpy as np
    return pd.Series(
        data=(np.log(df["close"]).diff().dropna()).values,
        index=df.index[1:],
        name="log_return",
    )
