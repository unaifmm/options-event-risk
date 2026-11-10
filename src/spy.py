"""
SPY (the S&P 500 ETF) from London Strategic Edge 1-minute bars: daily closes
and trading days. The options in this project are SPY options, so their strikes
and payoffs are set from SPY's own price.

Bars are stamped with their START time in UTC. Regular hours are 09:30-16:00
New York; the bar starting 15:59 closes at 16:00, the official close.
From 2026 the vendor also stores extended hours, which are dropped here.
"""
from __future__ import annotations

from datetime import time

import pandas as pd

from . import config


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars starting 09:30-15:59 New York; adds a New York time column 'ny'."""
    t = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(config.TZ)
    keep = (t.dt.time >= time(9, 30)) & (t.dt.time < time(16, 0))
    return df[keep].assign(ny=t[keep])


def daily_close(df: pd.DataFrame | None = None) -> pd.Series:
    """SPY's 16:00 close per trading day (the close of the last regular-hours bar)."""
    if df is None:
        files = sorted(config.DATA_LSE.glob("SPY_*.parquet"))
        if not files:
            raise FileNotFoundError("No SPY files - run: python 01_download_data.py --parts spy")
        df = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files], ignore_index=True)
    r = regular_hours(df)
    out = r.groupby(r["ny"].dt.normalize().dt.tz_localize(None))["close"].last()
    out.index.name = "date"
    return out
