"""
Daily series from FRED (Federal Reserve Bank of St. Louis), free CSV downloads
with no key: the S&P 500 index close and the 3-month T-bill rate.
"""
from __future__ import annotations

import io

import pandas as pd

from . import config
from .webapi import Api


def download(series: dict = config.FRED_SERIES, start: str = config.START, api: Api | None = None) -> pd.DataFrame:
    api = api or Api("https://fred.stlouisfed.org", pause=0.5)
    cols = {}
    for sid, name in series.items():
        code, text = api.get("/graph/fredgraph.csv", {"id": sid, "cosd": start}, raw=True)
        if code != 200 or not isinstance(text, str) or "," not in text:
            print(f"  !! FRED {sid}: HTTP {code} - skipped")
            continue
        df = pd.read_csv(io.StringIO(text))
        s = pd.to_numeric(df[df.columns[1]], errors="coerce")      # '.' marks holidays
        s.index = pd.to_datetime(df[df.columns[0]])
        cols[sid] = s
        print(f"  FRED {sid:6s} {name}: {s.notna().sum():,} days, {s.index.min().date()} -> {s.index.max().date()}")
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out = out[out.index >= pd.Timestamp(start)]
    out.index.name = "date"
    out.to_csv(config.FRED_DIR / "daily.csv")
    return out


def load() -> pd.DataFrame:
    return pd.read_csv(config.FRED_DIR / "daily.csv", parse_dates=["date"]).set_index("date")
