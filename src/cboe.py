"""
CBOE volatility indices: free daily CSV files (date, open, high, low, close).

VIX1D is the options market's expected volatility of the S&P 500 over the next
trading day, annualised, in %: a value of 16 means a one-day standard deviation of
16 / sqrt(252) = about 1.0%. VIX9D, VIX and VIX3M do the same for 9 days, 30 days
and 3 months.

Data traps seen in step 0: the latest row can be incomplete (no low or close
yet), and early rows can have open = high = low = close (only a close was
published). Only the close is used.
"""
from __future__ import annotations

import io

import pandas as pd

from . import config
from .webapi import Api


def parse(text: str) -> pd.DataFrame:
    """CBOE history CSV -> one row per date with open/high/low/close; rows without
    a close (e.g. today's, still in progress) are dropped."""
    d = pd.read_csv(io.StringIO(text))
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    return d[["date", "open", "high", "low", "close"]].reset_index(drop=True)


def download(names: dict = config.CBOE_INDICES, api: Api | None = None) -> dict[str, pd.DataFrame]:
    api = api or Api("https://cdn.cboe.com", pause=0.5)
    out = {}
    for name, what in names.items():
        path = config.CBOE_URL.format(name).replace("https://cdn.cboe.com", "")
        code, text = api.get(path, raw=True)
        if code != 200 or not isinstance(text, str) or "DATE" not in text.upper():
            print(f"  !! CBOE {name}: HTTP {code} - skipped")
            continue
        d = parse(text)
        d.to_csv(config.CBOE_DIR / f"{name}.csv", index=False)
        out[name] = d
        print(f"  CBOE {name:6s} {what}: {len(d):,} days, {d['date'].min():%Y-%m-%d} -> {d['date'].max():%Y-%m-%d}")
    return out


def load(name: str) -> pd.Series:
    """Daily close of one index, indexed by date."""
    d = pd.read_csv(config.CBOE_DIR / f"{name}.csv", parse_dates=["date"])
    return d.set_index("date")["close"]
