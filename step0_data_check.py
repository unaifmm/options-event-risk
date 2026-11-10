"""
PROJECT 3 - STEP 0: is there enough data to answer "Do options overprice macro news?"

The idea. The evening before a CPI, payrolls or Fed day, option prices say how big a
move the market expects in the S&P 500 the next day. After the day we know how big
the move really was. Do options charge too much for macro news (option sellers win
on average), too little, or about the right amount?

This one-off check looks at every source we would use and writes what it finds to
data_check/report.txt. Nothing is analysed properly yet.

  A. CBOE volatility indices, daily (free CSV files from cboe.com):
       VIX1D = expected S&P 500 volatility over the NEXT trading day (since May 2022)
       VIX9D = next 9 days    VIX = next 30 days    VIX3M = next 3 months
     They are annualised standard deviations in %: a VIX1D of 16 means a one-day
     standard deviation of 16 / sqrt(252) = about 1.0%.
  B. The S&P 500 itself: daily closes from FRED, and the 1-minute SPY and ES futures
     files already on disk from project 1 (London Strategic Edge). The two must agree.
  C. Release dates: CPI, payrolls and FOMC, from the project 1 economic calendar.
  D. First look, with a timing check: VIX1D should jump on the eve of a release and
     drop once it is out. If it does not, the dates or the timing are wrong.
  E. Option chains for the pricing toolkit (Black-Scholes, implied volatility):
     SPY options from Yahoo Finance through the yfinance package, if installed.

Run:  python step0_data_check.py        (needs internet; about a minute)
"""
from __future__ import annotations

import io
import math
import urllib.request
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data_check"
PROJECT1 = ROOT.parent / "information-diffusion-nasdaq-sp500" / "data" / "raw"
TZ = "America/New_York"

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
INDICES = {"VIX1D": "next day", "VIX9D": "next 9 days", "VIX": "next 30 days", "VIX3M": "next 3 months"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
RELEASES = {"CPI": r"^Inflation Rate MoM( [A-Z]{3})?$",
            "Payrolls": r"^Non Farm Payrolls( [A-Z]{3})?$",
            "FOMC": r"^Fed Interest Rate Decision$"}
ABS_OVER_SD = math.sqrt(2 / math.pi)   # average |move| of a normal distribution = 0.80 x its standard deviation

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (student research project)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------------------
# A. Volatility indices
# ---------------------------------------------------------------------------
def cboe_indices() -> dict[str, pd.Series]:
    say("A. CBOE volatility indices (daily, annualised %)")
    (RAW / "cboe").mkdir(parents=True, exist_ok=True)
    out = {}
    for name, horizon in INDICES.items():
        try:
            raw = fetch(CBOE.format(name))
        except Exception as e:                       # noqa: BLE001 - report and carry on
            say(f"   {name:6s} FAILED to download: {e}")
            continue
        (RAW / "cboe" / f"{name}.csv").write_bytes(raw)
        d = pd.read_csv(io.BytesIO(raw))
        d.columns = [c.strip().lower() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
        no_close = int(d["close"].isna().sum())
        flat = int((d[["open", "high", "low", "close"]].nunique(axis=1) == 1).sum())
        c = d.dropna(subset=["close"]).set_index("date")["close"]
        say(f"   {name:6s} ({horizon:13s}) {len(d):5d} days  {d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}"
            f"   median {c.median():5.1f}   last close {c.iloc[-1]:5.1f}")
        if no_close or flat:
            say(f"          check: {no_close} rows without a close, {flat} rows where open = high = low = close")
        out[name] = c
    return out


# ---------------------------------------------------------------------------
# B. The S&P 500
# ---------------------------------------------------------------------------
def regular_hours_close(files: list[Path]) -> pd.Series:
    """Last 1-minute close before 16:00 New York on each trading day."""
    d = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files], ignore_index=True)
    t = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(TZ)
    keep = (t.dt.time >= time(9, 30)) & (t.dt.time < time(16, 0))
    d, t = d[keep], t[keep]
    return d.groupby(t.dt.normalize().dt.tz_localize(None))["close"].last()


def sp500() -> pd.Series | None:
    say("\nB. The S&P 500")
    spx = None
    try:
        raw = fetch(FRED.format("SP500"))
        (RAW / "fred").mkdir(parents=True, exist_ok=True)
        (RAW / "fred" / "SP500.csv").write_bytes(raw)
        d = pd.read_csv(io.BytesIO(raw), na_values=".")
        d.columns = ["date", "close"]
        d["date"] = pd.to_datetime(d["date"])
        spx = d.dropna().set_index("date")["close"]
        say(f"   S&P 500 index close (FRED)  {len(spx)} days  {spx.index.min():%Y-%m-%d} to {spx.index.max():%Y-%m-%d}"
            f"   last {spx.iloc[-1]:,.2f}")
    except Exception as e:                           # noqa: BLE001
        say(f"   FRED SP500 FAILED to download: {e}")

    for sym in ("SPY", "ES.F"):
        files = sorted((PROJECT1 / "lse").glob(f"{sym}_*.parquet"))
        if not files:
            say(f"   {sym:4s} 1-minute: no files in {PROJECT1 / 'lse'}")
            continue
        last = pd.read_parquet(files[-1], columns=["timestamp"])["timestamp"].max()
        say(f"   {sym:4s} 1-minute (project 1): {len(files)} yearly files {files[0].stem[-4:]}-{files[-1].stem[-4:]},"
            f" last bar {pd.Timestamp(last):%Y-%m-%d %H:%M} UTC")
        if sym == "SPY" and spx is not None:
            close = regular_hours_close(files[-3:])
            both = pd.concat([np.log(close).diff().rename("spy"), np.log(spx).diff().rename("spx")], axis=1).dropna()
            if len(both) > 20:
                gap = (both["spy"] - both["spx"]).abs() * 1e4
                say(f"        SPY 16:00 close vs the index close, daily returns: correlation {both.corr().iloc[0, 1]:.4f},"
                    f" median gap {gap.median():.1f} bp ({len(both)} days) - they should agree closely")
    return spx


# ---------------------------------------------------------------------------
# C. Release dates
# ---------------------------------------------------------------------------
def releases(start: pd.Timestamp) -> pd.DataFrame:
    say("\nC. Release dates (project 1 economic calendar)")
    f = PROJECT1 / "economic_calendar_us.parquet"
    if not f.exists():
        say(f"   not found: {f}")
        return pd.DataFrame(columns=["kind", "date"])
    cal = pd.read_parquet(f)
    rows = []
    for kind, pattern in RELEASES.items():
        sel = cal[cal["event"].str.match(pattern) & cal["actual"].notna()]
        rows += [{"kind": kind, "date": pd.Timestamp(x)} for x in sorted(sel["date"].unique())]
    ev = pd.DataFrame(rows)
    ev = ev[ev["date"] >= start].sort_values("date").reset_index(drop=True)
    counts = ev["kind"].value_counts()
    say(f"   since {start:%Y-%m-%d}: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in RELEASES)
        + f"   (calendar ends {pd.Timestamp(cal['date'].max()):%Y-%m-%d})")
    same_day = ev[ev.duplicated("date", keep=False)]
    if len(same_day):
        say(f"   {same_day['date'].nunique()} days with two releases (e.g. CPI and FOMC): "
            + ", ".join(f"{d:%Y-%m-%d}" for d in sorted(same_day["date"].unique())[:6]))
    return ev


# ---------------------------------------------------------------------------
# D. First look and timing check
# ---------------------------------------------------------------------------
def first_look(vix1d: pd.Series, spx: pd.Series, ev: pd.DataFrame) -> None:
    say("\nD. First look (a preview to check the data, not a result)")
    days = vix1d.index.intersection(spx.index)
    v = vix1d.reindex(days)
    r = np.log(spx.reindex(days)).diff()
    eve = v.shift(1)                          # VIX1D at the previous close prices the day ahead
    after = v                                 # ...and at the close of the day itself, the day after
    kind = pd.Series("normal day", index=days)
    for k in RELEASES:
        kind[kind.index.isin(ev.loc[ev["kind"] == k, "date"])] = k
    d = pd.DataFrame({"kind": kind, "eve": eve, "after": after, "ret": r}).dropna()
    d["implied_sd_pct"] = d["eve"] / math.sqrt(252)
    d["implied_abs_pct"] = d["implied_sd_pct"] * ABS_OVER_SD
    d["actual_abs_pct"] = d["ret"].abs() * 100
    say("   Timing check: VIX1D the evening before vs the evening after (median). It should be higher before a"
        " release\n   and fall once the news is out; on normal days there is no reason for a drop.")
    say(f"   {'':12s}{'days':>6s}{'VIX1D before':>14s}{'after':>8s}   implied |move|   actual |move|   actual / implied")
    for k in list(RELEASES) + ["normal day"]:
        g = d[d["kind"] == k]
        if not len(g):
            continue
        say(f"   {k:12s}{len(g):6d}{g['eve'].median():14.1f}{g['after'].median():8.1f}"
            f"   {g['implied_abs_pct'].mean():10.2f}%   {g['actual_abs_pct'].mean():11.2f}%   {g['actual_abs_pct'].mean() / g['implied_abs_pct'].mean():12.2f}")
    say("   implied |move| = VIX1D / sqrt(252) x 0.80 (the average absolute move of a normal distribution);"
        " a ratio\n   below 1 means the S&P moved less than options priced (sellers ahead), above 1 more.")
    d.to_csv(OUT / "first_look_days.csv")


# ---------------------------------------------------------------------------
# E. Option chains
# ---------------------------------------------------------------------------
def option_chains() -> None:
    say("\nE. Option chains (for the pricing toolkit)")
    try:
        import yfinance as yf
    except ImportError:
        say("   yfinance is not installed. Run:  pip install yfinance   then run this check again.")
        return
    try:
        tk = yf.Ticker("SPY")
        expiries = tk.options
        if not expiries:
            say("   SPY: no expiries returned (Yahoo may be rate-limiting; try again later)")
            return
        chain = tk.option_chain(expiries[0])
        calls = chain.calls
        say(f"   SPY: {len(expiries)} expiries from {expiries[0]} to {expiries[-1]}; nearest one has "
            f"{len(calls)} calls and {len(chain.puts)} puts")
        say(f"   columns: {', '.join(calls.columns)}")
        calls.to_csv(OUT / "spy_calls_nearest_expiry.csv", index=False)
    except Exception as e:                           # noqa: BLE001
        say(f"   SPY option chain FAILED: {e}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    say("PROJECT 3 - STEP 0 DATA CHECK  (" + pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M New York") + ")\n")
    vol = cboe_indices()
    spx = sp500()
    start = vol["VIX1D"].index.min() if "VIX1D" in vol else pd.Timestamp("2022-05-13")
    ev = releases(start)
    if "VIX1D" in vol and spx is not None and len(ev):
        first_look(vol["VIX1D"], spx, ev)
    option_chains()
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved data_check/report.txt")


if __name__ == "__main__":
    main()
