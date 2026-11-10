"""
PROJECT 3 - STEP 0b: can we get real option prices around CPI, payrolls and Fed days?

Step 0 used VIX1D, an index computed from S&P 500 options. London Strategic Edge
(your data vendor) also has every SPY option trade since 2014, as 1-minute candles
per contract. With those we can price the real trade:

    the evening before a release, buy (or sell) the at-the-money call + put that
    expire the next day (a "straddle"), and hold it to expiry.

The straddle's price divided by SPY's price is the move the options market charges
for; the next day's actual move decides who won. This check tries a few past
releases and reports what comes back. SPY has had an expiry every trading day only
since mid-November 2022, so earlier releases are not tried.

Needs: your LSE key (saved by `lse auth`, read from ~/.lse/config.json - never printed),
the lse-data package (pip install -r requirements.txt), and project 1's SPY 1-minute
files and economic calendar (for SPY's price and the release dates).

Run:  .venv\\Scripts\\python step0b_options_check.py      (about a minute)
"""
from __future__ import annotations

import json
import math
import os
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_check"
PROJECT1 = ROOT.parent / "information-diffusion-nasdaq-sp500" / "data" / "raw"
TZ = "America/New_York"
DAILY_EXPIRIES_FROM = pd.Timestamp("2022-11-15")     # SPY: Tuesday and Thursday expiries added in Nov 2022
RELEASES = {"CPI": r"^Inflation Rate MoM( [A-Z]{3})?$",
            "Payrolls": r"^Non Farm Payrolls( [A-Z]{3})?$",
            "FOMC": r"^Fed Interest Rate Decision$"}
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def load_key() -> str:
    """The key saved by `lse auth` (the same file the other projects use). Never printed."""
    if os.environ.get("LSE_API_KEY"):
        return os.environ["LSE_API_KEY"]
    cfg = Path.home() / ".lse" / "config.json"
    if cfg.exists():
        key = json.loads(cfg.read_text()).get("api_key", "")
        if key:
            return key
    raise SystemExit("No LSE API key found. Run:  .venv\\Scripts\\lse auth YOUR_KEY_HERE")


def spy_closes() -> pd.Series:
    """SPY's last 1-minute close before 16:00 New York, per day (project 1 files)."""
    files = sorted((PROJECT1 / "lse").glob("SPY_*.parquet"))
    files = [f for f in files if int(f.stem[-4:]) >= 2022]
    d = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files], ignore_index=True)
    t = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(TZ)
    keep = (t.dt.time >= time(9, 30)) & (t.dt.time < time(16, 0))
    return d[keep].groupby(t[keep].dt.normalize().dt.tz_localize(None))["close"].last()


def pick_releases(days: pd.DatetimeIndex) -> pd.DataFrame:
    """A handful of releases spread over the sample: the first CPI with daily SPY
    expiries, one from 2024, and the latest CPI, payrolls and FOMC on disk."""
    cal = pd.read_parquet(PROJECT1 / "economic_calendar_us.parquet")
    rows = []
    for kind, pattern in RELEASES.items():
        sel = cal[cal["event"].str.match(pattern) & cal["actual"].notna()]
        rows += [{"kind": kind, "date": pd.Timestamp(x)} for x in sorted(sel["date"].unique())]
    ev = pd.DataFrame(rows)
    ev = ev[(ev["date"] >= DAILY_EXPIRIES_FROM) & ev["date"].isin(days)].sort_values("date")
    cpi = ev[ev["kind"] == "CPI"]
    picks = [cpi.iloc[0], cpi[cpi["date"].dt.year == 2024].iloc[len(cpi[cpi["date"].dt.year == 2024]) // 2],
             cpi.iloc[-1], ev[ev["kind"] == "Payrolls"].iloc[-1], ev[ev["kind"] == "FOMC"].iloc[-1]]
    return pd.DataFrame(picks).drop_duplicates("date").reset_index(drop=True)


def candles(client, strike: float, expiry: pd.Timestamp, right: str, start: pd.Timestamp) -> pd.DataFrame:
    rows = client.option_candles("SPY", strike=strike, expiry=expiry.date().isoformat(), type=right,
                                 start=start.date().isoformat(),
                                 end=(expiry + pd.Timedelta(days=1)).date().isoformat())
    d = pd.DataFrame(rows)
    if len(d):
        d["minute"] = pd.to_datetime(d["minute"], utc=True)
        d["ny"] = d["minute"].dt.tz_convert(TZ)
    return d


def last_before(d: pd.DataFrame, day: pd.Timestamp, clock: str) -> tuple[float, str]:
    """Close of the last candle on `day` before `clock` New York time."""
    if not len(d):
        return math.nan, "-"
    cut = pd.Timestamp(f"{day.date()} {clock}", tz=TZ)
    x = d[(d["ny"] < cut) & (d["ny"].dt.normalize() == cut.normalize())]
    if not len(x):
        return math.nan, "-"
    r = x.iloc[-1]
    return float(r["close"]), r["ny"].strftime("%H:%M")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    say("PROJECT 3 - STEP 0b: SPY OPTION PRICES AROUND RELEASES\n")
    from lse import LSE
    client = LSE(api_key=load_key())

    close = spy_closes()
    days = close.index
    ev = pick_releases(days)
    say(f"SPY daily closes on disk: {days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}. Releases tried: {len(ev)}\n")

    rows = []
    for _, e in ev.iterrows():
        day = e["date"]
        eve = days[days < day].max()
        s0, s1 = close[eve], close[day]
        k = round(s0)                                   # at-the-money strike (SPY strikes are $1 apart)
        say(f"{e['kind']:8s} {day:%Y-%m-%d}   eve {eve:%Y-%m-%d}: SPY {s0:.2f} -> next close {s1:.2f}"
            f" ({100 * (s1 / s0 - 1):+.2f}%)   strike {k}, expiry {day:%Y-%m-%d}")
        got = {}
        for right in ("call", "put"):
            try:
                d = candles(client, k, day, right, eve)
            except Exception as exc:                    # noqa: BLE001 - report and carry on
                say(f"   {right:4s}: request FAILED: {str(exc)[:160]}")
                d = pd.DataFrame()
            got[right] = d
            if len(d):
                say(f"   {right:4s}: {len(d):4d} minutes with trades, {d['ny'].min():%m-%d %H:%M} to "
                    f"{d['ny'].max():%m-%d %H:%M} New York; volume {int(d['volume'].sum()):,}; "
                    f"columns {', '.join(c for c in d.columns if c not in ('ny',))}")
            else:
                say(f"   {right:4s}: no candles returned")
        c, tc = last_before(got["call"], eve, "16:00")
        p, tp = last_before(got["put"], eve, "16:00")
        straddle = c + p
        payoff = abs(s1 - k)
        if math.isfinite(straddle):
            say(f"   eve close: call {c:.2f} ({tc}) + put {p:.2f} ({tp}) = straddle {straddle:.2f}"
                f" = {100 * straddle / s0:.2f}% of SPY.  At expiry it pays |{s1:.2f} - {k}| = {payoff:.2f}"
                f"  -> seller {'+' if straddle >= payoff else ''}{straddle - payoff:.2f} per share")
        rows.append({"kind": e["kind"], "date": day.date(), "eve": eve.date(), "spy_eve": s0, "spy_day": s1,
                     "strike": k, "call": c, "put": p, "straddle": straddle,
                     "straddle_pct": 100 * straddle / s0, "payoff": payoff})
        say()

    pd.DataFrame(rows).to_csv(OUT / "options_check.csv", index=False)
    say("Notes: candle times are converted from UTC to New York; a straddle uses the last trade before 16:00"
        " on the eve.\nCompare 'straddle % of SPY' with step 0's VIX1D 'implied |move|' on the same eves.")

    try:
        chain = client.options("SPY", max_dte=1, limit=50)
        say(f"\nLive chain (today, for the pricing toolkit): {len(chain)} rows; fields: {', '.join(chain[0].keys()) if chain else '-'}")
    except Exception as exc:                            # noqa: BLE001
        say(f"\nLive chain request FAILED: {str(exc)[:160]}")
    (OUT / "report_options.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved data_check/report_options.txt")


if __name__ == "__main__":
    main()
