"""
PROJECT 3 - STEP 0c: from when are SPY option prices available?

Step 0b got 1-minute option candles for 2026 but "no stored data" for 2022 and 2024.
Before designing the study we need the exact coverage. This check only ASKS the
vendor how much data a download would contain (the `estimate` call); it downloads
nothing, so it is quick and does not use up download quota.

  A. 1-minute candles of the at-the-money SPY call expiring the day after each
     month's CPI eve, month by month since November 2022: where does data start?
  B. The same day as a whole-chain trade file ("tick" data for every SPY contract):
     is the older history (the catalogue lists SPY option trades since 2014)
     reachable that way instead?
  C. Minute-by-minute implied volatility and Greeks ("greeks" data) for one recent
     contract: available, and from when?

Run:  .venv\\Scripts\\python step0c_coverage.py       (about a minute)
"""
from __future__ import annotations

import json
import os
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_check"
PROJECT1 = ROOT.parent / "information-diffusion-nasdaq-sp500" / "data" / "raw"
TZ = "America/New_York"
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def load_key() -> str:
    if os.environ.get("LSE_API_KEY"):
        return os.environ["LSE_API_KEY"]
    cfg = Path.home() / ".lse" / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("api_key"):
        return json.loads(cfg.read_text())["api_key"]
    raise SystemExit("No LSE API key found. Run:  .venv\\Scripts\\lse auth YOUR_KEY_HERE")


def spy_closes() -> pd.Series:
    files = [f for f in sorted((PROJECT1 / "lse").glob("SPY_*.parquet")) if int(f.stem[-4:]) >= 2022]
    d = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files], ignore_index=True)
    t = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(TZ)
    keep = (t.dt.time >= time(9, 30)) & (t.dt.time < time(16, 0))
    return d[keep].groupby(t[keep].dt.normalize().dt.tz_localize(None))["close"].last()


def occ(expiry: pd.Timestamp, right: str, strike: float) -> str:
    """OCC ticker, e.g. SPY260911C00758000 (call, strike 758, expiring 11 Sep 2026)."""
    return f"SPY{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def ask(client, symbol: str, timeframe: str, start, end, data: str = "candles") -> str:
    """'rows / bytes' from the vendor's estimate, or the reason it has none."""
    if len(symbol) > 6 and hasattr(client, "_occ"):    # the vendor spells contracts with the root padded:
        symbol = client._occ(symbol)                     # "SPY   260911C00758000"
    try:
        e = client.estimate("us_options", symbol, timeframe, str(pd.Timestamp(start).date()),
                            str(pd.Timestamp(end).date()), data=data)
    except Exception as exc:                            # noqa: BLE001 - "no data" comes back as an error
        return f"none ({str(exc)[:70]})"
    rows = e.get("rows") or e.get("row_count") or e.get("count")
    size = e.get("bytes") or e.get("size_bytes") or e.get("size")
    if rows is None and size is None:
        return "answer: " + json.dumps(e)[:120]
    mb = f", {float(size) / 1e6:,.1f} MB" if size is not None else ""
    return f"{int(rows):,} rows{mb}" if rows is not None else f"? rows{mb}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    say("PROJECT 3 - STEP 0c: SPY OPTION DATA COVERAGE (estimates only, nothing downloaded)\n")
    from lse import LSE
    client = LSE(api_key=load_key())

    close = spy_closes()
    days = close.index
    cal = pd.read_parquet(PROJECT1 / "economic_calendar_us.parquet")
    cpi = sorted(pd.Timestamp(x) for x in cal.loc[cal["event"].str.match(r"^Inflation Rate MoM( [A-Z]{3})?$")
                                                  & cal["actual"].notna(), "date"].unique())
    cpi = [d for d in cpi if d >= pd.Timestamp("2022-11-15") and d in days]

    say("A. 1-minute candles, at-the-money call expiring on CPI day, from the eve")
    got = []
    for day in cpi:
        eve = days[days < day].max()
        k = round(close[eve])
        res = ask(client, occ(day, "C", k), "1m", eve, day + pd.Timedelta(days=1))
        got.append((day, res[0].isdigit() and not res.startswith("0 rows")))    # real rows, not an error or "0 rows"
        say(f"   CPI {day:%Y-%m-%d}  {occ(day, 'C', k)}  ->  {res}")
    start = next((d for i, (d, _) in enumerate(got) if all(ok for _, ok in got[i:])), None)
    say(f"   1-minute option candles at every CPI from: {start:%Y-%m-%d}" if start is not None
        else "   no continuous run of CPIs with 1-minute option candles")

    say("\nB. Whole SPY option tape ('tick' = every trade, all contracts) for a few eves")
    for day in [cpi[0], cpi[len(cpi) // 2], cpi[-1]]:
        eve = days[days < day].max()
        say(f"   SPY ticks {eve:%Y-%m-%d}  ->  {ask(client, 'SPY', 'tick', eve, eve + pd.Timedelta(days=1))}")
        say(f"   SPY 1m    {eve:%Y-%m-%d}  ->  {ask(client, 'SPY', '1m', eve, eve + pd.Timedelta(days=1))}")

    say("\nC. Minute-by-minute implied volatility and Greeks ('greeks' data)")
    for day in [cpi[len(cpi) // 2], cpi[-1]]:
        eve = days[days < day].max()
        k = round(close[eve])
        res = ask(client, occ(day, "C", k), "1m", eve, day + pd.Timedelta(days=1), data="greeks")
        say(f"   {occ(day, 'C', k)}  ->  {res}")

    try:
        mk = [m for m in client.markets() if str(m.get("market") or m.get("code") or m.get("id")) == "us_options"]
        if mk:
            say("\nVendor's description of the options market: " + json.dumps(mk[0], default=str)[:700])
    except Exception as exc:                            # noqa: BLE001
        say(f"\nmarkets() FAILED: {str(exc)[:100]}")
    (OUT / "report_coverage.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved data_check/report_coverage.txt")


if __name__ == "__main__":
    main()
