"""
STEP 1 - Download everything the study needs (read-only: public data + your LSE key).

Run on YOUR computer, from this folder:

    .venv\\Scripts\\python 01_download_data.py                  # everything
    .venv\\Scripts\\python 01_download_data.py --parts options  # only some parts (see PARTS below)

Safe to stop and re-run at any time: finished pieces are saved in data/raw/ and
skipped next time. If the vendor's daily download limit is reached, the option
download stops cleanly; run the same command again later and it carries on.

What it downloads
  1. cboe      daily VIX1D, VIX9D, VIX and VIX3M (the options market's expected volatility)
  2. fred      daily S&P 500 close and the 3-month T-bill rate
  3. calendar  US macro calendar (London Strategic Edge) -> CPI, payrolls and FOMC dates
  4. spy       SPY 1-minute bars since May 2022 (London Strategic Edge)
  5. options   1-minute candles of SPY calls and puts at the 3 strikes around each eve's
               close, expiring the next day, for every trading day since mid-September 2025;
               plus minute implied volatility and Greeks of the at-the-money pair on release days
  6. chain     a snapshot of today's listed SPY contracts with the vendor's implied
               volatility and Greeks (for the pricing toolkit)

Everything is written under data/raw/ (not uploaded to GitHub).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import cboe, config, events, fred, options, spy

sys.stdout.reconfigure(errors="replace")
PARTS = ["cboe", "fred", "calendar", "spy", "options", "chain"]


def release_days() -> set:
    ev = events.load_events({k: events.RELEASES[k] for k in config.EVENT_KINDS})
    return set(pd.to_datetime(ev["date"]).dt.normalize())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS), help="comma-separated, from: " + ", ".join(PARTS))
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",")]

    if "cboe" in parts:
        print("=== CBOE volatility indices ===")
        cboe.download()
    if "fred" in parts:
        print("\n=== FRED: S&P 500 and T-bill ===")
        fred.download()

    client = None
    if any(p in parts for p in ("calendar", "spy", "options", "chain")):
        from src.lse import connect
        client = connect()
        try:
            u = client.usage()
            print("\nVendor usage today:", {k: v for k, v in u.items() if "key" not in str(k).lower()})
        except Exception as exc:                               # noqa: BLE001
            print(f"\n(usage not available: {str(exc)[:80]})")

    if "calendar" in parts:
        print("\n=== US macro calendar (London Strategic Edge) ===")
        from src.lse import download_calendar
        download_calendar(client, start="2022-01-01")
    if "spy" in parts:
        print("\n=== SPY 1-minute bars (London Strategic Edge) ===")
        from src.lse import resolve_dataset, update_symbol
        catalog = client.datasets()
        pd.DataFrame(catalog).to_csv(config.LSE_CATALOG, index=False)
        for name, symbol in config.LSE_SYMBOLS.items():
            ds = config.LSE_DATASETS.get(name) or resolve_dataset(catalog, symbol)
            if ds is None:
                print(f"  !! {name}: '{symbol}' not in the LSE catalog - skipped")
                continue
            print(f"--- {name} ({symbol}, dataset '{ds}')")
            update_symbol(client, name, symbol, ds, start=config.LSE_START)
    if "options" in parts:
        print("\n=== SPY options: 1-minute candles around every expiry since "
              f"{config.OPTIONS_START} (this is the long part) ===")
        cov = options.download(client, spy.daily_close(), release_days())
        first = cov[cov["rows"] > 0]["expiry"].min() if (cov["rows"] > 0).any() else None
        print(f"  first expiry with candles: {first}   (table: data/raw/options/coverage.csv)")
    if "chain" in parts:
        print("\n=== Today's SPY option chain ===")
        options.snapshot_chain(client)

    print("\nDone. Tell Claude it has finished.")


if __name__ == "__main__":
    main()
