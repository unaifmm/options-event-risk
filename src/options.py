"""
SPY options from London Strategic Edge: 1-minute candles of single contracts.

The trade studied: the evening before day E, the at-the-money call and put that
expire at E's close (a "straddle"). Its price over SPY's price is the move the
options market charges for; at expiry it pays |SPY close on E - strike|.

What step 0c showed about the data (3 Nov 2026):
  * candles exist only from about October 2025, one file per contract;
  * a contract with no data comes back as an error ("no stored data") - we save an
    empty file so it is not asked for again, except for the last few days, which
    are asked again in case the vendor is still filling them in;
  * minute implied volatility and Greeks exist for 2026 contracts.
The vendor limits how much can be downloaded per day: if it refuses, the download
stops cleanly and the same command later carries on where it stopped.
"""
from __future__ import annotations

import pandas as pd

from . import config

CANDLE_COLS = ["minute", "open", "high", "low", "close", "volume", "trades", "ticker"]


class QuotaReached(Exception):
    """The vendor refused more downloads for now."""


def occ(expiry, right: str, strike: float, root: str = config.OPTIONS_UNDERLYING) -> str:
    """OCC ticker, e.g. SPY260911C00758000 = SPY call, strike 758, expiring 11 Sep 2026."""
    return f"{root}{pd.Timestamp(expiry):%y%m%d}{right.upper()[0]}{int(round(float(strike) * 1000)):08d}"


def schedule(days: pd.DatetimeIndex, start, end=None) -> pd.DataFrame:
    """One row per expiry day E from `start` (to `end`), with its eve = the trading day before."""
    days = pd.DatetimeIndex(sorted(set(days)))
    d = pd.DataFrame({"eve": days[:-1], "expiry": days[1:]})
    d = d[d["expiry"] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d["expiry"] <= pd.Timestamp(end)]
    return d.reset_index(drop=True)


def strikes_for(spot: float, offsets=config.STRIKE_OFFSETS) -> list[int]:
    k = int(round(spot))
    return [k + o for o in offsets]


def candle_path(expiry, right: str, strike: float):
    return config.CANDLES_DIR / f"{occ(expiry, right, strike)}.parquet"


def greeks_path(expiry, right: str, strike: float):
    return config.GREEKS_DIR / f"{occ(expiry, right, strike)}.parquet"


def _is_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return getattr(exc, "status", None) == 429 or "429" in text or "quota" in text or "too many" in text


def _is_missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "no stored data" in text or "not available" in text


def _frame(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    if "time" in d.columns and "minute" not in d.columns:
        d = d.rename(columns={"time": "minute"})
    if len(d) and "minute" in d.columns:
        d["minute"] = pd.to_datetime(d["minute"], utc=True)
    return d


def fetch(client, expiry, right: str, strike: float, eve, what: str = "candles") -> pd.DataFrame:
    """Candles (or minute Greeks) of one contract from the eve's open to the expiry's close.
    Empty frame if the vendor has none; QuotaReached if it refuses for now."""
    start, end = pd.Timestamp(eve).date().isoformat(), (pd.Timestamp(expiry) + pd.Timedelta(days=1)).date().isoformat()
    try:
        if what == "candles":
            rows = client.option_candles(config.OPTIONS_UNDERLYING, strike=strike, expiry=pd.Timestamp(expiry).date().isoformat(),
                                         type="call" if right.upper().startswith("C") else "put", start=start, end=end)
        else:
            symbol = client._occ(occ(expiry, right, strike)) if hasattr(client, "_occ") else occ(expiry, right, strike)
            rows = client.rows("us_options", symbol, "1m", start, end, data="greeks")
    except Exception as exc:                                 # noqa: BLE001 - sorted out below
        if _is_quota(exc):
            raise QuotaReached(str(exc)[:200]) from exc
        if _is_missing(exc):
            return pd.DataFrame(columns=CANDLE_COLS if what == "candles" else ["minute"])
        raise
    return _frame(rows)


def needs(path, expiry, today, recheck_days: int = 5) -> bool:
    """Download unless the file exists - but ask again for an empty file of the last few days."""
    if not path.exists():
        return True
    recent = pd.Timestamp(expiry) >= pd.Timestamp(today).normalize() - pd.Timedelta(days=recheck_days)
    return recent and len(pd.read_parquet(path)) == 0


def download(client, closes: pd.Series, release_days: set, start=config.OPTIONS_START, today=None) -> pd.DataFrame:
    """Candles of the calls and puts at the strikes around each eve's close, for every
    expiry day since `start`; minute Greeks of the at-the-money pair on release days.
    Returns (and saves) a coverage table built from the files on disk."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz=config.TZ).date())
    sched = schedule(closes.index, start, end=today - pd.Timedelta(days=1))
    asked = got = 0
    try:
        for i, r in sched.iterrows():
            spot = closes[r["eve"]]
            jobs = [("candles", right, k) for k in strikes_for(spot) for right in ("C", "P")]
            if r["expiry"] in release_days:
                jobs += [("greeks", right, int(round(spot))) for right in ("C", "P")]
            for what, right, k in jobs:
                path = (candle_path if what == "candles" else greeks_path)(r["expiry"], right, k)
                if not needs(path, r["expiry"], today):
                    continue
                d = fetch(client, r["expiry"], right, k, r["eve"], what)
                d.to_parquet(path, index=False)
                asked += 1
                got += bool(len(d))
            if (i + 1) % 20 == 0:
                print(f"  {r['expiry']:%Y-%m-%d}: {i + 1} of {len(sched)} expiry days done "
                      f"({asked} contracts asked this run, {got} with data)", flush=True)
    except QuotaReached as exc:
        print(f"\n  !! The vendor refused more downloads for now ({exc}).\n"
              f"     Everything finished is saved: run the same command again later (or tomorrow).")
    cov = coverage(sched, closes)
    print(f"  options: {asked} contracts asked this run, {got} with data; "
          f"{(cov['rows'] > 0).sum()} of {len(cov)} contracts on disk have candles")
    return cov


def coverage(sched: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    rows = []
    for _, r in sched.iterrows():
        for k in strikes_for(closes[r["eve"]]):
            for right in ("C", "P"):
                p = candle_path(r["expiry"], right, k)
                n = len(pd.read_parquet(p)) if p.exists() else -1
                rows.append({"eve": r["eve"].date(), "expiry": r["expiry"].date(), "right": right, "strike": k,
                             "atm": k == int(round(closes[r["eve"]])), "rows": n})
    cov = pd.DataFrame(rows)
    cov.to_csv(config.OPTIONS_COVERAGE, index=False)
    return cov


def snapshot_chain(client, max_dte: int = config.CHAIN_MAX_DTE) -> pd.DataFrame:
    """Today's listed SPY contracts with the vendor's implied volatility and Greeks."""
    d = pd.DataFrame(client.options(config.OPTIONS_UNDERLYING, max_dte=max_dte, limit=20000))
    if len(d):
        path = config.CHAIN_DIR / f"SPY_{pd.Timestamp.now(tz=config.TZ):%Y-%m-%d_%H%M}.parquet"
        d.to_parquet(path, index=False)
        print(f"  chain: {len(d):,} contracts (expiring within {max_dte} days) -> {path.name}")
    return d


def load_candles(expiry, right: str, strike: float) -> pd.DataFrame:
    """Saved candles of one contract, with New York time in 'ny'."""
    d = pd.read_parquet(candle_path(expiry, right, strike))
    if len(d):
        d["ny"] = pd.to_datetime(d["minute"], utc=True).dt.tz_convert(config.TZ)
    return d
