"""Which trading days carry a scheduled release (CPI, payrolls, FOMC decision)."""
from __future__ import annotations

import pandas as pd

from . import config, events

PRIORITY = ("FOMC", "Payrolls", "CPI")     # a day with two releases is labelled by the last in this order (CPI first)


def kinds(days) -> pd.Series:
    """'CPI', 'Payrolls', 'FOMC' or 'normal' for each date in `days`."""
    ev = events.load_events({k: events.RELEASES[k] for k in config.EVENT_KINDS})
    out = pd.Series("normal", index=pd.DatetimeIndex(days))
    for k in PRIORITY:
        out[out.index.isin(pd.to_datetime(ev.loc[ev["kind"] == k, "date"]))] = k
    return out


# --- the market clock ------------------------------------------------------------------
NY = "America/New_York"


def close_hour(day) -> int:
    """NYSE closes at 13:00 New York time the day after Thanksgiving, and on 3 July and
    24 December when they fall Monday-Thursday (on a Friday they are holidays); else 16:00."""
    d = pd.Timestamp(day)
    thursdays = pd.date_range(f"{d.year}-11-01", f"{d.year}-11-30", freq="W-THU")
    if d.normalize() == thursdays[3] + pd.Timedelta(days=1):
        return 13
    if (d.month, d.day) in ((7, 3), (12, 24)) and d.dayofweek <= 3:
        return 13
    return 16


def expiry_time(day) -> pd.Timestamp:
    """When an option expiring on `day` stops trading for good: the close, in UTC."""
    d = pd.Timestamp(day).normalize()
    return (d + pd.Timedelta(hours=close_hour(d))).tz_localize(NY).tz_convert("UTC")


def years_to_expiry(now, expiry_day) -> float:
    """Calendar time to the expiry close in years of 365 days (the vendor's convention)."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    return (expiry_time(expiry_day) - now).total_seconds() / (365 * 86400)
