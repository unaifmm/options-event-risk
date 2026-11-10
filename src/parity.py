"""
Put-call parity for SPY options that expire the next trading day.

Owning a call and being short a put on the same strike K is the same as owning SPY
and owing K at expiry. So, at the eve's close,

    call - put = SPY - K + (interest on K until it is paid) - (any dividend before expiry)

Two details decide the size of the two extra terms:
  * When money moves. Trades settle one settlement day later (T+1). Buying SPY at
    the eve's close is paid on the next settlement day; with the call and put, K is
    paid when the exercise settles, one settlement day after expiry. The interest is
    on K for the calendar days in between: one day usually, three on a Thursday eve
    (Friday's exercise settles on Monday), more around bank holidays.
  * Dividends. SPY's holders at the close before the ex-dividend date get the
    dividend; the option holder does not, so the call is worth less and the put more.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config

DAY_COUNT = 360          # money-market convention (T-bill rates are quoted on it)


def settlement_days(trading_days) -> pd.DatetimeIndex:
    """Days on which trades settle: exchange trading days on which banks are also open
    (stocks trade on Columbus Day and Veterans Day, but nothing settles), plus a few
    future bank business days so the last expiry still has a settlement date."""
    t = pd.DatetimeIndex(trading_days).normalize().unique().sort_values()
    fed = USFederalHolidayCalendar()
    hol = fed.holidays(t.min(), t.max() + pd.Timedelta(days=15))
    after = pd.bdate_range(t.max() + pd.Timedelta(days=1), periods=10, freq="C", holidays=hol)
    return t[~t.isin(hol)].append(after)


def next_settlement(day, settle: pd.DatetimeIndex) -> pd.Timestamp:
    """First settlement day strictly after `day` (T+1)."""
    i = settle.searchsorted(pd.Timestamp(day).normalize(), side="right")
    return settle[i] if i < len(settle) else pd.NaT


def financing_days(eve, expiry, settle: pd.DatetimeIndex) -> int:
    """Calendar days between paying for SPY (eve + T+1) and paying K (expiry + T+1)."""
    return int((next_settlement(expiry, settle) - next_settlement(eve, settle)).days)


def dividend(eve, expiry, dividends: dict = config.SPY_DIVIDENDS) -> float:
    """Dividend per share whose ex-date falls after the eve and on or before expiry."""
    e, x = pd.Timestamp(eve), pd.Timestamp(expiry)
    return float(sum(v for d, v in dividends.items() if e < pd.Timestamp(d) <= x))


def expected_miss(strike: float, rate_pct: float, days: int, div: float) -> float:
    """What (call - put) - (SPY - K) should be: interest on K minus the dividend."""
    return strike * rate_pct / 100 * days / DAY_COUNT - div


def implied_rate_pct(miss: float, strike: float, days: int) -> float:
    """The interest rate at which the observed miss is exactly the interest on K (no dividend)."""
    return miss * DAY_COUNT / (strike * days) * 100 if days else float("nan")
