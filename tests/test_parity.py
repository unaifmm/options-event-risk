"""Settlement calendar, dividends and the parity gap."""
import pandas as pd

from src import parity

# NYSE trading days, Sep - Oct 2026 (Labor Day 7 Sep closed; Columbus Day 12 Oct open)
TRADING = pd.bdate_range("2026-09-01", "2026-10-30").drop(pd.Timestamp("2026-09-07"))


def test_t_plus_one_and_the_thursday_weekend():
    s = parity.settlement_days(TRADING)
    assert parity.next_settlement("2026-09-15", s) == pd.Timestamp("2026-09-16")
    assert parity.financing_days("2026-09-15", "2026-09-16", s) == 1     # Tue eve -> Wed expiry
    assert parity.financing_days("2026-09-17", "2026-09-18", s) == 3     # Thu eve -> Fri expiry: K paid Monday
    assert parity.financing_days("2026-09-18", "2026-09-21", s) == 1     # Fri eve -> Mon expiry


def test_columbus_day_trades_but_does_not_settle():
    s = parity.settlement_days(TRADING)
    assert pd.Timestamp("2026-10-12") not in s
    assert parity.financing_days("2026-10-08", "2026-10-09", s) == 4     # K paid Tue 13 Oct
    assert parity.financing_days("2026-10-09", "2026-10-12", s) == 0     # both settle Tue 13 Oct


def test_dividend_only_when_the_ex_date_is_the_expiry():
    divs = {"2026-09-18": 1.8888}
    assert parity.dividend("2026-09-17", "2026-09-18", divs) == 1.8888
    assert parity.dividend("2026-09-18", "2026-09-21", divs) == 0.0
    assert parity.dividend("2026-09-16", "2026-09-17", divs) == 0.0


def test_expected_miss_and_implied_rate_invert_each_other():
    m = parity.expected_miss(720, 3.6, 3, 0.0)
    assert abs(m - 720 * 0.036 * 3 / 360) < 1e-12
    assert abs(parity.implied_rate_pct(m, 720, 3) - 3.6) < 1e-9
    assert abs(parity.expected_miss(720, 3.6, 3, 1.8888) - (m - 1.8888)) < 1e-12
