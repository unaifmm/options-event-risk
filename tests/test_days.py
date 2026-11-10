"""The market clock: early closes and time to expiry."""
import pandas as pd

from src import days


def test_early_closes():
    assert days.close_hour("2026-11-27") == 13        # day after Thanksgiving
    assert days.close_hour("2025-11-28") == 13
    assert days.close_hour("2026-12-24") == 13        # Thursday
    assert days.close_hour("2025-07-03") == 13        # Thursday
    assert days.close_hour("2026-11-04") == 16
    assert days.close_hour("2026-11-26") == 16        # Thanksgiving itself: closed, not an early close


def test_years_to_expiry():
    now = pd.Timestamp("2026-11-03 16:00", tz="America/New_York")
    assert abs(days.years_to_expiry(now, "2026-11-04") - 1 / 365) < 1e-12
    assert abs(days.years_to_expiry(now, "2026-11-27") - (23 * 24 + 21) / (24 * 365)) < 1e-12   # 13:00 close
    assert days.expiry_time("2026-11-04") == pd.Timestamp("2026-11-04 21:00", tz="UTC")
