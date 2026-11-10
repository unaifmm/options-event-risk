"""The option downloader against a fake vendor: tickers, schedule, missing data, limits."""
import pandas as pd
import pytest

from src import config, options


def test_occ_ticker():
    assert options.occ("2026-09-11", "C", 758) == "SPY260911C00758000"
    assert options.occ("2026-09-11", "put", 757.5) == "SPY260911P00757500"


def test_schedule_pairs_each_expiry_with_the_previous_trading_day():
    days = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07"])   # 3 July holiday + weekend
    s = options.schedule(days, start="2026-07-02")
    assert [(a.strftime("%m-%d"), b.strftime("%m-%d")) for a, b in zip(s["eve"], s["expiry"])] == \
           [("07-01", "07-02"), ("07-02", "07-06"), ("07-06", "07-07")]


def test_strikes_around_the_close():
    assert options.strikes_for(757.6) == [757, 758, 759]


class FakeVendor:
    def __init__(self, refuse_after=None):
        self.calls, self.refuse_after = [], refuse_after

    def option_candles(self, underlying, strike, expiry, type, start, end):
        self.calls.append((expiry, type, strike))
        if self.refuse_after is not None and len(self.calls) > self.refuse_after:
            raise RuntimeError("[429] too many export requests")
        if expiry < "2025-10-01":
            raise RuntimeError("[404] Omega has no stored data for this selection")
        return [{"minute": f"{start}T19:59:00Z", "open": 3.0, "high": 3.1, "low": 2.9, "close": 3.0,
                 "volume": 10, "trades": 2, "ticker": "X"}]


@pytest.fixture()
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CANDLES_DIR", tmp_path / "c"); (tmp_path / "c").mkdir()
    monkeypatch.setattr(config, "GREEKS_DIR", tmp_path / "g"); (tmp_path / "g").mkdir()
    monkeypatch.setattr(config, "OPTIONS_COVERAGE", tmp_path / "coverage.csv")
    return tmp_path


def test_download_keeps_empty_files_for_missing_data_and_skips_them_next_time(folders):
    closes = pd.Series([700.2, 701.0, 702.4], index=pd.to_datetime(["2025-09-29", "2025-09-30", "2025-10-01"]))
    v = FakeVendor()
    cov = options.download(v, closes, release_days=set(), start="2025-09-30", today="2025-12-01")
    assert len(v.calls) == 12                                # 2 expiries x 3 strikes x call and put
    assert (cov.loc[cov["expiry"].astype(str) == "2025-09-30", "rows"] == 0).all()   # no stored data: empty file
    assert (cov.loc[cov["expiry"].astype(str) == "2025-10-01", "rows"] == 1).all()
    v2 = FakeVendor()
    options.download(v2, closes, release_days=set(), start="2025-09-30", today="2025-12-01")
    assert v2.calls == []                                    # everything on disk: nothing asked again


def test_download_stops_cleanly_when_the_vendor_refuses(folders):
    closes = pd.Series([700.2, 701.0, 702.4], index=pd.to_datetime(["2025-10-01", "2025-10-02", "2025-10-03"]))
    v = FakeVendor(refuse_after=4)
    cov = options.download(v, closes, release_days=set(), start="2025-10-02", today="2025-12-01")
    assert (cov["rows"] > 0).sum() == 4 and (cov["rows"] == -1).sum() == 8   # 4 saved, the rest left for next run
