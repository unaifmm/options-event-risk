"""CBOE files and SPY daily closes."""
import pandas as pd

from src import cboe, spy


def test_cboe_parse_drops_an_unfinished_row_and_sorts():
    text = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
            "05/16/2022,26.40,26.40,26.40,26.40\n"
            "05/13/2022,33.63,33.63,33.63,33.63\n"
            "10/16/2026,7.77,25.66\n")                       # today's row, no low or close yet
    d = cboe.parse(text)
    assert list(d["date"].dt.strftime("%Y-%m-%d")) == ["2022-05-13", "2022-05-16"]
    assert list(d["close"]) == [33.63, 26.40]


def test_spy_daily_close_is_the_last_regular_hours_bar():
    ts = pd.to_datetime(["2026-09-10 13:29", "2026-09-10 13:30", "2026-09-10 19:59", "2026-09-10 20:00",
                         "2026-09-10 23:00"], utc=True)          # 09:29, 09:30, 15:59, 16:00, 19:00 New York
    df = pd.DataFrame({"timestamp": ts, "close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    c = spy.daily_close(df)
    assert list(c.index.strftime("%Y-%m-%d")) == ["2026-09-10"] and c.iloc[0] == 3.0


def test_spy_comes_from_the_free_us_stocks_feed_not_the_paid_nasdaq_one():
    from src.lse import resolve_dataset
    catalog = [{"dataset": d, "symbol": "SPY"} for d in ("nasdaq", "us_options", "us_stocks")]
    assert resolve_dataset(catalog, "SPY") == "us_stocks"
