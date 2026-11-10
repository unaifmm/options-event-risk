"""Bootstrap helpers and the volatility-index conversion."""
import numpy as np
import pandas as pd

from src import stats


def test_vix_of_about_16_is_a_one_percent_day():
    assert abs(stats.one_day_sd(15.87) - 1.0) < 0.001
    assert abs(stats.ABS_OVER_SD - 0.7979) < 1e-4


def test_bootstrap_interval_brackets_the_mean_and_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    small, big = rng.normal(0.1, 1, 50), rng.normal(0.1, 1, 5000)
    m, lo, hi = stats.bootstrap(small, n=2000)
    M, LO, HI = stats.bootstrap(big, n=2000)
    assert lo < m < hi and LO < M < HI and (HI - LO) < (hi - lo)


def test_ratio_and_difference():
    r, lo, hi = stats.bootstrap_ratio([1, 2, 3], [2, 4, 6], n=500)
    assert r == 0.5 and lo == 0.5 and hi == 0.5
    d, lo, hi = stats.bootstrap_diff([2.0] * 10, [1.0] * 10, n=500)
    assert d == 1.0 and lo == 1.0 and hi == 1.0


def test_trim_top_drops_the_same_share_of_the_largest_rows():
    x = pd.DataFrame({"v": [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 4.0, 8.0, 6.0, 0.0]})
    assert stats.trim_top(x, "v", 0).equals(x)
    assert sorted(stats.trim_top(x, "v", 0.10)["v"]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert sorted(stats.trim_top(x, "v", 0.25)["v"]) == [0, 1, 2, 3, 4, 5, 6]      # ceil(2.5) = 3 rows
    assert len(stats.trim_top(x.iloc[:5], "v", 0.02)) == 4                          # at least one row once share > 0


def test_risk_measures():
    x = [1.0, -2.0, 0.5, -1.0, 3.0]
    assert stats.max_drawdown(x) == -2.5                    # running total 1, -1, -0.5, -1.5: peak 1 -> -1.5
    assert stats.max_drawdown([-1.0, -1.0]) == -2.0         # from the starting point
    assert stats.expected_shortfall(x, 0.2) == -2.0
    assert stats.expected_shortfall(x, 0.4) == -1.5
    assert abs(stats.sharpe([1.0, -1.0] * 10, periods=1)) < 1e-12
    assert stats.sharpe([1.0, 2.0, 3.0], periods=4) == 2 * 2.0 / 1.0
