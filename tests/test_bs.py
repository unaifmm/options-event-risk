"""Black-Scholes against textbook numbers, put-call parity, finite differences, and the IV round trip."""
import numpy as np

from src import bs


def test_hull_prices():
    # Hull, Options, Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
    assert abs(bs.price(42, 40, 0.5, 0.10, 0.20, "C") - 4.76) < 0.005
    assert abs(bs.price(42, 40, 0.5, 0.10, 0.20, "P") - 0.81) < 0.005


def test_hull_greeks():
    # Hull's Greeks example: S=49, K=50, r=5%, sigma=20%, T=20 weeks -> call 2.40, delta 0.522,
    # gamma 0.066, vega 12.1 per 100 points, theta -4.31 a year
    g = bs.greeks(49, 50, 20 / 52, 0.05, 0.20, "C")
    assert abs(bs.price(49, 50, 20 / 52, 0.05, 0.20, "C") - 2.40) < 0.005
    assert abs(g["delta"] - 0.522) < 0.001
    assert abs(g["gamma"] - 0.066) < 0.0005
    assert abs(g["vega"] * 100 - 12.1) < 0.05
    assert abs(g["theta"] * 365 - (-4.31)) < 0.01


def test_put_call_parity_with_dividend():
    s, k, t, r, v, d = 700.0, np.array([680.0, 700.0, 720.0]), 30 / 365, 0.04, 0.18, 1.9
    c, p = bs.price(s, k, t, r, v, "C", d), bs.price(s, k, t, r, v, "P", d)
    assert np.allclose(c - p, (s - d) - k * np.exp(-r * t))


def test_greeks_match_finite_differences():
    s, k, t, r, v = 700.0, 705.0, 3 / 365, 0.04, 0.17
    for right in ("C", "P"):
        g, h = bs.greeks(s, k, t, r, v, right), 0.01
        p = lambda **kw: bs.price(kw.get("s", s), k, kw.get("t", t), r, kw.get("v", v), right)
        assert abs(g["delta"] - (p(s=s + h) - p(s=s - h)) / (2 * h)) < 1e-6
        assert abs(g["gamma"] - (p(s=s + h) - 2 * p() + p(s=s - h)) / h ** 2) < 1e-4
        assert abs(g["vega"] - (p(v=v + 1e-4) - p(v=v - 1e-4)) / 2e-4 / 100) < 1e-6
        assert abs(g["theta"] - (p(t=t - 1e-5) - p(t=t + 1e-5)) / 2e-5 / 365) < 1e-5


def test_implied_vol_round_trip_and_impossible_prices():
    s, k = 700.0, np.array([650.0, 690.0, 700.0, 710.0, 760.0])
    for t in (5 / (24 * 365), 1 / 365, 30 / 365):          # 5 hours, 1 day, 30 days
        for right in ("C", "P"):
            v = np.array([0.35, 0.20, 0.15, 0.13, 0.12])
            target = bs.price(s, k, t, 0.04, v, right)
            back = bs.implied_vol(target, s, k, t, 0.04, right)
            floor = bs.price(s, k, t, 0.04, 1e-4, right)    # worth at ~zero volatility
            ok = target - floor > 1e-6                      # no time value left = no information about volatility
            assert ok.sum() >= 3
            assert np.allclose(back[ok], v[ok], atol=1e-6)
    assert np.isnan(bs.implied_vol(5.0, 700, 690, 1 / 365, 0.04, "C"))   # below intrinsic value
    assert np.isnan(bs.implied_vol(2.0, 700, 700, 0.0, 0.04, "C"))       # expired
    assert bs.is_call(["call", "P", "C", "put"]).tolist() == [True, False, True, False]


def test_american_tree():
    # without dividends an American call is never exercised early: same as the European price
    assert abs(bs.american_price(700, 705, 10 / 365, 0.04, 0.17, "C", 800) - bs.price(700, 705, 10 / 365, 0.04, 0.17, "C")) < 0.01
    # Hull's American put: S=50, K=50, r=10%, sigma=40%, T=5 months -> about 4.28 (European 4.08)
    am = bs.american_price(50, 50, 5 / 12, 0.10, 0.40, "P", 800)
    assert abs(am - 4.28) < 0.02
    assert am > bs.price(50, 50, 5 / 12, 0.10, 0.40, "P") + 0.15


def test_european_tree_matches_the_formula():
    for right in ("C", "P"):
        tree = bs.american_price(700, 690, 9 / 365, 0.04, 0.18, right, 800, early=False)
        assert abs(tree - float(bs.price(700, 690, 9 / 365, 0.04, 0.18, right))) < 0.01
