"""
Small statistics helpers: bootstrap confidence intervals and the conversion from
an annualised volatility index to a one-day move.

Bootstrap: resample the observations with replacement many times, recompute the
statistic each time, and read the 2.5% and 97.5% percentiles. No formula needed,
and it copes with fat tails (a few huge days), which ordinary t-tests handle badly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ABS_OVER_SD = math.sqrt(2 / math.pi)      # average |move| of a normal distribution = 0.80 x its standard deviation
TRADING_DAYS = 252


def one_day_sd(index_level):
    """Annualised volatility in % (e.g. VIX1D = 16) -> one-day standard deviation in % (about 1.0)."""
    return np.asarray(index_level, dtype=float) / math.sqrt(TRADING_DAYS)


def bootstrap(x, stat=np.mean, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """(statistic, 2.5% and 97.5% bootstrap percentiles) of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(stat(x)) if len(x) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(stat(x)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_ratio(num, den, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a bootstrap interval, resampling (num, den) pairs together."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(num), len(num))
        draws.append(num[i].sum() / den[i].sum())
    return float(num.sum() / den.sum()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_diff(a, b, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b), resampling each group separately."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    draws = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def trim_top(x: pd.DataFrame, col: str, share: float) -> pd.DataFrame:
    """Drop the ceil(share x rows) rows with the largest `col`. Applied with the same share to every
    group, so no day is removed by hand."""
    k = int(math.ceil(share * len(x) - 1e-9))
    return x.drop(x.nlargest(k, col).index) if k else x


def sharpe(x, periods: int = TRADING_DAYS) -> float:
    """Annualised mean / standard deviation of a daily profit series (days with no trade count as 0)."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def max_drawdown(x) -> float:
    """Largest fall of the running total from its previous peak (same units as x, e.g. % of the index)."""
    total = np.cumsum(np.asarray(x, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], total]))[1:]
    return float((total - peak).min()) if len(total) else float("nan")


def expected_shortfall(x, share: float = 0.05) -> float:
    """Average of the worst `share` of outcomes (at least one): what a bad day costs on average."""
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(math.ceil(share * len(x) - 1e-9)))
    return float(x[:k].mean())
