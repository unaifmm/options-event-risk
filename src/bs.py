"""
Black-Scholes(-Merton) for European options: price, Greeks and implied volatility.

Inputs (numpy arrays or numbers, broadcast together):
  s        spot price of the underlying (SPY)
  k        strike
  t        time to expiry in years (calendar days / 365)
  r        risk-free rate, continuously compounded, a year (0.039 = 3.9%)
  sigma    volatility, a year (0.16 = 16%)
  right    "C"/"call" or "P"/"put" (one value or an array)
  div_pv   present value of the dividends paid before expiry, in $ ("escrowed
           dividend": the option is priced on s - div_pv)

The formulas, with f = s - div_pv and d1, d2 = [ln(f/k) + (r +/- sigma^2/2) t] / (sigma sqrt t):
  call = f N(d1) - k e^(-rt) N(d2)          put = k e^(-rt) N(-d2) - f N(-d1)
so call - put = f - k e^(-rt): put-call parity holds by construction.

SPY options are American. For the short-dated, near-the-money options studied here
early exercise is worth almost nothing, so the European formula is the standard
approximation; deep in-the-money options are where the two can differ.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

DAYS_A_YEAR = 365.0
SQRT_2PI = np.sqrt(2 * np.pi)


def is_call(right) -> np.ndarray:
    r = np.char.upper(np.asarray(right).astype(str))
    return np.char.startswith(r, "C")


def _pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def _d1d2(s, k, t, r, sigma, div_pv):
    f = np.asarray(s, float) - div_pv
    vol = np.asarray(sigma, float) * np.sqrt(np.maximum(t, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(f / k) + (r + 0.5 * np.asarray(sigma, float) ** 2) * t) / vol
    return f, d1, d1 - vol


def price(s, k, t, r, sigma, right, div_pv=0.0):
    """Option price. At or after expiry (t <= 0): the intrinsic value."""
    s, k, t, r, sigma = (np.asarray(x, float) for x in (s, k, t, r, sigma))
    call = is_call(right)
    f, d1, d2 = _d1d2(s, k, t, r, sigma, div_pv)
    disc = k * np.exp(-r * np.maximum(t, 0))
    c = f * ndtr(d1) - disc * ndtr(d2)
    p = disc * ndtr(-d2) - f * ndtr(-d1)
    out = np.where(call, c, p)
    intrinsic = np.where(call, np.maximum(f - k, 0), np.maximum(k - f, 0))
    return np.where(t > 0, out, intrinsic)


def greeks(s, k, t, r, sigma, right, div_pv=0.0) -> dict:
    """delta, gamma (per $1 of spot), vega (per 1 volatility point = 0.01) and theta (per
    calendar day) - the units option screens use."""
    s, k, t, r, sigma = (np.asarray(x, float) for x in (s, k, t, r, sigma))
    call = is_call(right)
    f, d1, d2 = _d1d2(s, k, t, r, sigma, div_pv)
    sq = np.sqrt(np.maximum(t, 1e-12))
    pdf, disc = _pdf(d1), k * np.exp(-r * t)
    decay = -f * pdf * sigma / (2 * sq)
    return {
        "delta": np.where(call, ndtr(d1), ndtr(d1) - 1),
        "gamma": pdf / (f * sigma * sq),
        "vega": f * pdf * sq / 100,
        "theta": np.where(call, decay - r * disc * ndtr(d2), decay + r * disc * ndtr(-d2)) / DAYS_A_YEAR,
    }


def implied_vol(target, s, k, t, r, right, div_pv=0.0, lo=1e-4, hi=10.0, steps=80):
    """The volatility at which the formula gives the observed price. Solved by bisection,
    which always converges because the price rises with volatility. NaN when no volatility
    fits: a price below what the option is worth at zero volatility, above its maximum,
    or at/after expiry."""
    target, s, k, t, r = np.broadcast_arrays(*(np.asarray(x, float) for x in (target, s, k, t, r)))
    right = np.broadcast_to(np.asarray(right), target.shape)
    div_pv = np.broadcast_to(np.asarray(div_pv, float), target.shape)
    a, b = np.full(target.shape, lo), np.full(target.shape, hi)
    floor = price(s, k, t, r, a, right, div_pv)
    cap = price(s, k, t, r, b, right, div_pv)
    ok = (t > 0) & np.isfinite(target) & (target > floor) & (target < cap)
    for _ in range(steps):
        mid = 0.5 * (a + b)
        low = price(s, k, t, r, mid, right, div_pv) < target
        a, b = np.where(low, mid, a), np.where(low, b, mid)
    out = np.where(ok, 0.5 * (a + b), np.nan)
    return out if out.ndim else float(out)


def american_price(s, k, t, r, sigma, right, steps: int = 500, early: bool = True) -> float:
    """American option (no dividends) on a Cox-Ross-Rubinstein binomial tree: at every node
    the holder keeps the better of exercising now and holding on. One option at a time.
    early=False gives the European price on the same tree, so the difference between the
    two is the early-exercise value without the tree's own rounding error."""
    if t <= 0:
        return float(max(s - k, 0) if is_call(right) else max(k - s, 0))
    dt = t / steps
    u = np.exp(sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - 1 / u) / (u - 1 / u)
    disc = np.exp(-r * dt)
    sign = 1.0 if bool(is_call(right)) else -1.0
    j = np.arange(steps + 1)
    v = np.maximum(sign * (s * u ** (2 * j - steps) - k), 0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        v = disc * (p * v[1:] + (1 - p) * v[:-1])
        if early:
            v = np.maximum(v, sign * (s * u ** (2 * j - i) - k))
    return float(v[0])
