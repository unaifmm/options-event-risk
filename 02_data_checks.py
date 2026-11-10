"""
STEP 2 - Check the data before using any of it ("validation first").

Every expiry day E since October 2025: the at-the-money SPY call and put that
expire at E's close, priced at the close of the eve (the trading day before).

  A. Coverage: which expiry days have both options, and which were half days
     (the market closed at 13:00, so a "16:00 close" does not exist).
  B. Stale prices: how long before 16:00 did each option last trade on the eve?
     An old trade is not the price you could have traded at.
  C. Put-call parity: for options on the same strike and expiry,
     call - put = SPY - strike + interest on the strike - any dividend before expiry
     (src/parity.py explains both terms; the interest covers the T+1 settlement gap,
     so it is three days' worth on Thursday eves). Prices taken in the same minute
     must satisfy it to within a few cents once both terms are counted.
  D. Expiry: in the last minutes before expiry an option is worth its intrinsic
     value |SPY - strike| on the right side, so its last price must be close to it.
  E. The bridge to the long sample: the straddle's price as a % of SPY (the move
     you pay for) against VIX1D's implied absolute move the same evening.
  F. Clock check with VIX1D (as in step 0): expected volatility should be higher
     on release eves and fall once the news is out.

Run:  .venv\\Scripts\\python 02_data_checks.py
Writes results/02_*.csv and figures/02_bridge.png.
"""
from __future__ import annotations

import math
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cboe, config, days, fred, options, parity, spy

ABS_OVER_SD = math.sqrt(2 / math.pi)        # average |move| of a normal distribution = 0.80 x its standard deviation
KIND_COLOR = {"CPI": "#d1495b", "Payrolls": "#edae49", "FOMC": "#00798c", "normal": "#9aa5b1"}


def spy_minutes() -> pd.Series:
    """SPY 1-minute closes in regular hours since the option sample starts, by bar START (New York)."""
    files = [f for f in sorted(config.DATA_LSE.glob("SPY_*.parquet")) if int(f.stem[-4:]) >= 2025]
    d = spy.regular_hours(pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files]))
    return d.set_index("ny")["close"].sort_index()


def day_kinds(dates) -> pd.Series:
    return days.kinds(dates)


def last_trade(d: pd.DataFrame, day, before: str = "16:00"):
    """(price, minutes before `before`) of the last trade on `day` that started before `before` New York."""
    if not len(d):
        return np.nan, np.nan
    cut = pd.Timestamp(f"{pd.Timestamp(day).date()} {before}", tz=config.TZ)
    x = d[(d["ny"] < cut) & (d["ny"] >= cut.normalize())]
    if not len(x):
        return np.nan, np.nan
    r = x.iloc[-1]
    return float(r["close"]), (cut - r["ny"]).total_seconds() / 60 - 1   # 0 = traded in the 15:59 bar


def parity_gap(c: pd.DataFrame, p: pd.DataFrame, s_min: pd.Series, day, k: float):
    """(C - P) - (S - K) in the last minute before 16:00 in which both options traded."""
    cut = pd.Timestamp(f"{pd.Timestamp(day).date()} 16:00", tz=config.TZ)
    both = (c[(c["ny"] < cut) & (c["ny"] >= cut.normalize())].set_index("ny")["close"].rename("c").to_frame()
            .join(p.set_index("ny")["close"].rename("p"), how="inner"))
    both = both.join(s_min.rename("s"), how="inner")
    if not len(both):
        return np.nan, np.nan
    r = both.iloc[-1]
    return float((r["c"] - r["p"]) - (r["s"] - k)), (cut - both.index[-1]).total_seconds() / 60 - 1


def build() -> pd.DataFrame:
    closes = spy.daily_close()
    s_min = spy_minutes()
    last_bar = s_min.groupby(s_min.index.normalize().tz_localize(None)).apply(lambda x: x.index[-1].time())
    vix = cboe.load("VIX1D")
    settle = parity.settlement_days(closes.index)
    tbill = fred.load()["DTB3"].ffill()
    sched = options.schedule(closes.index, config.OPTIONS_START,
                             end=pd.Timestamp.now(tz=config.TZ).tz_localize(None).normalize() - pd.Timedelta(days=1))
    kinds = day_kinds(sched["expiry"])
    rows = []
    for _, r in sched.iterrows():
        eve, day = r["eve"], r["expiry"]
        s0, s1 = closes[eve], closes.get(day, np.nan)
        k = options.strikes_for(s0)[len(config.STRIKE_OFFSETS) // 2]
        c, p = (options.load_candles(day, x, k) if options.candle_path(day, x, k).exists() else pd.DataFrame()
                for x in ("C", "P"))
        c_px, c_age = last_trade(c, eve)
        p_px, p_age = last_trade(p, eve)
        par, par_age = parity_gap(c, p, s_min, eve, k) if len(c) and len(p) else (np.nan, np.nan)
        c_end, _ = last_trade(c, day)
        p_end, _ = last_trade(p, day)
        s_end = s_min[s_min.index.normalize() == pd.Timestamp(day).tz_localize(config.TZ)]
        s_end = float(s_end.iloc[-1]) if len(s_end) else np.nan
        v = vix.get(eve, np.nan)
        fin_days, rate, div = parity.financing_days(eve, day, settle), float(tbill.asof(eve)), parity.dividend(eve, day)
        rows.append({
            "eve": eve.date(), "expiry": day.date(), "kind": kinds[day], "spy_eve": s0, "spy_expiry": s1, "strike": k,
            "half_day": any(last_bar.get(x, time(15, 59)) < time(15, 59) for x in (eve, day)),
            "call_eve": c_px, "put_eve": p_px, "call_minutes_stale": c_age, "put_minutes_stale": p_age,
            "parity_miss": par, "parity_minutes_stale": par_age,
            "financing_days": fin_days, "tbill_pct": rate, "dividend": div,
            "parity_expected": parity.expected_miss(k, rate, fin_days, div),
            "call_end_minus_intrinsic": c_end - max(s_end - k, 0), "put_end_minus_intrinsic": p_end - max(k - s_end, 0),
            "straddle_pct": 100 * (c_px + p_px) / s0,
            "vix1d_eve": v, "vix1d_move_pct": v / math.sqrt(252) * ABS_OVER_SD,
            "actual_move_pct": 100 * abs(s1 / s0 - 1),
            "seller_pnl_pct": 100 * ((c_px + p_px) - abs(s1 - k)) / s0,
        })
    out = pd.DataFrame(rows)
    out["parity_left"] = out["parity_miss"] - out["parity_expected"]
    return out


def report(d: pd.DataFrame) -> None:
    ok = d["call_eve"].notna() & d["put_eve"].notna()
    print(f"A. Coverage: {len(d)} expiry days {d['expiry'].min()} -> {d['expiry'].max()}; "
          f"both options priced on {ok.sum()}; first with data {d.loc[ok, 'expiry'].min()}")
    half = d[d["half_day"]]
    print(f"   half days (13:00 close) on the eve or expiry: {len(half)} "
          + ("(" + ", ".join(str(x) for x in half['expiry']) + ") - left out below" if len(half) else ""))
    by_kind = d[ok].groupby("kind").size()
    print("   expiry days by kind: " + ", ".join(f"{k} {n}" for k, n in by_kind.items()))
    d = d[ok & ~d["half_day"]]

    stale = pd.concat([d["call_minutes_stale"], d["put_minutes_stale"]])
    print(f"\nB. Stale prices on the eve: last trade in the 15:59 minute for {100 * (stale == 0).mean():.0f}% of options,"
          f" within 5 minutes for {100 * (stale <= 5).mean():.0f}%, worst {stale.max():.0f} minutes before 16:00")

    m = d["parity_miss"].dropna()
    print(f"\nC. Put-call parity at the eve close ({len(m)} days, prices in the same minute,"
          f" median {d['parity_minutes_stale'].median():.0f} minutes before 16:00):")
    print(f"   (C - P) - (SPY - K): median {m.median():+.3f} $, 90% between {m.quantile(0.05):+.3f} and"
          f" {m.quantile(0.95):+.3f} $ (SPY ~$700: 1 cent = 0.0014%)")
    print("   Why: call - put = SPY - K + interest on K over the settlement gap - dividend (src/parity.py)")
    print(f"   {'':27s}{'eves':>5s}{'observed':>10s}{'expected':>10s}{'left':>8s}   (medians, $)")
    plain = d[(d["dividend"] == 0) & d["parity_miss"].notna()]
    for lab, g in (("1 day of interest", plain[plain["financing_days"] == 1]),
                   ("3+ days (Thursday eves...)", plain[plain["financing_days"] >= 3])):
        print(f"   {lab:27s}{len(g):5d}{g['parity_miss'].median():+10.3f}{g['parity_expected'].median():+10.3f}"
              f"{g['parity_left'].median():+8.3f}")
    for _, r in d[(d["dividend"] > 0) & d["parity_miss"].notna()].iterrows():
        print(f"   ex-dividend {r['expiry']}{1:9d}{r['parity_miss']:+10.3f}{r['parity_expected']:+10.3f}"
              f"{r['parity_left']:+8.3f}   (dividend ${r['dividend']:.2f}, {r['financing_days']} days of interest)")
    left = d["parity_left"].dropna()
    print(f"   after interest and dividends: median {left.median():+.3f} $, {100 * (left.abs() <= 0.10).mean():.0f}% within"
          f" 10 cents, {100 * (left.abs() <= 0.25).mean():.0f}% within 25 cents")
    paid = plain[plain["financing_days"] > 0]
    implied = [parity.implied_rate_pct(x, k, n) for x, k, n in zip(paid["parity_miss"], paid["strike"], paid["financing_days"])]
    print(f"   interest rate implied by the options: median {np.median(implied):.2f}% a year;"
          f" 3-month T-bill {paid['tbill_pct'].median():.2f}%")
    for _, r in d.loc[left.abs().sort_values(ascending=False).index[:5]].iterrows():
        print(f"   largest left over: eve {r['eve']} strike {r['strike']}: {r['parity_left']:+.2f} $"
              f" (observed {r['parity_miss']:+.2f})")

    for side in ("call", "put"):
        e = d[f"{side}_end_minus_intrinsic"].dropna()
        print(f"{'D' if side == 'call' else ' '}. Expiry, {side}s: last price minus intrinsic value: median {e.median():+.3f} $,"
              f" 90% between {e.quantile(0.05):+.3f} and {e.quantile(0.95):+.3f} $")

    b = d.dropna(subset=["straddle_pct", "vix1d_move_pct"])
    ratio = b["straddle_pct"] / b["vix1d_move_pct"]
    print(f"\nE. The bridge ({len(b)} eves): straddle price vs VIX1D's implied absolute move:"
          f" correlation {b[['straddle_pct', 'vix1d_move_pct']].corr().iloc[0, 1]:.3f},"
          f" straddle / VIX1D median {ratio.median():.3f} (90% between {ratio.quantile(0.05):.2f} and {ratio.quantile(0.95):.2f})")
    for k, g in b.groupby("kind"):
        rr = g["straddle_pct"] / g["vix1d_move_pct"]
        print(f"   {k:9s} {len(g):3d} eves: straddle {g['straddle_pct'].mean():.2f}%  VIX1D move {g['vix1d_move_pct'].mean():.2f}%"
              f"  ratio {rr.median():.3f}  actual move {g['actual_move_pct'].mean():.2f}%"
              f"  seller P&L {g['seller_pnl_pct'].mean():+.3f}% of SPY (preview, no costs)")


def clocks() -> None:
    v = cboe.load("VIX1D")
    kinds = day_kinds(v.index)
    d = pd.DataFrame({"kind": kinds, "eve": v.shift(1), "after": v}).dropna()
    print("\nF. Clock check, VIX1D (May 2022 - now), median on the eve vs the evening after:")
    for k in list(config.EVENT_KINDS) + ["normal"]:
        g = d[d["kind"] == k]
        print(f"   {k:9s} {len(g):4d} days: {g['eve'].median():5.1f} -> {g['after'].median():5.1f}")


def figure(d: pd.DataFrame) -> None:
    d = d.dropna(subset=["straddle_pct", "vix1d_move_pct"])
    d = d[~d["half_day"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, g in d.groupby("kind"):
        a1.scatter(g["vix1d_move_pct"], g["straddle_pct"], s=18 if k == "normal" else 34, alpha=0.75,
                   color=KIND_COLOR[k], label=f"{k} ({len(g)})", edgecolors="none")
    lim = [0, max(d["vix1d_move_pct"].max(), d["straddle_pct"].max()) * 1.05]
    a1.plot(lim, lim, color="#555", linestyle="--", linewidth=0.8)
    a1.set_xlim(lim), a1.set_ylim(lim)
    a1.set_xlabel("VIX1D implied absolute move, % (eve close)")
    a1.set_ylabel("SPY at-the-money straddle, % of SPY (eve close)")
    a1.set_title("Two measures of the move the options market charges for")
    a1.legend(loc="upper left", frameon=False)
    t = pd.to_datetime(d["expiry"])
    a2.plot(t, d["vix1d_move_pct"], color="#555", linewidth=1, label="VIX1D implied move")
    a2.plot(t, d["straddle_pct"], color="#00798c", linewidth=1, label="straddle price")
    ev = d[d["kind"] != "normal"]
    a2.scatter(pd.to_datetime(ev["expiry"]), ev["straddle_pct"], s=14, color=[KIND_COLOR[k] for k in ev["kind"]], zorder=3)
    a2.set_ylabel("% of SPY")
    a2.set_title("Day by day (dots: release days)")
    a2.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_bridge.png", dpi=150)
    print("\nsaved figure -> figures/02_bridge.png")


def main() -> None:
    d = build()
    d.to_csv(config.RESULTS / "02_daily.csv", index=False)
    report(d)
    clocks()
    figure(d)


if __name__ == "__main__":
    main()
