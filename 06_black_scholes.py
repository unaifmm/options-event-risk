"""
STEP 6 - My own option pricer, checked against the vendor and used on the whole chain.

src/bs.py prices European options with Black-Scholes, gives their Greeks, finds the
implied volatility (the volatility at which the formula gives the market price) and
prices American options on a binomial tree. tests/test_bs.py checks it against
textbook numbers. Here it is checked against real data and put to work:

  1. Minute by minute: the at-the-money call and put on the release days in the
     sample. From the same price, SPY price, time and rate as the vendor, do I get
     the vendor's implied volatility and Greeks?
  2. The whole chain (snapshot of 3-4 Nov 2026: every strike, expiries up to 44 days),
     where the clock matters: 27 Nov (day after Thanksgiving) closes at 13:00.
  3. American vs European: SPY options can be exercised early. How much is that worth
     for the options in this study, and where does it start to matter?
  4. What the solver shows: the volatility smile (crash protection costs more), and a
     VIX rebuilt from SPY options with the CBOE's formula - against the official VIX9D
     and VIX, and against at-the-money volatility.

Run:  .venv\\Scripts\\python 06_black_scholes.py
Writes results/06_*.csv and figures/06_black_scholes.png.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import bs, cboe, config, days

SMILE_EXPIRIES = ("2026-11-06", "2026-11-12", "2026-11-20", "2026-12-11")
EX_DIVIDEND = ("2026-12-18",)                    # SPY goes ex-dividend on the December expiry: my simple VIX ignores dividends
WINGS = (-5.0, 1.5)                              # traded strikes must reach 5 standard deviations down and 1.5 up
VIX_PAIR = ("2026-11-27", "2026-12-04")          # expiries either side of 30 days, as the CBOE does
VIX9D_EXPIRY = "2026-11-12"                      # 9 days after 3 Nov
MIN_PRICE = 0.05                                 # cheaper options: a 1-cent tick is too big a share of the price


def vendor_minutes() -> pd.DataFrame:
    rows = []
    for f in sorted(config.GREEKS_DIR.glob("*.parquet")):
        x = pd.read_parquet(f)
        if len(x):
            rows.append(x.assign(contract=f.stem, right=f.stem[-9], strike=int(f.stem[-8:]) / 1000))
    if not rows:
        raise SystemExit("no minute Greeks found - run 01_download_data.py options first")
    d = pd.concat(rows, ignore_index=True)
    return d[d["iv"].notna() & (d["years_to_expiry"] > 0)].copy()


def chain() -> tuple[pd.DataFrame, float]:
    files = sorted(config.CHAIN_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("no chain snapshot found - run 01_download_data.py chain first")
    c = pd.read_parquet(files[-1])
    for col in ("greeks_at", "last_trade_at"):
        c[col] = pd.to_datetime(c[col], utc=True)
    c["expiry"] = pd.to_datetime(c["expiry"])
    # keep options whose last trade is the moment the vendor computed its numbers, so price and SPY match
    c = c[c["iv"].notna() & (c["greeks_at"] == c["last_trade_at"])].copy()
    c["t"] = [days.years_to_expiry(a, e) for a, e in zip(c["greeks_at"], c["expiry"])]
    c["t_16h"] = (c["expiry"].dt.tz_localize(days.NY) + pd.Timedelta(hours=16) - c["greeks_at"]).dt.total_seconds() / (365 * 86400)
    rate = float(pd.read_parquet(sorted(config.GREEKS_DIR.glob("*.parquet"))[-1])["rate"].median())
    return c, rate


def vix_style(otm: pd.DataFrame, expiry, spot: float, now: pd.Timestamp, r: float) -> tuple[float, float, float]:
    """CBOE VIX formula on one expiry: variance = 2/T sum(dK/K^2 e^rT Q(K)) - (F/K0 - 1)^2 / T, with Q the
    out-of-the-money option price at each $1 strike. Prices come from my implied volatilities (interpolated
    across strikes, flat beyond the last traded one) repriced at one common SPY price, because the last
    trades happened at different moments of the day."""
    x = otm[otm["expiry"] == pd.Timestamp(expiry)].sort_values("m")
    t = days.years_to_expiry(now, expiry)
    k = np.arange(np.floor(spot * (1 + x["m"].min())), np.ceil(spot * (1 + x["m"].max())) + 1)
    iv = np.interp(k / spot - 1, x["m"], x["my_iv"])
    f = spot * np.exp(r * t)
    k0 = k[k <= f].max()
    put, call = bs.price(spot, k, t, r, iv, "P"), bs.price(spot, k, t, r, iv, "C")
    q = np.where(k < k0, put, np.where(k > k0, call, 0.5 * (put + call)))
    var = 2 / t * np.sum(np.gradient(k) / k ** 2 * np.exp(r * t) * q) - (f / k0 - 1) ** 2 / t
    return t, var, float(np.interp(0.0, x["m"], x["my_iv"]))


def main() -> None:
    # 1. minute by minute
    v = vendor_minutes()
    v["my_iv"] = bs.implied_vol(v["option_price"], v["underlying_price"], v["strike"], v["years_to_expiry"], v["rate"],
                                v["right"], v["dividends_pv"])
    g = bs.greeks(v["underlying_price"], v["strike"], v["years_to_expiry"], v["rate"], v["my_iv"], v["right"], v["dividends_pv"])
    gap = 100 * (v["my_iv"] - v["iv"]).abs()
    print(f"1. Minute by minute: {len(v):,} minutes, {v['contract'].nunique()} contracts (at-the-money call and put,"
          f" {v['contract'].str[3:9].nunique()} release days)")
    print(f"   implied volatility, mine vs vendor: median gap {gap.median():.1e} vol points, 99% within {gap.quantile(0.99):.1e},"
          f" largest {gap.max():.1e}; {int(v['my_iv'].isna().sum())} prices with no solution")
    for k in ("delta", "gamma", "vega", "theta"):
        rel = ((g[k] - v[k]).abs() / v[k].abs()).replace(np.inf, np.nan)
        print(f"   {k:6s} median relative gap {rel.median():.1e}, 99% within {rel.quantile(0.99):.1e}")
    print("   -> the vendor uses exactly this model: European Black-Scholes, calendar time / 365, a constant rate"
          f" ({100 * v['rate'].min():.2f}-{100 * v['rate'].max():.2f}%), no dividends in these files")

    # 2. the chain
    c, rate = chain()
    c["my_iv"] = bs.implied_vol(c["last_price"], c["greeks_underlying_price"], c["strike"], c["t"], rate, c["contract_type"])
    c["my_iv_16h"] = bs.implied_vol(c["last_price"], c["greeks_underlying_price"], c["strike"], c["t_16h"], rate, c["contract_type"])
    gap = 100 * (c["my_iv"] - c["iv"]).abs()
    print(f"\n2. Chain snapshot: {len(c):,} options traded at the vendor's timestamp, {c['expiry'].nunique()} expiries"
          f" ({c['dte'].min()}-{c['dte'].max()} days), rate {100 * rate:.2f}%")
    print(f"   median gap {gap.median():.4f} vol points (the vendor rounds to 0.01 points), {100 * (gap < 0.01).mean():.1f}% within 0.01")
    half = c["expiry"].dt.date.map(days.close_hour) == 13
    if half.any():
        g16 = 100 * (c.loc[half, "my_iv_16h"] - c.loc[half, "iv"])
        print(f"   {', '.join(sorted({f'{e:%d %b}' for e in c.loc[half, 'expiry']}))} closes at 13:00: with a 16:00 clock my volatility"
              f" would be off by {g16.median():+.3f} points ({int(half.sum())} options) - the vendor gets the half day right, and so must I")
    c[["ticker", "expiry", "dte", "strike", "contract_type", "last_price", "greeks_underlying_price", "t", "iv", "my_iv"]] \
        .to_csv(config.RESULTS / "06_chain_iv.csv", index=False)

    # 3. American vs European
    print("\n3. Early exercise: American minus European put on the same binomial tree (800 steps), $ per share")
    vol = 0.16
    rows = []
    for label, t in (("1 day", 1 / 365), ("9 days", 9 / 365), ("44 days", 44 / 365)):
        for m in (-0.05, -0.02, 0.0, 0.02, 0.05):                       # strike / spot - 1: put in the money when > 0
            s, k = 750.0, 750.0 * (1 + m)
            prem = bs.american_price(s, k, t, rate, vol, "P", 800) - bs.american_price(s, k, t, rate, vol, "P", 800, early=False)
            rows.append({"maturity": label, "t": t, "moneyness_pct": 100 * m, "put_premium": prem})
    e = pd.DataFrame(rows)
    e.to_csv(config.RESULTS / "06_early_exercise.csv", index=False)
    for label, x in e.groupby("maturity", sort=False):
        print(f"   put, {label:7s} " + "  ".join(f"K {r['moneyness_pct']:+.0f}%: {r['put_premium']:.4f}" for _, r in x.iterrows()))
    atm1 = e.loc[(e["maturity"] == "1 day") & (e["moneyness_pct"] == 0), "put_premium"].iloc[0]
    print(f"   calls: worth nothing extra without a dividend before expiry. For this study's straddles (at the money,"
          f" 1 day) it is {100 * atm1:.2f} cents a share, under 0.1% of a ~$4.40 straddle: the European formula is fine."
          f" Deep in-the-money puts gain about a day of interest on the strike by exercising now.")

    # 4. smile and VIX
    c["m"] = c["strike"] / c["greeks_underlying_price"] - 1
    otm = c[(c["last_price"] >= MIN_PRICE) & c["my_iv"].notna()
            & (((c["contract_type"] == "put") & (c["m"] < 0)) | ((c["contract_type"] == "call") & (c["m"] >= 0)))]
    last = c[c["greeks_at"] <= c["greeks_at"].dt.normalize().max() + pd.Timedelta(hours=21)]
    now = last["greeks_at"].max()
    spot = float(last.loc[last["greeks_at"] == now, "greeks_underlying_price"].iloc[0])
    eve = now.tz_convert(days.NY).normalize().tz_localize(None)
    print(f"\n4. SPY {spot:.2f} at {now.tz_convert(days.NY):%d %b %H:%M} New York; VIX-style volatility vs at-the-money, by expiry")
    rows, skipped = [], []
    for ex in sorted(otm["expiry"].unique()):
        x = otm[otm["expiry"] == ex].sort_values("m")
        sd = np.interp(0.0, x["m"], x["my_iv"]) * np.sqrt(x["t"].mean())
        if pd.Timestamp(ex) in pd.to_datetime(EX_DIVIDEND) or x["m"].min() / sd > WINGS[0] or x["m"].max() / sd < WINGS[1]:
            skipped.append(f"{pd.Timestamp(ex):%d %b}")
            continue
        t, var, atm = vix_style(otm, ex, spot, now, rate)
        rows.append({"expiry": ex, "days": 365 * t, "options": len(x), "vix_style": 100 * np.sqrt(var), "atm": 100 * atm})
    ts = pd.DataFrame(rows)
    ts.to_csv(config.RESULTS / "06_term_structure.csv", index=False)
    for _, r in ts.iterrows():
        print(f"   {r['expiry']:%d %b} ({r['days']:5.1f} days, {r['options']:3d} options): VIX-style {r['vix_style']:5.2f}"
              f"   at the money {r['atm']:5.2f}   gap {r['vix_style'] - r['atm']:+.2f}")
    print(f"   left out: {', '.join(skipped)} (traded strikes too close to SPY for the formula, or ex-dividend)")
    t1, v1, _ = vix_style(otm, VIX_PAIR[0], spot, now, rate)
    t2, v2, _ = vix_style(otm, VIX_PAIR[1], spot, now, rate)
    n30 = 30 / 365
    vix30 = 100 * np.sqrt((t1 * v1 * (t2 - n30) / (t2 - t1) + t2 * v2 * (n30 - t1) / (t2 - t1)) / n30)
    _, v9, _ = vix_style(otm, VIX9D_EXPIRY, spot, now, rate)
    official = {n: float(cboe.load(n).get(eve, np.nan)) for n in ("VIX9D", "VIX")}
    print(f"   rebuilt from SPY options: 30-day {vix30:.2f} vs official VIX {official['VIX']:.2f};"
          f" 9-day {100 * np.sqrt(v9):.2f} vs VIX9D {official['VIX9D']:.2f} (the VIX uses S&P 500 index options and quotes, not trades)")
    figure(otm, ts, e, official, vix30, eve)


def figure(otm, ts, e, official, vix30, eve) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    for ex, col in zip(SMILE_EXPIRIES, ("#d1495b", "#edae49", "#00798c", "#30638e")):
        x = otm[(otm["expiry"] == pd.Timestamp(ex)) & (otm["m"].abs() <= 0.10)].sort_values("m")
        if len(x):
            a1.plot(100 * x["m"], 100 * x["my_iv"], color=col, linewidth=1.6, marker=".", markersize=4,
                    label=f"{pd.Timestamp(ex):%d %b} ({int(x['dte'].iloc[0]) + 1} days)")
    a1.axvline(0, color="#555", linewidth=0.8)
    a1.set_xlabel("strike vs SPY, %  (left: puts, right: calls)")
    a1.set_ylabel("implied volatility, %")
    a1.set_title(f"The smile, {eve:%d %b %Y} (my solver)")
    a1.legend(frameon=False)

    a2.plot(ts["days"], ts["atm"], color="#9aa5b1", marker="o", linewidth=1.6, label="at the money (SPY)")
    a2.plot(ts["days"], ts["vix_style"], color="#30638e", marker="o", linewidth=1.6, label="VIX formula on SPY options")
    a2.scatter([9, 30], [official["VIX9D"], official["VIX"]], color="#d1495b", zorder=3, s=40, label="official VIX9D, VIX")
    a2.scatter([30], [vix30], color="#30638e", marker="x", s=60, zorder=3, label=f"rebuilt 30-day: {vix30:.1f}")
    a2.set_xlabel("days to expiry")
    a2.set_ylabel("volatility, %")
    a2.set_title("Term structure: the VIX sits above at-the-money vol")
    a2.legend(frameon=False, fontsize=8.5)

    for (label, x), col in zip(e.groupby("maturity", sort=False), ("#edae49", "#00798c", "#30638e")):
        a3.plot(x["moneyness_pct"], 100 * x["put_premium"], color=col, marker="o", linewidth=1.6, label=label)
    a3.set_xlabel("strike vs SPY, %  (right: put in the money)")
    a3.set_ylabel("early-exercise value, cents per share")
    a3.set_title("American minus European put (vol 16%)")
    a3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "06_black_scholes.png", dpi=150)
    print("\nsaved figure -> figures/06_black_scholes.png")


if __name__ == "__main__":
    main()
