"""
STEP 7 - When does the event risk leave the price? Implied variance minute by minute.

An option that expires at the close of day E prices how much SPY can still move until
then. That amount is W = (implied volatility)^2 x (time to expiry, in years): the
variance of SPY's log return until the close, as the market sees it. W falls as the
day goes on, because there is less time left, and falls faster when a risk is removed,
for example by an announcement. I compute W myself from the 1-minute candles of the
at-the-money call and put (the strike nearest SPY on the eve), with my own solver
(step 6), for every expiry day from October 2025: release days and normal days alike.
Each minute uses the out-of-the-money option of the pair, whose whole price is time
value (the in-the-money one is mostly intrinsic value plus noise).

  1. Check: my W against the vendor's minute Greeks on the release days.
  2. Overnight. CPI and payrolls come out at 08:30, before the open. How much of the
     eve's W is still there at 09:35, on release days against normal days? The
     difference is the variance the options priced for the release itself. Then: did
     the overnight move (eve close to 09:35) match what was priced for it?
  3. The FOMC afternoon (statement 14:00, press conference 14:30). W relative to 13:55,
     minute by minute, on FOMC days against other days; and the move from 13:55 to
     15:45 against what was priced for it.

Only one-night eves (no weekend or holiday between the eve and E), so every day is
compared on the same clock. Half days are left out.

Run after 02:  .venv\\Scripts\\python 07_intraday.py
Writes results/07_*.csv and figures/07_intraday.png.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import bs, config, days, fred, options, spy, stats

GROUPS = ["CPI", "Payrolls", "FOMC", "normal"]
COLOR = {"CPI": "#d1495b", "Payrolls": "#edae49", "FOMC": "#00798c", "normal": "#9aa5b1"}
ABS_OVER_SD = np.sqrt(2 / np.pi)
STALE_MIN = 5                              # use an option's last trade only if it is at most 5 minutes old
OPEN, PRE, POST = "09:35", "13:55", "15:45"


def spy_minutes() -> pd.Series:
    files = [f for f in sorted(config.DATA_LSE.glob("SPY_*.parquet")) if int(f.stem[-4:]) >= 2025]
    d = spy.regular_hours(pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files]))
    return d.set_index("ny")["close"].sort_index()


def leg(expiry, right: str, k: float, grid: pd.DatetimeIndex) -> pd.Series:
    """Last traded price of one option at each minute of `grid` (bar starts, New York), same day, at most STALE_MIN old."""
    path = options.candle_path(expiry, right, k)
    d = options.load_candles(expiry, right, k) if path.exists() else pd.DataFrame()
    if not len(d):
        return pd.Series(np.nan, index=grid)
    d = d.sort_values("ny")[["ny", "close"]]
    g = pd.DataFrame({"ny": grid})
    d["ny"] = d["ny"].dt.tz_convert(config.TZ).astype(g["ny"].dtype)
    m = pd.merge_asof(g, d, on="ny", direction="backward", tolerance=pd.Timedelta(minutes=STALE_MIN))
    return pd.Series(m["close"].to_numpy(), index=grid)


def day_path(eve, expiry, k: float, s_min: pd.Series, rate: float) -> pd.DataFrame:
    """W at the eve's last minute and at every minute of the expiry day."""
    e_ny = pd.Timestamp(expiry).tz_localize(config.TZ)
    v_ny = pd.Timestamp(eve).tz_localize(config.TZ)
    grid = s_min.index[(s_min.index.normalize() == e_ny) | (s_min.index == v_ny + pd.Timedelta(hours=15, minutes=59))]
    s = s_min.loc[grid].to_numpy()
    t = np.array([days.years_to_expiry(g + pd.Timedelta(minutes=1), expiry) for g in grid])
    c, p = leg(expiry, "C", k, grid).to_numpy(), leg(expiry, "P", k, grid).to_numpy()
    w_c = bs.implied_vol(c, s, k, t, rate, "C") ** 2 * t
    w_p = bs.implied_vol(p, s, k, t, rate, "P") ** 2 * t
    otm_is_call = s <= k
    w = np.where(otm_is_call, np.where(np.isnan(w_c), w_p, w_c), np.where(np.isnan(w_p), w_c, w_p))
    return pd.DataFrame({"s": s, "t": t, "w": w, "w_call": w_c, "w_put": w_p}, index=grid)


def at(path: pd.DataFrame, day, hm: str) -> pd.Series:
    """The row at New York time `hm` on `day` (bar start), or NaNs."""
    key = pd.Timestamp(f"{pd.Timestamp(day).date()} {hm}", tz=config.TZ)
    return path.loc[key] if key in path.index else pd.Series({c: np.nan for c in path.columns})


def build() -> tuple[pd.DataFrame, dict]:
    closes = spy.daily_close()
    s_min = spy_minutes()
    tbill = fred.load()["DTB3"].ffill() / 100
    sched = options.schedule(closes.index, config.OPTIONS_START,
                             end=pd.Timestamp.now(tz=config.TZ).tz_localize(None).normalize() - pd.Timedelta(days=1))
    sched = sched[(sched["expiry"] - sched["eve"]).dt.days == 1]
    sched = sched[[days.close_hour(e) == 16 and days.close_hour(v) == 16 for e, v in zip(sched["expiry"], sched["eve"])]]
    kinds = days.kinds(sched["expiry"])
    rows, paths = [], {}
    for _, r in sched.iterrows():
        eve, e = r["eve"], r["expiry"]
        k = options.strikes_for(closes[eve])[len(config.STRIKE_OFFSETS) // 2]
        path = day_path(eve, e, k, s_min, float(tbill.asof(eve)))
        if path["w"].notna().sum() < 100:
            continue
        paths[e] = path
        row = {"expiry": e, "kind": kinds[e], "strike": k}
        for lab, day, hm in (("eve", eve, "15:59"), ("open", e, OPEN), ("pre", e, PRE), ("post", e, POST)):
            x = at(path, day, hm)
            row[f"w_{lab}"], row[f"s_{lab}"] = x["w"], x["s"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("expiry"), paths


def check_vendor(paths: dict) -> pd.DataFrame:
    out = []
    for f in sorted(config.GREEKS_DIR.glob("*.parquet")):
        g = pd.read_parquet(f)
        e = pd.Timestamp("20" + f.stem[3:9])
        if not len(g) or e not in paths:
            continue
        g = g[g["iv"].notna() & (g["years_to_expiry"] > 0)]
        g = g.assign(ny=pd.to_datetime(g["minute"], utc=True).dt.tz_convert(config.TZ),
                     w_vendor=g["iv"] ** 2 * g["years_to_expiry"]).set_index("ny")
        call = f.stem[-9] == "C"
        g = g[(g["underlying_price"] < int(f.stem[-8:]) / 1000) == call]          # the out-of-the-money leg only
        col = "w_call" if call else "w_put"
        mine = paths[e][[col]].rename(columns={col: "w_mine"})
        g.index, mine.index = g.index.as_unit("ns"), mine.index.as_unit("ns")
        j = g[["w_vendor"]].join(mine, how="inner").dropna()
        out.append(j.assign(contract=f.stem))
    return pd.concat(out) if out else pd.DataFrame(columns=["w_vendor", "w_mine"])


def main() -> None:
    d, paths = build()
    d.to_csv(config.RESULTS / "07_days.csv")
    print(f"Expiry days with a one-night eve: {len(d)} ({', '.join(f'{g} {(d.kind == g).sum()}' for g in GROUPS)})")

    v = check_vendor(paths)
    rel = (v["w_mine"] / v["w_vendor"] - 1).abs()
    print(f"\n1. My W vs the vendor's, same out-of-the-money option and minute: {len(v):,} minutes, median gap"
          f" {100 * rel.median():.2f}%, 90% within {100 * rel.quantile(0.9):.1f}% (the vendor uses SPY's price at the"
          f" trade's second, I use the minute's close)")

    # 2. overnight
    d["left_open"] = d["w_open"] / d["w_eve"]
    d["move_ovn"] = 100 * np.abs(np.log(d["s_open"] / d["s_eve"]))
    left = d.groupby("kind")["left_open"].median()
    # the overnight share of the eve's priced variance, per kind of day (a median: one day's W at the open also
    # carries that morning's news about the rest of the day)
    d["implied_ovn"] = 100 * ABS_OVER_SD * np.sqrt((1 - d["kind"].map(left)) * d["w_eve"])
    print(f"\n2. Overnight: median share of the eve's W left at {OPEN}; the eve close -> {OPEN} move vs what was priced for it")
    rows = [segment(d, "overnight", "left_open", "implied_ovn", "move_ovn", left)]
    for g in ("CPI", "Payrolls"):
        x = d[d["kind"] == g]
        extra = left["normal"] - left[g]
        print(f"   {g}: the release took out {100 * extra:+.0f} points of the eve's W more than a normal night, a"
              f" {100 * ABS_OVER_SD * np.sqrt(max(extra, 0) * x['w_eve']).mean():.2f}% move priced for the release itself")

    # 3. FOMC afternoon
    d["left_post"] = d["w_post"] / d["w_pre"]
    d["move_aft"] = 100 * np.abs(np.log(d["s_post"] / d["s_pre"]))
    left_aft = d.groupby("kind")["left_post"].median()
    d["implied_aft"] = 100 * ABS_OVER_SD * np.sqrt((1 - d["kind"].map(left_aft)) * d["w_pre"])
    print(f"\n3. Afternoon: median share of the {PRE} W left at {POST}; the {PRE} -> {POST} move vs what was priced for it")
    rows.append(segment(d, "afternoon", "left_post", "implied_aft", "move_aft", left_aft))
    pd.concat(rows).to_csv(config.RESULTS / "07_segments.csv", index=False)
    prof = profiles(d, paths)
    prof.to_csv(config.RESULTS / "07_afternoon_profile.csv")
    f = prof["FOMC"]
    print(f"   FOMC days, W relative to {PRE} (median): 14:05 {f.get('14:05', np.nan):.2f}, 14:25 {f.get('14:25', np.nan):.2f},"
          f" 14:35 {f.get('14:35', np.nan):.2f}, 15:00 {f.get('15:00', np.nan):.2f}, 15:30 {f.get('15:30', np.nan):.2f};"
          f" normal days at 15:00 {prof['normal'].get('15:00', np.nan):.2f}")
    figure(d, prof, pd.concat(rows))


def segment(d: pd.DataFrame, name: str, left_col: str, implied: str, actual: str, left: pd.Series) -> pd.DataFrame:
    print(f"   {'':9s}{'days':>5s}{'W left':>8s}{'priced |move|':>15s}{'actual |move|':>15s}{'actual / priced [95% CI]':>27s}")
    rows = []
    for g in GROUPS:
        x = d[d["kind"] == g].dropna(subset=[implied, actual])
        r, lo, hi = stats.bootstrap_ratio(x[actual], x[implied])
        rows.append({"segment": name, "kind": g, "days": len(x), "w_left": left[g], "priced_abs_pct": x[implied].mean(),
                     "actual_abs_pct": x[actual].mean(), "ratio": r, "low": lo, "high": hi})
        print(f"   {g:9s}{len(x):5d}{left[g]:8.2f}{x[implied].mean():14.2f}%{x[actual].mean():14.2f}%"
              f"{r:12.2f} [{lo:.2f}, {hi:.2f}]")
    return pd.DataFrame(rows)


def profiles(d: pd.DataFrame, paths: dict) -> pd.DataFrame:
    """Median over days of W(t) / W(13:55), minute by minute from 13:00 to 15:58, per kind of day."""
    out = {}
    for g in GROUPS:
        cols = []
        for e in d.index[d["kind"] == g]:
            p = paths[e]
            p = p[p.index.normalize() == pd.Timestamp(e).tz_localize(config.TZ)]
            ref = at(p, e, PRE)["w"]
            if not np.isfinite(ref) or ref <= 0:
                continue
            s = (p["w"] / ref).rename(e)
            s.index = s.index.strftime("%H:%M")
            cols.append(s)
        out[g] = pd.concat(cols, axis=1).median(axis=1) if cols else pd.Series(dtype=float)
    prof = pd.DataFrame(out)
    return prof[(prof.index >= "13:00") & (prof.index <= "15:58")]


def figure(d: pd.DataFrame, prof: pd.DataFrame, seg: pd.DataFrame) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(GROUPS))
    left = [d.loc[d["kind"] == g, "left_open"].median() for g in GROUPS]
    a1.bar(x, left, color=[COLOR[g] for g in GROUPS])
    for i, val in enumerate(left):
        a1.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
    a1.set_xticks(x, [f"{g}\n({(d['kind'] == g).sum()})" for g in GROUPS])
    a1.set_ylim(0, 1)
    a1.set_ylabel(f"share of the eve's W left at {OPEN}")
    a1.set_title("CPI nights remove extra risk; payrolls nights don't")

    minutes = pd.to_datetime(prof.index, format="%H:%M")
    for g in GROUPS:
        a2.plot(minutes, prof[g], color=COLOR[g], linewidth=2.2 if g == "FOMC" else 1.3, label=g)
    for hm, lab in (("14:00", "statement"), ("14:30", "press conference")):
        tt = pd.to_datetime(hm, format="%H:%M")
        a2.axvline(tt, color="#555", linestyle="--", linewidth=0.8)
        a2.text(tt, 1.02, f" {lab}", fontsize=8, va="bottom")
    a2.set_ylabel(f"W(t) / W({PRE}), median over days")
    a2.set_title("Expiry-day afternoon: remaining priced variance")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a2.legend(frameon=False, loc="lower left")
    a2.set_ylim(0, 1.1)

    for i, (name, lab, alpha) in enumerate((("overnight", f"overnight (eve close -> {OPEN})", 1.0),
                                            ("afternoon", f"afternoon ({PRE} -> {POST})", 0.45))):
        r = seg[seg["segment"] == name].set_index("kind").loc[GROUPS]
        xs = x + (i - 0.5) * 0.38
        a3.bar(xs, r["ratio"], 0.38, color=[COLOR[g] for g in GROUPS], alpha=alpha, label=lab)
        a3.errorbar(xs, r["ratio"], yerr=[r["ratio"] - r["low"], r["high"] - r["ratio"]], fmt="none", color="#333", capsize=3)
    a3.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    a3.set_xticks(x, GROUPS)
    a3.set_ylabel("actual |move| / priced |move| (95% CI)")
    a3.set_title("Did the options price each part of the day right?")
    a3.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "07_intraday.png", dpi=150)
    print("\nsaved figure -> figures/07_intraday.png")


if __name__ == "__main__":
    main()
