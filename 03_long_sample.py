"""
STEP 3 - The long sample: do options overprice macro news? (May 2022 - now)

Every trading day t, the evening before (t-1), VIX1D says how volatile the S&P 500
is expected to be on day t. We compare that with what happened on day t, and ask
whether release days (CPI, payrolls, FOMC) differ from normal days.

Three ways to compare, from simplest to most realistic:
  1. Absolute move: actual |return| / implied |return|, where the implied absolute
     move is VIX1D / sqrt(252) x 0.80 (0.80 = average |move| of a normal
     distribution per unit of standard deviation). Below 1 = options overpriced.
  2. Variance: actual return^2 / implied variance (VIX1D / sqrt(252))^2. This is the
     "variance risk premium"; it needs no normality assumption, but a few huge days
     weigh a lot.
  3. A straddle seller's profit, in % of the index: the straddle priced as step 2
     measured it (0.94 x the VIX1D implied absolute move, the "bridge"), minus what it
     pays at expiry, |return|. Before trading costs (step 4 adds them).
Confidence intervals: bootstrap over days (95%). Release days vs normal days: the
difference with its own interval. Also year by year, and the worst days.

Run:  .venv\\Scripts\\python 03_long_sample.py
Writes results/03_*.csv and figures/03_long_sample.png.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cboe, config, days, fred, stats

BRIDGE = 0.94                 # straddle price / VIX1D implied absolute move, measured in step 2 (Oct 2025 - Nov 2026)
GROUPS = ["CPI", "Payrolls", "FOMC", "normal"]
COLOR = {"CPI": "#d1495b", "Payrolls": "#edae49", "FOMC": "#00798c", "normal": "#9aa5b1"}


def build() -> pd.DataFrame:
    vix = cboe.load("VIX1D")
    spx = fred.load()["SP500"].dropna()
    t = vix.index.intersection(spx.index)
    d = pd.DataFrame({"vix1d_eve": vix.reindex(t).shift(1),
                      "ret_pct": 100 * np.log(spx.reindex(t)).diff()}, index=t).dropna()
    gap = pd.Series(t, index=t).diff().dt.days.reindex(d.index)
    d = d[gap <= 4]                                     # skip any day whose previous close is missing (data gaps)
    d["kind"] = days.kinds(d.index)
    d["implied_sd_pct"] = stats.one_day_sd(d["vix1d_eve"])
    d["implied_abs_pct"] = d["implied_sd_pct"] * stats.ABS_OVER_SD
    d["actual_abs_pct"] = d["ret_pct"].abs()
    d["straddle_pct"] = BRIDGE * d["implied_abs_pct"]
    d["seller_pnl_pct"] = d["straddle_pct"] - d["actual_abs_pct"]
    d["year"] = d.index.year
    return d


def summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        x = d[d["kind"] == g]
        ab = stats.bootstrap_ratio(x["actual_abs_pct"], x["implied_abs_pct"])
        va = stats.bootstrap_ratio(x["ret_pct"] ** 2, x["implied_sd_pct"] ** 2)
        pnl = stats.bootstrap(x["seller_pnl_pct"])
        rows.append({"kind": g, "days": len(x), "implied_abs_pct": x["implied_abs_pct"].mean(),
                     "actual_abs_pct": x["actual_abs_pct"].mean(),
                     "abs_ratio": ab[0], "abs_low": ab[1], "abs_high": ab[2],
                     "var_ratio": va[0], "var_low": va[1], "var_high": va[2],
                     "seller_pnl_pct": pnl[0], "pnl_low": pnl[1], "pnl_high": pnl[2],
                     "seller_wins_pct": 100 * (x["seller_pnl_pct"] > 0).mean(),
                     "worst_day_pct": x["seller_pnl_pct"].min()})
    return pd.DataFrame(rows)


def main() -> None:
    d = build()
    d.to_csv(config.RESULTS / "03_days.csv")
    s = summary(d)
    s.to_csv(config.RESULTS / "03_by_kind.csv", index=False)
    print(f"Days: {len(d):,} ({d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}); "
          + ", ".join(f"{g} {int((d['kind'] == g).sum())}" for g in GROUPS))

    print("\n1-2. Actual vs implied, by kind of day (95% bootstrap intervals; below 1 = options charged too much)")
    print(f"   {'':9s}{'days':>5s}{'implied |move|':>16s}{'actual |move|':>15s}   {'|move| ratio':>22s}   {'variance ratio':>22s}")
    for _, r in s.iterrows():
        print(f"   {r['kind']:9s}{r['days']:5.0f}{r['implied_abs_pct']:15.2f}%{r['actual_abs_pct']:14.2f}%"
              f"   {r['abs_ratio']:6.2f} [{r['abs_low']:.2f}, {r['abs_high']:.2f}]"
              f"   {r['var_ratio']:6.2f} [{r['var_low']:.2f}, {r['var_high']:.2f}]")

    print(f"\n3. Straddle seller, before costs (straddle = {BRIDGE} x implied |move|), % of the index per trade")
    for _, r in s.iterrows():
        print(f"   {r['kind']:9s} mean {r['seller_pnl_pct']:+.3f}% [{r['pnl_low']:+.3f}, {r['pnl_high']:+.3f}]"
              f"   seller wins {r['seller_wins_pct']:.0f}% of days   worst day {r['worst_day_pct']:+.2f}%")
    normal = d.loc[d["kind"] == "normal", "seller_pnl_pct"]
    rel = d["kind"] != "normal"
    for g in ["all releases"] + GROUPS[:3]:
        x = d.loc[rel if g == "all releases" else d["kind"] == g, "seller_pnl_pct"]
        diff = stats.bootstrap_diff(x, normal)
        print(f"   {g:13s} minus normal: {diff[0]:+.3f}% [{diff[1]:+.3f}, {diff[2]:+.3f}]")
    prem = d.loc[rel, "implied_abs_pct"].mean() / d.loc[~rel, "implied_abs_pct"].mean()
    move = d.loc[rel, "actual_abs_pct"].mean() / d.loc[~rel, "actual_abs_pct"].mean()
    print(f"   on release days options charge {100 * (prem - 1):.0f}% more than on normal days;"
          f" the S&P actually moves {100 * (move - 1):.0f}% more")

    by_year = (d.assign(group=np.where(rel, "releases", "normal"))
                .groupby(["year", "group"])
                .agg(days=("seller_pnl_pct", "size"), seller_pnl_pct=("seller_pnl_pct", "mean"),
                     abs_ratio=("actual_abs_pct", "sum"), implied=("implied_abs_pct", "sum")))
    by_year["abs_ratio"] = by_year["abs_ratio"] / by_year.pop("implied")
    by_year.to_csv(config.RESULTS / "03_by_year.csv")
    print("\n   By year (seller P&L per trade, % | actual / implied |move|):")
    for y in sorted(d["year"].unique()):
        rr = by_year.loc[y]
        print(f"   {y}: releases {rr.loc['releases', 'seller_pnl_pct']:+.3f}% | {rr.loc['releases', 'abs_ratio']:.2f}"
              f" ({int(rr.loc['releases', 'days'])} days)    normal {rr.loc['normal', 'seller_pnl_pct']:+.3f}% |"
              f" {rr.loc['normal', 'abs_ratio']:.2f} ({int(rr.loc['normal', 'days'])} days)")

    print("\n   Worst release days for a seller:")
    for t, r in d[rel].nsmallest(5, "seller_pnl_pct").iterrows():
        print(f"   {t:%Y-%m-%d} {r['kind']:9s} S&P {r['ret_pct']:+.2f}% vs implied |move| {r['implied_abs_pct']:.2f}%"
              f" -> seller {r['seller_pnl_pct']:+.2f}%")
    figure(d, s)


def figure(d: pd.DataFrame, s: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(s))
    a1.bar(x, s["abs_ratio"], color=[COLOR[k] for k in s["kind"]], width=0.6)
    a1.errorbar(x, s["abs_ratio"], yerr=[s["abs_ratio"] - s["abs_low"], s["abs_high"] - s["abs_ratio"]],
                fmt="none", color="#333", capsize=4)
    a1.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    a1.set_xticks(x, [f"{k}\n({n})" for k, n in zip(s["kind"], s["days"])])
    a1.set_ylabel("actual |move| / implied |move|  (95% CI)")
    a1.set_title("Below 1: the S&P moved less than options priced")
    for g in GROUPS:
        x_ = d.loc[d["kind"] == g, "seller_pnl_pct"]
        a2.plot(x_.index, x_.cumsum(), color=COLOR[g], linewidth=1.6 if g != "normal" else 1.0,
                label=f"{g} ({len(x_)} trades)")
    a2.axhline(0, color="#555", linewidth=0.8)
    a2.set_ylabel("cumulative seller profit, % of the index")
    a2.set_title("Selling the next-day straddle every evening, before costs")
    a2.legend(loc="upper left", frameon=False)
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_long_sample.png", dpi=150)
    print("\nsaved figure -> figures/03_long_sample.png")


if __name__ == "__main__":
    main()
