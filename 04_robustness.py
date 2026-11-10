"""
STEP 4 - Is step 3's result real, or a handful of days?

Step 3 compared what options charged for the next day's move (VIX1D the evening
before) with the S&P 500's actual move, and found the premium sellers earn on
normal days shrinks or disappears on release days. With about 50 days per release
type, one or two huge days can drive an average. One of the worst payrolls days,
4 April 2025, was really the tariff crash: China announced 34% retaliatory tariffs
that morning. Dropping that day by hand because it hurts the result would be
cherry-picking, so every check below applies the same rule to every kind of day:

  1. Trim: drop the biggest x% of surprises (actual / implied move) in EACH group.
  2. Medians and ranks: the median day instead of the average, and a rank test
     (Mann-Whitney) in which no single day can carry the result.
  3. Leave one out: the result without the single most important day of each
     group, and without the tariff day.
  4. Two periods: 2022, the inflation shock (CPI peaked at 9.1% in June 2022),
     against 2023 - now. Also what options charged and how much the S&P moved on
     each release type relative to normal days, year by year.
  5. The bridge: the seller's profit if the straddle / VIX1D factor were at either
     end of step 2's range. Comparisons between groups do not depend on it.

Run after 03:  .venv\\Scripts\\python 04_robustness.py
Writes results/04_*.csv and figures/04_robustness.png.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from src import config, stats

GROUPS = ["CPI", "Payrolls", "FOMC", "normal"]
RELEASES = GROUPS[:3]
COLOR = {"CPI": "#d1495b", "Payrolls": "#edae49", "FOMC": "#00798c", "normal": "#9aa5b1", "releases": "#444"}
TRIMS = (0, 0.02, 0.05, 0.10)
TARIFF_DAY = pd.Timestamp("2025-04-04")    # payrolls day; China announced 34% retaliatory tariffs that morning
SPLIT_YEAR = 2023                          # 2022 = the inflation shock; 2023 on = after it
BRIDGES = (0.83, 0.94, 1.08)               # step 2: straddle / VIX1D implied move, 5th pct, median, 95th pct
NORMAL_MEDIAN = 0.6745 / 0.7979            # median / mean of |move| for a normal distribution: the "fair" median surprise


def load() -> pd.DataFrame:
    path = config.RESULTS / "03_days.csv"
    if not path.exists():
        raise SystemExit("results/03_days.csv not found - run 03_long_sample.py first")
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d["surprise"] = d["actual_abs_pct"] / d["implied_abs_pct"]
    return d


def pick(d: pd.DataFrame, g: str) -> pd.DataFrame:
    return d[d["kind"] != "normal"] if g == "releases" else d[d["kind"] == g]


def ratio(x: pd.DataFrame) -> float:
    return x["actual_abs_pct"].sum() / x["implied_abs_pct"].sum()


def trimming(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for share in TRIMS:
        for g in GROUPS + ["releases"]:
            x = stats.trim_top(pick(d, g), "surprise", share)
            rows.append({"trim_pct": 100 * share, "group": g, "days": len(x), "abs_ratio": ratio(x),
                         "seller_pnl_pct": x["seller_pnl_pct"].mean()})
    return pd.DataFrame(rows)


def periods(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, m in ((f"2022", d["year"] < SPLIT_YEAR), (f"{SPLIT_YEAR}-{d['year'].max()}", d["year"] >= SPLIT_YEAR)):
        for g in GROUPS + ["releases"]:
            x = pick(d[m], g)
            r, lo, hi = stats.bootstrap_ratio(x["actual_abs_pct"], x["implied_abs_pct"])
            rows.append({"period": lab, "group": g, "days": len(x), "abs_ratio": r, "abs_low": lo, "abs_high": hi,
                         "seller_pnl_pct": x["seller_pnl_pct"].mean()})
    return pd.DataFrame(rows)


def charge_vs_move(d: pd.DataFrame) -> pd.DataFrame:
    """Per year: average implied and actual move on each release type, as a multiple of normal days."""
    rows = []
    for y, x in d.groupby("year"):
        n = x[x["kind"] == "normal"]
        for g in RELEASES:
            e = x[x["kind"] == g]
            rows.append({"year": y, "group": g, "days": len(e),
                         "charge_x_normal": e["implied_abs_pct"].mean() / n["implied_abs_pct"].mean(),
                         "move_x_normal": e["actual_abs_pct"].mean() / n["actual_abs_pct"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    normal = pick(d, "normal")
    print(f"Days: {len(d):,} ({d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}). Surprise = actual |move| / implied |move|.")

    t = trimming(d)
    t.to_csv(config.RESULTS / "04_trim.csv", index=False)
    print("\n1. Drop the biggest x% of surprises in EVERY group: actual / implied |move| (seller profit per trade, %)")
    print("   " + " " * 6 + "".join(f"{g:>9s}{'':9s}" for g in GROUPS + ["releases"]))
    for share in TRIMS:
        r = t[t["trim_pct"] == 100 * share].set_index("group")
        print(f"   {100 * share:3.0f}%  " + "".join(f"{r.loc[g, 'abs_ratio']:9.2f} ({r.loc[g, 'seller_pnl_pct']:+.3f})"
                                             for g in GROUPS + ["releases"]))

    print(f"\n2. The median day: median surprise, 95% CI (if options were fair and moves normal: {NORMAL_MEDIAN:.2f})")
    print("   rank test (Mann-Whitney): are surprises on this kind of day bigger than on normal days?")
    for g in GROUPS + ["releases"]:
        x = pick(d, g)
        m, lo, hi = stats.bootstrap(x["surprise"], np.median)
        p = "" if g == "normal" else \
            f"   p = {sps.mannwhitneyu(x['surprise'], normal['surprise'], alternative='greater').pvalue:.3f}"
        print(f"   {g:9s}{len(x):5d} days: {m:.2f} [{lo:.2f}, {hi:.2f}]{p}")
    rel = pick(d, "releases")
    print(f"   medians, release vs normal days: options charge {100 * (rel['implied_abs_pct'].median() / normal['implied_abs_pct'].median() - 1):.0f}%"
          f" more, the S&P moves {100 * (rel['actual_abs_pct'].median() / normal['actual_abs_pct'].median() - 1):.0f}% more"
          f" (averages: {100 * (rel['implied_abs_pct'].mean() / normal['implied_abs_pct'].mean() - 1):.0f}% and"
          f" {100 * (rel['actual_abs_pct'].mean() / normal['actual_abs_pct'].mean() - 1):.0f}%)")

    print("\n3. Leave one out: the result without the single day that matters most")
    for g in GROUPS + ["releases"]:
        x = pick(d, g)
        out = pd.Series({t_: ratio(x.drop(t_)) for t_ in x.nlargest(10, "surprise").index})
        top = out.idxmin()
        print(f"   {g:9s} {ratio(x):.2f} -> {out[top]:.2f} without {top:%Y-%m-%d} (S&P {x.loc[top, 'ret_pct']:+.2f}%,"
              f" implied {x.loc[top, 'implied_abs_pct']:.2f}%)")
    pay = pick(d, "Payrolls")
    rank = int((d["surprise"] > d.loc[TARIFF_DAY, "surprise"]).sum()) + 1
    print(f"   tariff day {TARIFF_DAY:%Y-%m-%d}: surprise {d.loc[TARIFF_DAY, 'surprise']:.2f} (#{rank} of {len(d)} days);"
          f" payrolls {ratio(pay):.2f} -> {ratio(pay.drop(TARIFF_DAY)):.2f} without it,"
          f" seller {pay['seller_pnl_pct'].mean():+.3f}% -> {pay.drop(TARIFF_DAY)['seller_pnl_pct'].mean():+.3f}% per trade;"
          f" normal days {ratio(normal):.2f}")

    p = periods(d)
    p.to_csv(config.RESULTS / "04_periods.csv", index=False)
    print("\n4. Two periods: actual / implied |move| [95% CI] (seller profit per trade)")
    for lab, g_ in p.groupby("period", sort=False):
        print(f"   {lab}: " + "   ".join(f"{r['group']} {r['abs_ratio']:.2f} [{r['abs_low']:.2f}, {r['abs_high']:.2f}]"
                                         f" ({r['seller_pnl_pct']:+.2f}%, {r['days']})" for _, r in g_.iterrows()))
    c = charge_vs_move(d)
    c.to_csv(config.RESULTS / "04_charge_vs_move.csv", index=False)
    print("   Each year, release days vs normal days: options charge x times more | the S&P moves y times more")
    for y, g_ in c.groupby("year"):
        print(f"   {y}: " + "   ".join(f"{r['group']} x{r['charge_x_normal']:.2f} | x{r['move_x_normal']:.2f}"
                                       for _, r in g_.iterrows()))

    print("\n5. The bridge (straddle = factor x VIX1D implied move): seller profit per trade, %")
    print("   " + " " * 9 + "".join(f"{g:>10s}" for g in GROUPS + ["releases"]))
    for b in BRIDGES:
        print(f"   x {b:.2f}  " + "".join(f"{(b * pick(d, g)['implied_abs_pct'] - pick(d, g)['actual_abs_pct']).mean():+10.3f}"
                                     for g in GROUPS + ["releases"]))
    figure(d, p, c)


def figure(d: pd.DataFrame, p: pd.DataFrame, c: pd.DataFrame) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    shares = np.arange(0, 0.105, 0.005)
    for g in GROUPS:
        a1.plot(100 * shares, [ratio(stats.trim_top(pick(d, g), "surprise", s)) for s in shares], color=COLOR[g],
                linewidth=1.8, label=g)
    a1.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    a1.set_xlabel("biggest surprises removed from each group, %")
    a1.set_ylabel("actual |move| / implied |move|")
    a1.set_title("Same trim for every group")
    a1.legend(frameon=False)

    labs = list(p["period"].unique())
    w = 0.38
    x = np.arange(len(GROUPS))
    for i, lab in enumerate(labs):
        r = p[p["period"] == lab].set_index("group").loc[GROUPS]
        a2.bar(x + (i - 0.5) * w, r["abs_ratio"], w, color=[COLOR[g] for g in GROUPS], alpha=0.45 if i == 0 else 1,
               label=lab)
        a2.errorbar(x + (i - 0.5) * w, r["abs_ratio"], yerr=[r["abs_ratio"] - r["abs_low"], r["abs_high"] - r["abs_ratio"]],
                    fmt="none", color="#333", capsize=3)
    a2.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    a2.set_xticks(x, GROUPS)
    a2.set_ylabel("actual |move| / implied |move| (95% CI)")
    a2.set_title(f"{labs[0]} (light) vs {labs[1]} (dark)")

    for g in ("CPI", "Payrolls"):
        e = c[c["group"] == g]
        a3.plot(e["year"], e["charge_x_normal"], color=COLOR[g], linewidth=2, marker="o", label=f"{g}: options charge")
        a3.plot(e["year"], e["move_x_normal"], color=COLOR[g], linewidth=1.2, linestyle="--", marker="o",
                label=f"{g}: S&P moves")
    a3.axhline(1, color="#555", linewidth=0.8)
    a3.set_xticks(sorted(c["year"].unique()))
    a3.set_ylabel("x a normal day")
    a3.set_title("Release days vs normal days, by year")
    a3.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "04_robustness.png", dpi=150)
    print("\nsaved figure -> figures/04_robustness.png")


if __name__ == "__main__":
    main()
