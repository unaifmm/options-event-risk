"""
STEP 5 - The real trade: selling SPY's next-day straddle, with costs and tail risk.

Steps 3-4 used VIX1D over 4.5 years. For the year in which real option prices exist
(Oct 2025 - Nov 2026) we trade them: every evening at 15:59, sell the at-the-money SPY
call and put that expire the next day; at expiry, pay what the one that ends in the
money is worth, |SPY close - strike|.

  1. Proxy vs real: on the same days, does the VIX1D estimate used in steps 3-4 match
     the real straddle's profit? It is what lets us trust the 2022-2026 results.
  2. Profit by kind of day, before and after trading costs. The data has trades but no
     bid/ask quotes, so costs are scenarios, anchored on the data: in the last minute
     before expiry, options trade a median 1-2 cents above their intrinsic value
     (step 2), which is the price of trading immediately. Per option traded:
       professional   1 cent half-spread + $0.20 fees per contract (100 shares)
       retail         2 cents half-spread + $0.70 fees per contract
     A straddle trades 3 options: sell the call and the put on the eve, buy back the
     one that ends in the money before expiry (the other expires worthless).
  3. Strategies and tail risk: sell every day, skip release days, or only release
     days. Mean, Sharpe ratio, worst day, average of the worst 5% of trades, worst
     drawdown - for the real year and for the 2022-2026 VIX1D sample.
  4. Data flags: the result without the days whose put-call parity is off by more
     than $0.50 after interest and dividends (step 2), in case those prices are wrong.

Run after 02 and 03:  .venv\\Scripts\\python 05_straddles.py
Writes results/05_*.csv and figures/05_straddles.png.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, fred, stats

GROUPS = ["CPI", "Payrolls", "FOMC", "normal"]
COLOR = {"CPI": "#d1495b", "Payrolls": "#edae49", "FOMC": "#00798c", "normal": "#9aa5b1"}
COSTS = {"professional": 0.01 + 0.20 / 100, "retail": 0.02 + 0.70 / 100}   # $ per share, per option traded
LEGS = 3
PARITY_FLAG = 0.50                       # $ left over after interest and dividends
STRATEGIES = {"every day": lambda k: np.ones(len(k), bool),
              "skip release days": lambda k: (k == "normal").to_numpy(),
              "release days only": lambda k: (k != "normal").to_numpy()}


def real() -> pd.DataFrame:
    path = config.RESULTS / "02_daily.csv"
    if not path.exists():
        raise SystemExit("results/02_daily.csv not found - run 02_data_checks.py first")
    d = pd.read_csv(path, parse_dates=["eve", "expiry"])
    d = d[~d["half_day"] & d["call_eve"].notna() & d["put_eve"].notna() & d["spy_expiry"].notna()].copy()
    d["premium"] = d["call_eve"] + d["put_eve"]                            # $ per share received
    d["payout"] = (d["spy_expiry"] - d["strike"]).abs()                    # $ per share paid at expiry
    d["pnl_usd"] = d["premium"] - d["payout"]
    d["pnl_pct"] = 100 * d["pnl_usd"] / d["spy_eve"]
    for name, c in COSTS.items():
        d[f"pnl_pct_{name}"] = d["pnl_pct"] - 100 * LEGS * c / d["spy_eve"]
    return d.set_index("expiry")


def proxy() -> pd.DataFrame:
    path = config.RESULTS / "03_days.csv"
    if not path.exists():
        raise SystemExit("results/03_days.csv not found - run 03_long_sample.py first")
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    spy = fred.load()["SP500"].ffill().shift(1).reindex(d.index) / 10      # SPY is about the S&P 500 / 10
    d["pnl_pct"] = d["seller_pnl_pct"]
    for name, c in COSTS.items():
        d[f"pnl_pct_{name}"] = d["pnl_pct"] - 100 * LEGS * c / spy
    return d


def strategy_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for name, rule in STRATEGIES.items():
        on = rule(d["kind"])
        daily = np.where(on, d[col], 0.0)                                  # no trade = 0 that day
        x = d.loc[on, col]
        rows.append({"strategy": name, "trades": int(on.sum()), "mean_pct": x.mean(), "total_pct": x.sum(),
                     "win_pct": 100 * (x > 0).mean(), "sharpe": stats.sharpe(daily),
                     "worst_pct": x.min(), "worst_5pct_avg": stats.expected_shortfall(x, 0.05),
                     "max_drawdown_pct": stats.max_drawdown(daily)})
    return pd.DataFrame(rows)


def main() -> None:
    r, p = real(), proxy()
    r.to_csv(config.RESULTS / "05_real_trades.csv")
    print(f"Real straddles: {len(r)} trades, {r.index.min():%Y-%m-%d} to {r.index.max():%Y-%m-%d};"
          f" premium {r['premium'].median():.2f} $ a share ({r['premium'].median() * 100:.0f} $ a straddle),"
          f" {100 * (r['premium'] / r['spy_eve']).median():.2f}% of SPY (medians)")

    j = r[["kind", "pnl_pct"]].join(p[["pnl_pct"]].rename(columns={"pnl_pct": "proxy_pct"}), how="inner")
    print(f"\n1. Proxy vs real on the same {len(j)} days: correlation {j[['pnl_pct', 'proxy_pct']].corr().iloc[0, 1]:.3f},"
          f" typical gap {(j['pnl_pct'] - j['proxy_pct']).abs().median():.3f}% of the index")
    for g in GROUPS:
        x = j[j["kind"] == g]
        print(f"   {g:9s}{len(x):4d} days: real {x['pnl_pct'].mean():+.3f}%   VIX1D proxy {x['proxy_pct'].mean():+.3f}%")

    print("\n2. Seller profit per trade, % of SPY [95% CI before costs], by kind of day")
    print(f"   {'':9s}{'days':>5s}{'before costs':>27s}{'professional':>14s}{'retail':>9s}{'wins':>7s}"
          f"{'$ / straddle':>14s}")
    rows = []
    for g in GROUPS + ["all"]:
        x = r if g == "all" else r[r["kind"] == g]
        m, lo, hi = stats.bootstrap(x["pnl_pct"])
        usd = 100 * (x["pnl_usd"] - LEGS * COSTS["retail"]).mean()
        rows.append({"kind": g, "days": len(x), "pnl_pct": m, "low": lo, "high": hi,
                     "pnl_pct_professional": x["pnl_pct_professional"].mean(), "pnl_pct_retail": x["pnl_pct_retail"].mean(),
                     "win_pct": 100 * (x["pnl_pct_retail"] > 0).mean(), "usd_per_straddle_retail": usd})
        print(f"   {g:9s}{len(x):5d}   {m:+.3f}% [{lo:+.3f}, {hi:+.3f}]{x['pnl_pct_professional'].mean():+13.3f}%"
              f"{x['pnl_pct_retail'].mean():+8.3f}%{100 * (x['pnl_pct_retail'] > 0).mean():6.0f}%{usd:+12.0f} $")
    pd.DataFrame(rows).to_csv(config.RESULTS / "05_by_kind.csv", index=False)
    c = 100 * LEGS * pd.Series(COSTS) / r["spy_eve"].median()
    print(f"   costs per straddle: professional {c['professional']:.3f}%, retail {c['retail']:.3f}% of SPY"
          f" ({LEGS * COSTS['retail']:.3f} $ a share); $ / straddle = 100 shares, after retail costs")

    print("\n3. Strategies, after retail costs (Sharpe counts days without a trade as 0)")
    out = []
    for lab, d in (("Real straddles, Oct 2025 - Nov 2026", r), ("VIX1D proxy, May 2022 - Nov 2026", p)):
        t = strategy_table(d, "pnl_pct_retail")
        before = strategy_table(d, "pnl_pct").set_index("strategy")["sharpe"]
        out.append(t.assign(sample=lab))
        print(f"   {lab}")
        print(f"   {'':20s}{'trades':>7s}{'mean':>9s}{'total':>8s}{'wins':>6s}{'Sharpe':>8s}{'(no costs)':>11s}"
              f"{'worst':>8s}{'worst 5%':>10s}{'drawdown':>10s}")
        for _, s in t.iterrows():
            print(f"   {s['strategy']:20s}{s['trades']:7d}{s['mean_pct']:+8.3f}%{s['total_pct']:+7.1f}%{s['win_pct']:5.0f}%"
                  f"{s['sharpe']:8.2f}{before[s['strategy']]:11.2f}{s['worst_pct']:+7.2f}%{s['worst_5pct_avg']:+9.2f}%"
                  f"{s['max_drawdown_pct']:+9.2f}%")
    pd.concat(out).to_csv(config.RESULTS / "05_strategies.csv", index=False)
    prem = 100 * (r["premium"] / r["spy_eve"]).mean()
    print(f"   real year: the seller collects {prem:.2f}% a day on average; the worst day lost {-r['pnl_pct_retail'].min():.2f}%"
          f" ({-r['pnl_pct_retail'].min() / prem:.1f} days of premium), on {r['pnl_pct_retail'].idxmin():%Y-%m-%d}"
          f" ({r.loc[r['pnl_pct_retail'].idxmin(), 'kind']})")

    if "parity_left" in r.columns:
        bad = r["parity_left"].abs() > PARITY_FLAG
        print(f"\n4. Without the {int(bad.sum())} days whose parity is off by more than ${PARITY_FLAG:.2f}"
              f" ({', '.join(f'{t:%d %b %Y}' for t in r.index[bad])}):")
        print("   " + "   ".join(f"{g} {r.loc[~bad & (r['kind'] == g), 'pnl_pct'].mean():+.3f}%"
                                 f" (was {r.loc[r['kind'] == g, 'pnl_pct'].mean():+.3f}%)" for g in GROUPS))
    else:
        print("\n4. Parity flags: re-run 02_data_checks.py first (results/02_daily.csv has no parity_left column)")
    figure(r, j, p)


def figure(r: pd.DataFrame, j: pd.DataFrame, p: pd.DataFrame) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(GROUPS))
    for i, (col, lab, alpha) in enumerate((("pnl_pct", "before costs", 1.0), ("pnl_pct_retail", "after retail costs", 0.45))):
        m = [r.loc[r["kind"] == g, col].mean() for g in GROUPS]
        a1.bar(x + (i - 0.5) * 0.38, m, 0.38, color=[COLOR[g] for g in GROUPS], alpha=alpha, label=lab)
        if col == "pnl_pct":
            ci = [stats.bootstrap(r.loc[r["kind"] == g, col]) for g in GROUPS]
            a1.errorbar(x - 0.19, m, yerr=[[c[0] - c[1] for c in ci], [c[2] - c[0] for c in ci]], fmt="none",
                        color="#333", capsize=3)
    a1.axhline(0, color="#555", linewidth=0.8)
    a1.set_xticks(x, [f"{g}\n({(r['kind'] == g).sum()})" for g in GROUPS])
    a1.set_ylabel("seller profit per trade, % of SPY")
    a1.set_title("Real straddles, Oct 2025 - Nov 2026 (dark: no costs)")

    for g in GROUPS:
        y = j[j["kind"] == g]
        a2.scatter(y["proxy_pct"], y["pnl_pct"], s=14 if g == "normal" else 30, color=COLOR[g], alpha=0.8,
                   edgecolors="none", label=g)
    lim = [min(j["proxy_pct"].min(), j["pnl_pct"].min()) - 0.1, max(j["proxy_pct"].max(), j["pnl_pct"].max()) + 0.1]
    a2.plot(lim, lim, color="#555", linestyle="--", linewidth=0.8)
    a2.set_xlim(lim), a2.set_ylim(lim)
    a2.set_xlabel("VIX1D proxy (steps 3-4), % of the index")
    a2.set_ylabel("real straddle, % of SPY")
    a2.set_title("The proxy against the real trade, same days")
    a2.legend(frameon=False, loc="upper left")

    for name, style in (("every day", "-"), ("skip release days", "-"), ("release days only", "--")):
        on = STRATEGIES[name](p["kind"])
        a3.plot(p.index, np.cumsum(np.where(on, p["pnl_pct_retail"], 0.0)), linestyle=style, linewidth=1.4,
                color={"every day": "#555", "skip release days": "#9aa5b1", "release days only": "#d1495b"}[name], label=name)
    a3.axhline(0, color="#555", linewidth=0.8)
    a3.set_ylabel("cumulative profit, % of the index")
    a3.set_title("VIX1D proxy 2022 - 2026, after retail costs")
    a3.legend(frameon=False, loc="upper left")
    a3.xaxis.set_major_locator(mdates.YearLocator())
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05_straddles.png", dpi=150)
    print("\nsaved figure -> figures/05_straddles.png")


if __name__ == "__main__":
    main()
