"""AM-killzone-only analysis for both datasets.

Reuses already-generated trades.csv from the two backtest runs and filters
to AM trades only. No strategy code changes; the run state machine resets
per killzone so the AM-only subset is identical to running with only AM
enabled.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RISK_PER_TRADE = 100.0
START_EQUITY   = 10_000


def stats_from_r(r: np.ndarray) -> dict:
    pnl = r * RISK_PER_TRADE
    eq  = pnl.cumsum() + START_EQUITY
    peak = np.maximum.accumulate(eq)
    dd_abs = eq - peak
    dd_pct = dd_abs / peak * 100
    wins  = (r > 0).sum()
    losses = (r <= 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return {
        "trades":     int(len(r)),
        "wins":       int(wins),
        "losses":     int(losses),
        "win_rate_%": float(wins / len(r) * 100) if len(r) else 0.0,
        "expectancy_R": float(r.mean()) if len(r) else 0.0,
        "total_R":    float(r.sum()),
        "PF":         float(pf),
        "net_pnl_$":  float(pnl.sum()),
        "max_dd_$":   float(dd_abs.min()),
        "max_dd_%":   float(dd_pct.min()),
        "final_eq_$": float(eq[-1]) if len(eq) else START_EQUITY,
    }


def plot_curve(ax, times, r, title):
    pnl = r * RISK_PER_TRADE
    eq  = pnl.cumsum() + START_EQUITY
    peak = np.maximum.accumulate(eq)
    ax.plot(times, eq, color="tab:blue", linewidth=1.3)
    ax.fill_between(times, eq, peak, where=(eq < peak),
                    color="red", alpha=0.15)
    ax.axhline(START_EQUITY, color="grey", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)


def per_year(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty: return pd.DataFrame()
    t = tdf.copy()
    t["year"] = pd.to_datetime(t["entry_time"], utc=True).dt.year
    return t.groupby("year")["r"].agg([
        ("trades", "count"),
        ("mean_R", "mean"),
        ("sum_R",  "sum"),
        ("win_%",  lambda x: (x > 0).mean() * 100),
    ]).round(3)


def run(name: str, trades_csv: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    tdf = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
    am  = tdf[tdf["killzone"] == "am"].sort_values("entry_time").reset_index(drop=True)

    print(f"\n========== {name} ==========")
    print(f"  Source: {trades_csv}    AM trades: {len(am)} / {len(tdf)} total")

    if am.empty:
        return None, None

    r = am["r"].values.astype(float)
    s = stats_from_r(r)

    print("\n  --- AM-only stats ---")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"    {k:<14} {v:>14,.2f}")
        else:
            print(f"    {k:<14} {v:>14}")

    py = per_year(am)
    print("\n  --- AM by year ---")
    print(py.to_string())

    # Save artifacts
    am.to_csv(out_dir / "trades_am.csv", index=False)
    py.to_csv(out_dir / "stats_by_year.csv")
    with open(out_dir / "stats.json", "w") as f:
        json.dump({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in s.items()}, f, indent=2)
    return am, s


def main():
    out_root = Path("results_am_only")
    out_root.mkdir(exist_ok=True)

    am1, s1 = run("2020-2025  (in-sample, AM only)",
                  Path("results/trades.csv"), out_root / "is_2020_2025")
    am2, s2 = run("2016-2020  (out-of-sample, AM only)",
                  Path("results_oos_2016_2020/trades.csv"), out_root / "oos_2016_2020")

    # ---- Combined comparison plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    if am2 is not None:
        plot_curve(axes[0],
                   pd.to_datetime(am2["exit_time"].fillna(am2["entry_time"]).values, utc=True),
                   am2["r"].values.astype(float),
                   "AM Killzone Only — 2016-2020 (out-of-sample)")
    if am1 is not None:
        plot_curve(axes[1],
                   pd.to_datetime(am1["exit_time"].fillna(am1["entry_time"]).values, utc=True),
                   am1["r"].values.astype(float),
                   "AM Killzone Only — 2020-2025 (in-sample)")
    plt.tight_layout()
    plt.savefig(out_root / "am_only_equity_compare.png", dpi=130)
    plt.close(fig)

    # Side-by-side stats table
    if s1 and s2:
        cmp = pd.DataFrame({"OOS 2016-2020": s2, "IS 2020-2025": s1}).T
        print("\n========== HEAD-TO-HEAD ==========")
        print(cmp.to_string())
        cmp.to_csv(out_root / "head_to_head.csv")

    print(f"\nArtifacts under {out_root}/")


if __name__ == "__main__":
    main()
