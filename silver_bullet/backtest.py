"""Backtest stats + equity curve plotting."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .strategy import Trade


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = [{
        "entry_time": t.entry_time,
        "exit_time":  t.exit_time,
        "direction":  t.direction,
        "killzone":   t.killzone,
        "entry":      t.entry,
        "stop":       t.stop,
        "target":     t.target,
        "exit":       t.exit_price,
        "outcome":    t.outcome,
        "r":          t.r_multiple,
    } for t in trades]
    return pd.DataFrame(rows)


def compute_stats(tdf: pd.DataFrame, risk_per_trade: float = 100.0) -> dict:
    if tdf.empty:
        return {"trades": 0}
    r = tdf["r"].values
    pnl = r * risk_per_trade
    equity = pnl.cumsum() + 10_000
    wins = (r > 0).sum()
    losses = (r <= 0).sum()
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    max_dd_pct = (dd / peak).min() * 100
    expectancy_r = r.mean()
    by_kz = tdf.groupby("killzone")["r"].agg(["count", "mean", "sum"])
    return {
        "trades":       int(len(tdf)),
        "wins":         int(wins),
        "losses":       int(losses),
        "win_rate":     float(wins / len(tdf) * 100),
        "expectancy_R": float(expectancy_r),
        "total_R":      float(r.sum()),
        "gross_win_$":  float(gross_win),
        "gross_loss_$": float(gross_loss),
        "net_pnl_$":    float(pnl.sum()),
        "profit_factor": float(pf),
        "max_dd_$":     float(max_dd),
        "max_dd_%":     float(max_dd_pct),
        "final_equity": float(equity[-1]),
        "by_killzone":  by_kz,
    }


def plot_equity(tdf: pd.DataFrame, out_path: str, risk_per_trade: float = 100.0) -> None:
    if tdf.empty:
        return
    pnl = tdf["r"].values * risk_per_trade
    equity = pnl.cumsum() + 10_000
    times = pd.to_datetime(tdf["exit_time"].fillna(tdf["entry_time"]).values)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Equity curve
    axes[0].plot(times, equity, color="tab:blue", linewidth=1.4)
    peak = np.maximum.accumulate(equity)
    axes[0].fill_between(times, equity, peak, where=(equity < peak),
                         color="red", alpha=0.15, label="Drawdown")
    axes[0].set_title("ICT Silver Bullet — Equity Curve  (NAS 5m, 2020-2025)",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Drawdown
    dd_pct = (equity - peak) / peak * 100
    axes[1].fill_between(times, dd_pct, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    # R distribution
    axes[2].hist(tdf["r"].values, bins=40, color="tab:green", alpha=0.7,
                 edgecolor="black")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("R multiple per trade")
    axes[2].set_ylabel("Count")
    axes[2].set_title("R-multiple distribution")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_killzone_breakdown(tdf: pd.DataFrame, out_path: str,
                             risk_per_trade: float = 100.0) -> None:
    if tdf.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grouped = tdf.groupby("killzone")
    for name, g in grouped:
        eq = (g["r"].values * risk_per_trade).cumsum()
        axes[0].plot(pd.to_datetime(g["exit_time"].values), eq, label=name, linewidth=1.3)
    axes[0].set_title("Cumulative PnL by Killzone")
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    counts = grouped["r"].agg(["count", "mean", "sum"])
    counts.plot(kind="bar", ax=axes[1], subplots=False)
    axes[1].set_title("Killzone stats — count / mean R / sum R")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
