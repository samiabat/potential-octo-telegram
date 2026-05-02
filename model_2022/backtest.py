"""Backtest statistics and plots for the 2022 Model."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .strategy import Trade


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = [{
        "entry_time":       t.entry_time,
        "exit_time":        t.exit_time,
        "direction":        t.direction,
        "killzone":         t.killzone,
        "entry_type":       t.entry_type,
        "entry":            t.entry,
        "stop":             t.stop,
        "target":           t.target,
        "exit":             t.exit_price,
        "outcome":          t.outcome,
        "r":                t.r_multiple,
        "setup_sweep_time": t.setup_sweep_time,
        "setup_mss_time":   t.setup_mss_time,
    } for t in trades]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(tdf: pd.DataFrame, risk_per_trade: float = 100.0) -> dict:
    if tdf.empty:
        return {"trades": 0}

    r   = tdf["r"].values
    pnl = r * risk_per_trade
    equity = pnl.cumsum() + 10_000

    wins      = int((r > 0).sum())
    losses    = int((r <= 0).sum())
    gross_win  = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    peak     = np.maximum.accumulate(equity)
    dd       = equity - peak
    max_dd   = float(dd.min())
    max_dd_p = float((dd / peak).min() * 100)

    # Sharpe-like: annualised R / std(R), using trade-level R
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    by_kz = tdf.groupby("killzone")["r"].agg(["count", "mean", "sum"])
    by_type = tdf.groupby("entry_type")["r"].agg(["count", "mean", "sum"])
    by_dir  = tdf.groupby("direction")["r"].agg(["count", "mean", "sum"])

    return {
        "trades":        int(len(tdf)),
        "wins":          wins,
        "losses":        losses,
        "win_rate_%":    float(wins / len(tdf) * 100),
        "expectancy_R":  float(r.mean()),
        "total_R":       float(r.sum()),
        "sharpe_R":      sharpe,
        "gross_win_$":   gross_win,
        "gross_loss_$":  gross_loss,
        "net_pnl_$":     float(pnl.sum()),
        "profit_factor": float(pf),
        "max_dd_$":      max_dd,
        "max_dd_%":      max_dd_p,
        "final_equity":  float(equity[-1]),
        "by_killzone":   by_kz,
        "by_entry_type": by_type,
        "by_direction":  by_dir,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity(
    tdf: pd.DataFrame,
    out_path: str,
    risk_per_trade: float = 100.0,
) -> None:
    if tdf.empty:
        return
    pnl    = tdf["r"].values * risk_per_trade
    equity = pnl.cumsum() + 10_000
    times  = pd.to_datetime(
        tdf["exit_time"].fillna(tdf["entry_time"]).values
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    peak  = np.maximum.accumulate(equity)
    dd_pc = (equity - peak) / peak * 100

    axes[0].plot(times, equity, color="tab:blue", linewidth=1.4, label="Equity")
    axes[0].fill_between(times, equity, peak,
                         where=(equity < peak),
                         color="red", alpha=0.18, label="Drawdown")
    axes[0].set_title(
        "ICT 2022 Model — Equity Curve  (NAS 1m, HTF→15m→1m cascade)",
        fontsize=13, fontweight="bold",
    )
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(times, dd_pc, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    axes[2].hist(tdf["r"].values, bins=40,
                 color="tab:green", alpha=0.7, edgecolor="black")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("R multiple per trade")
    axes[2].set_ylabel("Count")
    axes[2].set_title("R-multiple distribution")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_breakdown(tdf: pd.DataFrame, out_path: str,
                   risk_per_trade: float = 100.0) -> None:
    """Two-panel: cumulative PnL by killzone, and breakdown bar charts."""
    if tdf.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cumulative PnL by killzone
    for kz, g in tdf.groupby("killzone"):
        eq = (g["r"].values * risk_per_trade).cumsum()
        axes[0].plot(
            pd.to_datetime(g["exit_time"].values), eq,
            label=kz, linewidth=1.3,
        )
    axes[0].set_title("Cumulative PnL by killzone")
    axes[0].set_ylabel("PnL ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Stats by entry type (FVG vs OB)
    by_type = tdf.groupby("entry_type")["r"].agg(["count", "mean", "sum"])
    by_type[["count", "mean", "sum"]].plot(kind="bar", ax=axes[1])
    axes[1].set_title("FVG vs OB — count / mean R / sum R")
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=0)

    # Stats by direction
    by_dir = tdf.groupby("direction")["r"].agg(["count", "mean", "sum"])
    by_dir[["count", "mean", "sum"]].plot(kind="bar", ax=axes[2])
    axes[2].set_title("Long vs Short — count / mean R / sum R")
    axes[2].grid(alpha=0.3)
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_monthly(tdf: pd.DataFrame, out_path: str,
                 risk_per_trade: float = 100.0) -> None:
    """Bar chart of monthly PnL."""
    if tdf.empty:
        return
    t = tdf.copy()
    t["month"] = pd.to_datetime(
        tdf["exit_time"].fillna(tdf["entry_time"])
    ).dt.tz_localize(None).dt.to_period("M")
    monthly = t.groupby("month")["r"].sum() * risk_per_trade

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in monthly.values]
    ax.bar(monthly.index.astype(str), monthly.values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly PnL — ICT 2022 Model")
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
