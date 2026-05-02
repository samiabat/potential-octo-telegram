"""Backtest statistics and plots for the 2022 Model."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


# ---------------------------------------------------------------------------
# Per-trade candlestick chart
# ---------------------------------------------------------------------------

def _draw_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw a simple OHLC candlestick chart on *ax* using numeric x positions."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    bar_width = 0.6  # fraction of 1 unit on the x-axis

    for i in range(n):
        color = "#26a69a" if c[i] >= o[i] else "#ef5350"  # teal up, red down
        # Wick
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.8, zorder=2)
        # Body
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        rect = mpatches.Rectangle(
            (i - bar_width / 2, body_lo),
            bar_width,
            max(body_hi - body_lo, 0.05),  # min height so doji bars are visible
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.set_xlim(-1, n)


def plot_trade_chart(
    trade: Trade,
    df1m: pd.DataFrame,
    out_path: str,
    trade_num: int = 0,
    context_bars_before: int = 60,   # bars before sweep (context)
    context_bars_after: int = 15,    # bars after exit (breathing room)
) -> None:
    """Draw a 1m candlestick chart for a single trade.

    The chart shows:
      - Price action from *context_bars_before* bars before the liquidity
        sweep event through to *context_bars_after* bars after exit
      - Vertical dashed lines: sweep (purple), MSS (orange), entry (blue)
      - Horizontal lines: entry (blue dashed), stop (red dotted), target (green dash-dot)
      - Shaded trade duration zone (green=win, red=loss, gold=expired)
      - Outcome and R-multiple in the title
    """
    sweep_ts = trade.setup_sweep_time
    entry_ts = trade.entry_time
    exit_ts  = trade.exit_time if trade.exit_time is not None else entry_ts

    # Use the sweep time as the left anchor if available, else entry
    anchor_ts = sweep_ts if sweep_ts is not None else entry_ts

    idx_arr = df1m.index

    def _find_pos(ts: pd.Timestamp | None, fallback: int = 0) -> int:
        if ts is None:
            return fallback
        pos = idx_arr.searchsorted(ts)
        return int(min(pos, len(idx_arr) - 1))

    anchor_pos = _find_pos(anchor_ts)
    exit_pos   = _find_pos(exit_ts)

    start_pos = max(0, anchor_pos - context_bars_before)
    end_pos   = min(len(idx_arr) - 1, exit_pos + context_bars_after)

    chart_df = df1m.iloc[start_pos : end_pos + 1].copy()
    if chart_df.empty:
        return

    n = len(chart_df)

    def _x(ts: pd.Timestamp | None) -> float | None:
        if ts is None:
            return None
        pos = chart_df.index.searchsorted(ts)
        return float(min(pos, n - 1))

    x_sweep = _x(sweep_ts)
    x_mss   = _x(trade.setup_mss_time)
    x_entry = _x(entry_ts)
    x_exit  = _x(exit_ts)

    fig, ax = plt.subplots(figsize=(16, 7))
    _draw_candles(ax, chart_df)

    # Horizontal price levels
    ax.axhline(trade.entry, color="dodgerblue", linewidth=1.6,
               linestyle="--", label=f"Entry  {trade.entry:.1f}", zorder=4)
    ax.axhline(trade.stop,  color="crimson",    linewidth=1.6,
               linestyle=":",  label=f"Stop   {trade.stop:.1f}",  zorder=4)
    ax.axhline(trade.target, color="limegreen", linewidth=1.6,
               linestyle="-.", label=f"Target {trade.target:.1f}", zorder=4)

    # Vertical event lines
    if x_sweep is not None:
        ax.axvline(x_sweep, color="mediumpurple", linewidth=1.2,
                   linestyle="--", alpha=0.8, label="15m Sweep")
    if x_mss is not None:
        ax.axvline(x_mss, color="darkorange", linewidth=1.2,
                   linestyle="--", alpha=0.8, label="15m MSS")
    if x_entry is not None:
        ax.axvline(x_entry, color="dodgerblue", linewidth=1.4,
                   linestyle="-", alpha=0.7, label="Entry bar")
    if x_exit is not None:
        ax.axvline(x_exit, color="gray", linewidth=1.0,
                   linestyle=":", alpha=0.6, label="Exit bar")

    # Shade trade duration
    if x_entry is not None and x_exit is not None:
        shade_color = (
            "limegreen" if trade.outcome == "win"
            else "crimson" if trade.outcome == "loss"
            else "gold"
        )
        ax.axvspan(x_entry, max(x_exit, x_entry + 0.5),
                   alpha=0.08, color=shade_color, zorder=1)

    # X-axis tick labels — show at most ~12 labels
    tick_step = max(1, n // 12)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [chart_df.index[p].strftime("%m/%d %H:%M") for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    # Title
    outcome_str = (trade.outcome or "open").upper()
    r_str = f"{trade.r_multiple:+.2f}R"
    direction_icon = "▲ LONG" if trade.direction == "long" else "▼ SHORT"
    title_color = (
        "darkgreen" if trade.outcome == "win"
        else "darkred" if trade.outcome == "loss"
        else "goldenrod"
    )
    ax.set_title(
        f"Trade #{trade_num:03d}  |  {direction_icon}  |  {trade.killzone.upper()}  "
        f"|  {outcome_str}  {r_str}\n"
        f"Entry: {entry_ts.strftime('%Y-%m-%d %H:%M')}  →  "
        f"Exit: {exit_ts.strftime('%H:%M') if exit_ts else '—'}  "
        f"|  Type: {trade.entry_type.upper()}",
        fontsize=11, fontweight="bold", color=title_color,
    )
    ax.set_ylabel("Price (NAS pts)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.2)

    # Auto Y-zoom with small margin
    price_range = chart_df["high"].max() - chart_df["low"].min()
    margin = price_range * 0.08
    ax.set_ylim(chart_df["low"].min() - margin, chart_df["high"].max() + margin)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_all_trade_charts(
    trades: list[Trade],
    df1m: pd.DataFrame,
    charts_dir: str | Path,
    context_bars_before: int = 60,
    context_bars_after: int = 15,
) -> None:
    """Generate one candlestick chart per trade, saved to *charts_dir*.

    Files are named:
        trade_001_long_win_p2.00R.png
        trade_002_short_loss_n1.03R.png
    """
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(trades)
    print(f"  Generating {total} trade charts → {out}/")
    for idx, t in enumerate(trades, start=1):
        outcome = t.outcome or "open"
        r_tag = (
            f"{t.r_multiple:+.2f}R"
            .replace("+", "p").replace("-", "n").replace(".", "d")
        )
        fname = f"trade_{idx:03d}_{t.direction}_{outcome}_{r_tag}.png"
        plot_trade_chart(
            t, df1m, str(out / fname),
            trade_num=idx,
            context_bars_before=context_bars_before,
            context_bars_after=context_bars_after,
        )
        if idx % 50 == 0:
            print(f"    ... {idx}/{total} done")
    print(f"  All {total} charts saved.")
