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
        "sweep_level":      t.sweep_level,
        "fvg_top":          t.fvg_top,
        "fvg_bottom":       t.fvg_bottom,
        "fvg_time":         t.fvg_time,
        "setup_mss_tf":     t.setup_mss_tf,
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
                 risk_per_trade: float = 100.0,
                 title_suffix: str = "ICT 2022 Model") -> None:
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
    ax.set_title(f"Monthly PnL — {title_suffix}")
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_weekly(tdf: pd.DataFrame, out_path: str,
                risk_per_trade: float = 100.0,
                title_suffix: str = "ICT 2022 Model") -> None:
    """Bar chart of weekly PnL (ISO week)."""
    if tdf.empty:
        return
    t = tdf.copy()
    dt = pd.to_datetime(tdf["exit_time"].fillna(tdf["entry_time"])).dt.tz_localize(None)
    t["week"] = dt.dt.to_period("W")
    weekly = t.groupby("week")["r"].sum() * risk_per_trade

    fig, ax = plt.subplots(figsize=(max(14, len(weekly) // 2), 4))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in weekly.values]
    labels = [str(w.start_time.strftime("%Y-%m-%d")) for w in weekly.index]
    ax.bar(range(len(weekly)), weekly.values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Weekly PnL — {title_suffix}")
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("Week starting")
    tick_step = max(1, len(weekly) // 20)
    ax.set_xticks(range(0, len(weekly), tick_step))
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_yearly(tdf: pd.DataFrame, out_path: str,
                risk_per_trade: float = 100.0,
                title_suffix: str = "ICT 2022 Model") -> None:
    """Bar chart of yearly PnL with per-year stats annotated."""
    if tdf.empty:
        return
    t = tdf.copy()
    t["year"] = pd.to_datetime(
        tdf["exit_time"].fillna(tdf["entry_time"])
    ).dt.tz_localize(None).dt.year
    grp = t.groupby("year")
    yearly_pnl = grp["r"].sum() * risk_per_trade
    yearly_wr  = grp["r"].apply(lambda x: (x > 0).mean() * 100)
    yearly_cnt = grp["r"].count()

    fig, ax = plt.subplots(figsize=(max(8, len(yearly_pnl) * 1.5), 5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in yearly_pnl.values]
    bars = ax.bar(yearly_pnl.index.astype(str), yearly_pnl.values,
                  color=colors, edgecolor="black", width=0.6)
    # Annotate each bar with trade count and win-rate
    for bar, yr in zip(bars, yearly_pnl.index):
        cnt = int(yearly_cnt[yr])
        wr  = float(yearly_wr[yr])
        y   = bar.get_height()
        va  = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (5 if y >= 0 else -5),
                f"{cnt}t  {wr:.0f}%wr", ha="center", va=va, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Yearly PnL — {title_suffix}")
    ax.set_ylabel("PnL ($)")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def compute_period_stats(tdf: pd.DataFrame,
                         risk_per_trade: float = 100.0) -> dict[str, pd.DataFrame]:
    """Return weekly, monthly, and yearly breakdown DataFrames."""
    if tdf.empty:
        return {"weekly": pd.DataFrame(), "monthly": pd.DataFrame(), "yearly": pd.DataFrame()}

    t = tdf.copy()
    dt = pd.to_datetime(t["exit_time"].fillna(t["entry_time"])).dt.tz_localize(None)

    t["week"]  = dt.dt.to_period("W")
    t["month"] = dt.dt.to_period("M")
    t["year"]  = dt.dt.year

    def _agg(grp):
        return grp["r"].agg(
            trades="count",
            wins=lambda x: int((x > 0).sum()),
            losses=lambda x: int((x <= 0).sum()),
            win_rate_pct=lambda x: round(float((x > 0).mean() * 100), 1),
            mean_R="mean",
            total_R="sum",
            net_pnl=lambda x: round(float(x.sum() * risk_per_trade), 2),
        ).round(3)

    weekly  = _agg(t.groupby("week"))
    monthly = _agg(t.groupby("month"))
    yearly  = _agg(t.groupby("year"))

    return {"weekly": weekly, "monthly": monthly, "yearly": yearly}


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


def _add_clean_levels(
    ax: plt.Axes,
    chart_df: pd.DataFrame,
    trade: "Trade",
    *,
    draw_fvg_box: bool = True,
) -> None:
    """Add only the essential ICT levels to an axis (thin lines, no vertical markers).

    Draws:
      - Purple dashed horizontal: 15m sweep level (liquidity taken)
      - Gold filled rectangle: FVG box from formation bar to entry bar
      - Blue thin line: entry price
      - Red thin line: stop-loss
      - Green thin line: take-profit

    No vertical lines, no shading, no background decorations.
    """
    n = len(chart_df)
    lw = 0.6   # universal thin linewidth for all price-level lines

    # ── Liquidity sweep level (horizontal) ──────────────────────────────────
    if trade.sweep_level is not None:
        ax.axhline(trade.sweep_level, color="mediumpurple", linewidth=lw,
                   linestyle="--", alpha=0.9, zorder=5,
                   label=f"Liquidity  {trade.sweep_level:.1f}")

    # ── FVG rectangle (from formation to entry, not full width) ─────────────
    if draw_fvg_box and trade.fvg_top is not None and trade.fvg_bottom is not None:
        # x_start = bar closest to FVG formation time; x_end = entry bar
        fvg_t = trade.fvg_time if trade.fvg_time is not None else trade.setup_mss_time
        entry_t = trade.entry_time

        def _xpos(ts):
            if ts is None:
                return None
            pos = chart_df.index.searchsorted(ts)
            return int(min(pos, n - 1))

        x0 = _xpos(fvg_t)
        x1 = _xpos(entry_t)
        if x0 is not None and x1 is not None and x1 >= x0:
            fvg_rect = mpatches.Rectangle(
                (x0 - 0.3, trade.fvg_bottom),
                (x1 - x0 + 0.6),
                trade.fvg_top - trade.fvg_bottom,
                linewidth=0.5,
                edgecolor="goldenrod",
                facecolor="gold",
                alpha=0.25,
                zorder=4,
                label=f"FVG  {trade.fvg_bottom:.1f}–{trade.fvg_top:.1f}",
            )
            ax.add_patch(fvg_rect)

    # ── Entry / SL / TP ─────────────────────────────────────────────────────
    ax.axhline(trade.entry,  color="dodgerblue", linewidth=lw,
               linestyle="-",  zorder=6, label=f"Entry  {trade.entry:.1f}")
    ax.axhline(trade.stop,   color="crimson",    linewidth=lw,
               linestyle="-",  zorder=6, label=f"SL  {trade.stop:.1f}")
    ax.axhline(trade.target, color="#00cc44",    linewidth=lw,
               linestyle="-",  zorder=6, label=f"TP  {trade.target:.1f}")


def plot_trade_chart(
    trade: "Trade",
    df1m: pd.DataFrame,
    out_path: str,
    trade_num: int = 0,
    context_bars_before: int = 80,   # 1m bars before sweep
    context_bars_after: int = 20,    # 1m bars after exit
) -> None:
    """1m entry chart: clean candles + horizontal levels only."""
    sweep_ts = trade.setup_sweep_time
    entry_ts = trade.entry_time
    exit_ts  = trade.exit_time if trade.exit_time is not None else entry_ts
    anchor_ts = sweep_ts if sweep_ts is not None else entry_ts

    idx_arr = df1m.index

    def _find_pos(ts, fallback=0):
        if ts is None:
            return fallback
        return int(min(idx_arr.searchsorted(ts), len(idx_arr) - 1))

    start_pos = max(0, _find_pos(anchor_ts) - context_bars_before)
    end_pos   = min(len(idx_arr) - 1, _find_pos(exit_ts) + context_bars_after)
    chart_df  = df1m.iloc[start_pos : end_pos + 1].copy()
    if chart_df.empty:
        return

    n = len(chart_df)
    fig, ax = plt.subplots(figsize=(18, 6))
    _draw_candles(ax, chart_df)
    _add_clean_levels(ax, chart_df, trade, draw_fvg_box=True)

    tick_step = max(1, n // 14)
    tpos = list(range(0, n, tick_step))
    ax.set_xticks(tpos)
    ax.set_xticklabels(
        [chart_df.index[p].strftime("%m/%d %H:%M") for p in tpos],
        rotation=30, ha="right", fontsize=7,
    )

    outcome_str = (trade.outcome or "open").upper()
    title_color = (
        "darkgreen" if trade.outcome == "win"
        else "darkred" if trade.outcome == "loss" else "goldenrod"
    )
    ax.set_title(
        f"#{trade_num:03d} {'▲' if trade.direction == 'long' else '▼'}"
        f"  {trade.killzone.upper()}  {outcome_str}  {trade.r_multiple:+.2f}R"
        f"  |  1m  |  {entry_ts.strftime('%Y-%m-%d %H:%M')}",
        fontsize=9, fontweight="bold", color=title_color,
    )
    ax.set_ylabel("Price", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, handlelength=1.5)
    ax.grid(alpha=0.15)

    pr = chart_df["high"].max() - chart_df["low"].min()
    margin = max(pr * 0.06, 2.0)
    ax.set_ylim(chart_df["low"].min() - margin, chart_df["high"].max() + margin)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_trade_tf_chart(
    trade: "Trade",
    df1m: pd.DataFrame,
    tf: str,
    out_path: str,
    trade_num: int = 0,
    context_bars_before: int = 80,
    context_bars_after: int = 6,
) -> None:
    """Generic TF context chart (15m or 5m).

    Shows 80 bars before the sweep and 6 bars after entry on the given
    timeframe, with clean horizontal levels only (no vertical lines).
    """
    from .data_loader import resample as _resample
    df_tf = _resample(df1m, tf)
    if df_tf.empty:
        return

    sweep_ts = trade.setup_sweep_time
    entry_ts = trade.entry_time
    anchor_ts = sweep_ts if sweep_ts is not None else entry_ts

    idx_arr = df_tf.index

    def _fp(ts, fallback=0):
        if ts is None:
            return fallback
        return int(min(idx_arr.searchsorted(ts), len(idx_arr) - 1))

    # The sweep event timestamp is shifted +1 bar by the strategy; the actual
    # candle that swept is 1 bar earlier on the TF chart.
    anchor_pos = max(0, _fp(anchor_ts) - 1)
    entry_pos  = _fp(entry_ts)

    start_pos  = max(0, anchor_pos - context_bars_before)
    end_pos    = min(len(idx_arr) - 1, entry_pos + context_bars_after)
    chart_df   = df_tf.iloc[start_pos : end_pos + 1].copy()
    if chart_df.empty:
        return

    n = len(chart_df)
    fig, ax = plt.subplots(figsize=(18, 6))
    _draw_candles(ax, chart_df)
    _add_clean_levels(ax, chart_df, trade, draw_fvg_box=True)

    tick_step = max(1, n // 14)
    tpos = list(range(0, n, tick_step))
    ax.set_xticks(tpos)
    ax.set_xticklabels(
        [chart_df.index[p].strftime("%m/%d %H:%M") for p in tpos],
        rotation=30, ha="right", fontsize=7,
    )

    outcome_str = (trade.outcome or "open").upper()
    title_color = (
        "darkgreen" if trade.outcome == "win"
        else "darkred" if trade.outcome == "loss" else "goldenrod"
    )
    tf_label = tf.replace("min", "m").replace("T", "m")
    ax.set_title(
        f"#{trade_num:03d} {'▲' if trade.direction == 'long' else '▼'}"
        f"  {trade.killzone.upper()}  {outcome_str}  {trade.r_multiple:+.2f}R"
        f"  |  {tf_label}  |  {entry_ts.strftime('%Y-%m-%d %H:%M')}",
        fontsize=9, fontweight="bold", color=title_color,
    )
    ax.set_ylabel("Price", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, handlelength=1.5)
    ax.grid(alpha=0.15)

    pr = chart_df["high"].max() - chart_df["low"].min()
    margin = max(pr * 0.07, 3.0)
    ax.set_ylim(chart_df["low"].min() - margin, chart_df["high"].max() + margin)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trade_chart_15m(
    trade: "Trade",
    df1m: pd.DataFrame,
    out_path: str,
    trade_num: int = 0,
    context_15m_before: int = 80,
    context_15m_after: int = 6,
) -> None:
    """15m context chart with 80 bars before sweep."""
    _plot_trade_tf_chart(
        trade, df1m, "15min", out_path,
        trade_num=trade_num,
        context_bars_before=context_15m_before,
        context_bars_after=context_15m_after,
    )


def plot_trade_chart_5m(
    trade: "Trade",
    df1m: pd.DataFrame,
    out_path: str,
    trade_num: int = 0,
    context_5m_before: int = 80,
    context_5m_after: int = 10,
) -> None:
    """5m context chart with 80 bars before sweep."""
    _plot_trade_tf_chart(
        trade, df1m, "5min", out_path,
        trade_num=trade_num,
        context_bars_before=context_5m_before,
        context_bars_after=context_5m_after,
    )



def _zigzag_swings(
    df_sw: pd.DataFrame,
    n_swings: int = 40,
) -> list[tuple[int, float, str]]:
    """Return the last *n_swings* turning-point (bar_pos, price, label).

    Labels: HH / LH (swing highs) and HL / LL (swing lows).
    bar_pos is the integer row index inside df_sw.
    """
    highs: list[tuple[int, float]] = []
    lows:  list[tuple[int, float]] = []
    for i in range(len(df_sw)):
        if df_sw["swing_high"].iat[i]:
            highs.append((i, float(df_sw["swing_high_price"].iat[i])))
        if df_sw["swing_low"].iat[i]:
            lows.append((i, float(df_sw["swing_low_price"].iat[i])))

    # Merge into one stream sorted by bar position
    all_swings: list[tuple[int, float, str]] = (
        [(i, p, "H") for i, p in highs] +
        [(i, p, "L") for i, p in lows]
    )
    all_swings.sort(key=lambda x: x[0])

    # Take last n_swings
    all_swings = all_swings[-n_swings:]

    # Label each point relative to the previous same-type swing
    labeled: list[tuple[int, float, str]] = []
    prev_h: float | None = None
    prev_l: float | None = None
    for i, price, typ in all_swings:
        if typ == "H":
            if prev_h is None:
                label = "SH"          # first swing high — no comparison yet
            elif price > prev_h:
                label = "HH"
            else:
                label = "LH"
            prev_h = price
        else:
            if prev_l is None:
                label = "SL"
            elif price > prev_l:
                label = "HL"
            else:
                label = "LL"
            prev_l = price
        labeled.append((i, price, label))

    return labeled


def plot_htf_swing_chart(
    trade: "Trade",
    df1m: pd.DataFrame,
    tf: str,
    out_path: str,
    trade_num: int = 0,
    context_bars_before: int = 60,
    context_bars_after: int = 5,
    swing_n: int = 3,
) -> None:
    """HTF (4H or Daily) chart with zigzag HH/HL/LH/LL drawing.

    Shows:
      • Candles (clean OHLC)
      • Zigzag line connecting confirmed swing highs and lows
      • HH / LH / HL / LL labels at each turning point
      • Horizontal lines for entry, SL, TP
      • Vertical dashed line marking the entry bar
    No lower-timeframe detail (no FVG boxes, no 1m noise).
    """
    from .data_loader import resample as _resample
    from .ict_primitives import swing_points as _swings

    df_tf = _resample(df1m, tf)
    if df_tf.empty:
        return

    df_sw = _swings(df_tf, n=swing_n)

    entry_ts = trade.entry_time
    idx_arr  = df_tf.index

    def _fp(ts, fallback=0):
        if ts is None:
            return fallback
        return int(min(idx_arr.searchsorted(ts), len(idx_arr) - 1))

    entry_pos = _fp(entry_ts)
    start_pos = max(0, entry_pos - context_bars_before)
    end_pos   = min(len(idx_arr) - 1, entry_pos + context_bars_after)
    chart_df  = df_sw.iloc[start_pos: end_pos + 1].copy()
    if chart_df.empty:
        return

    n = len(chart_df)
    fig, ax = plt.subplots(figsize=(18, 7))
    _draw_candles(ax, chart_df)

    # ── Zigzag ──────────────────────────────────────────────────────────
    zz = _zigzag_swings(chart_df, n_swings=60)
    if len(zz) >= 2:
        zz_x = [p for p, _, _ in zz]
        zz_y = [pr for _, pr, _ in zz]
        ax.plot(zz_x, zz_y,
                color="white", linewidth=1.2, alpha=0.85,
                linestyle="--", zorder=5)

    LABEL_COLORS = {
        "HH": "#00e676",  "LH": "#ff5252",
        "HL": "#69f0ae",  "LL": "#ff1744",
        "SH": "#b2dfdb",  "SL": "#ffcdd2",
    }
    for pos, price, label in zz:
        color = LABEL_COLORS.get(label, "white")
        is_high = label in ("HH", "LH", "SH")
        va = "bottom" if is_high else "top"
        offset = 0.3 if is_high else -0.3
        ax.annotate(
            label,
            xy=(pos, price),
            xytext=(0, 8 if is_high else -8),
            textcoords="offset points",
            ha="center", va=va,
            fontsize=7, fontweight="bold",
            color=color,
            zorder=7,
        )
        ax.plot(pos, price, "o", color=color, markersize=4, zorder=6)

    # ── Entry / SL / TP ─────────────────────────────────────────────────
    lw = 0.7
    ax.axhline(trade.entry,  color="dodgerblue", linewidth=lw,
               linestyle="-",  zorder=8, label=f"Entry  {trade.entry:.1f}")
    ax.axhline(trade.stop,   color="crimson",    linewidth=lw,
               linestyle="-",  zorder=8, label=f"SL  {trade.stop:.1f}")
    ax.axhline(trade.target, color="#00cc44",    linewidth=lw,
               linestyle="-",  zorder=8, label=f"TP  {trade.target:.1f}")

    # ── Vertical entry marker ────────────────────────────────────────────
    rel_entry = entry_pos - start_pos
    if 0 <= rel_entry < n:
        ax.axvline(rel_entry, color="gold", linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=5, label="Entry bar")

    # ── Axes ─────────────────────────────────────────────────────────────
    tick_step = max(1, n // 14)
    tpos = list(range(0, n, tick_step))
    fmt  = "%Y-%m-%d" if tf.upper() in ("1D", "D", "1d") else "%m/%d %H:%M"
    ax.set_xticks(tpos)
    ax.set_xticklabels(
        [chart_df.index[p].strftime(fmt) for p in tpos],
        rotation=30, ha="right", fontsize=7,
    )

    outcome_str = (trade.outcome or "open").upper()
    title_color = (
        "darkgreen" if trade.outcome == "win"
        else "darkred" if trade.outcome == "loss" else "goldenrod"
    )
    tf_label = tf.upper().replace("MIN", "m").replace("H", "H")
    bias_dir = "BULL" if trade.direction == "long" else "BEAR"
    ax.set_title(
        f"#{trade_num:03d}  {tf_label} Bias — {bias_dir}  |  "
        f"{'▲' if trade.direction == 'long' else '▼'}  "
        f"{outcome_str}  {trade.r_multiple:+.2f}R  |  "
        f"{entry_ts.strftime('%Y-%m-%d %H:%M')}",
        fontsize=9, fontweight="bold", color=title_color,
    )
    ax.set_ylabel("Price", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7, handlelength=1.5)
    ax.grid(alpha=0.15)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color(title_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    pr = chart_df["high"].max() - chart_df["low"].min()
    margin = max(pr * 0.08, 5.0)
    ax.set_ylim(chart_df["low"].min() - margin, chart_df["high"].max() + margin)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_all_trade_charts(
    trades: list[Trade],
    df1m: pd.DataFrame,
    charts_dir: str | Path,
    context_bars_before: int = 80,
    context_bars_after: int = 20,
) -> None:
    """Generate five charts per trade, saved into individual subfolders.

    Subfolder: {idx:03d}_{date}_{HHMM}_{direction}_{outcome}/
    Files:
        1d_bias.png      — Daily candles with zigzag HH/HL/LH/LL
        4h_bias.png      — 4H candles with zigzag HH/HL/LH/LL
        15m_context.png  — 15m candles (80 bars before sweep)
        5m_context.png   — 5m candles  (80 bars before sweep)
        1m_entry.png     — 1m candles  (80 bars before sweep, 20 after exit)
    Lower-TF charts show only: liquidity level, FVG box, entry, SL, TP.
    HTF charts show zigzag structure + entry/SL/TP.
    """
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(trades)
    print(f"  Generating {total} trade folders (5 charts each) → {out}/")
    for idx, t in enumerate(trades, start=1):
        outcome = t.outcome or "open"
        date_str = t.entry_time.strftime("%Y-%m-%d_%H%M")
        folder = out / f"{idx:03d}_{date_str}_{t.direction}_{outcome}"
        folder.mkdir(exist_ok=True)

        # HTF bias charts (zigzag structure)
        plot_htf_swing_chart(
            t, df1m, "1D", str(folder / "1d_bias.png"),
            trade_num=idx, context_bars_before=90, context_bars_after=3,
            swing_n=3,
        )
        plot_htf_swing_chart(
            t, df1m, "4h", str(folder / "4h_bias.png"),
            trade_num=idx, context_bars_before=60, context_bars_after=4,
            swing_n=3,
        )
        # Lower-TF execution charts
        plot_trade_chart_15m(t, df1m, str(folder / "15m_context.png"), trade_num=idx)
        plot_trade_chart_5m(t, df1m,  str(folder / "5m_context.png"),  trade_num=idx)
        plot_trade_chart(
            t, df1m, str(folder / "1m_entry.png"),
            trade_num=idx,
            context_bars_before=context_bars_before,
            context_bars_after=context_bars_after,
        )
        if idx % 25 == 0:
            print(f"    ... {idx}/{total} done")
    print(f"  All {total} trade folders saved to {out}/")
