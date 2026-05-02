"""Render 10 representative ICT-2022-Model trade screenshots.

Each chart shows the 1m candle context with every model component drawn:
  - HTF bias label
  - 15m liquidity-sweep event line (purple)
  - 15m MSS / CHoCH event line (orange)
  - FVG zone shaded (or OB zone shaded for OB entries)
  - Entry / Stop / Target horizontal lines
  - Trade outcome shaded (green=win, red=loss, gold=expired)

Trades are picked to span: a couple of clean wins (long & short),
a few losers, the biggest winner, the worst loser, and a few middle
performers — enough to verify the entries match the strategy rules.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from model_2022.data_loader import load_1m, resample
from silver_bullet.ict_primitives import htf_bias, align_bias_to_ltf

DATA_PATH    = "1m_data_2.csv"
TRADES_CSV   = "big_test_only_am/trades.csv"
OUT_DIR      = Path("trade_screenshots")
OUT_DIR.mkdir(exist_ok=True)

CONTEXT_BEFORE = 90    # 1m bars before the sweep event for context
CONTEXT_AFTER  = 10    # 1m bars after exit


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} …")
df1m = load_1m(DATA_PATH)
print(f"  {len(df1m):,} 1m bars")

df4h = resample(df1m, "4h")
bias_4h = htf_bias(df4h, ema_fast=20, ema_slow=50)
bias_1m = align_bias_to_ltf(bias_4h, df1m.index)

t = pd.read_csv(TRADES_CSV)
for col in ("entry_time", "exit_time", "setup_sweep_time", "setup_mss_time", "fvg_time", "ob_time"):
    t[col] = pd.to_datetime(t[col], utc=True, errors="coerce").dt.tz_convert("America/New_York")
print(f"  {len(t)} trades loaded")


# ── Pick 10 representative trades ─────────────────────────────────────────────
t = t.sort_values("entry_time").reset_index(drop=True)
sel = []

# 1. First trade (chronological)
sel.append(("first_trade", t.iloc[0]))
# 2. Best winner (max r)
best = t.loc[t["r"].idxmax()]
sel.append(("best_winner", best))
# 3. Worst loser (min r)
worst = t.loc[t["r"].idxmin()]
sel.append(("worst_loser", worst))
# 4. A clean +2R long winner (closest r to 2.0 on a long)
long_wins = t[(t["direction"] == "long") & (t["outcome"] == "win")].copy()
long_wins["dist"] = (long_wins["r"] - 2.0).abs()
if not long_wins.empty:
    sel.append(("clean_long_win", long_wins.sort_values("dist").iloc[0]))
# 5. A clean +2R short winner
short_wins = t[(t["direction"] == "short") & (t["outcome"] == "win")].copy()
short_wins["dist"] = (short_wins["r"] - 2.0).abs()
if not short_wins.empty:
    sel.append(("clean_short_win", short_wins.sort_values("dist").iloc[0]))
# 6. A long loser (closest r to -1.0)
long_losses = t[(t["direction"] == "long") & (t["outcome"] == "loss")].copy()
long_losses["dist"] = (long_losses["r"] - (-1.0)).abs()
if not long_losses.empty:
    sel.append(("long_loser", long_losses.sort_values("dist").iloc[0]))
# 7. A short loser (closest r to -1.0)
short_losses = t[(t["direction"] == "short") & (t["outcome"] == "loss")].copy()
short_losses["dist"] = (short_losses["r"] - (-1.0)).abs()
if not short_losses.empty:
    sel.append(("short_loser", short_losses.sort_values("dist").iloc[0]))
# 8. An expired trade (if any)
expired = t[t["outcome"] == "expired"]
if not expired.empty:
    sel.append(("expired_trade", expired.iloc[len(expired) // 2]))
# 9. Trade from a big winning month (early 2024)
feb24 = t[(t["entry_time"].dt.year == 2024) & (t["entry_time"].dt.month == 2) & (t["outcome"] == "win")]
if not feb24.empty:
    sel.append(("feb_2024_win", feb24.sort_values("r", ascending=False).iloc[0]))
# 10. Trade from worst month (Oct 2023)
oct23 = t[(t["entry_time"].dt.year == 2023) & (t["entry_time"].dt.month == 10)]
if not oct23.empty:
    sel.append(("oct_2023_drawdown", oct23.sort_values("r").iloc[0]))

# Cap at 10
sel = sel[:10]

print(f"\nSelected {len(sel)} representative trades:")
for tag, row in sel:
    print(f"  {tag:<22}  {row['entry_time']}  {row['direction']:<5}  r={row['r']:+.2f}  ({row['outcome']})")


# ── Plotter ───────────────────────────────────────────────────────────────────

def _draw_candles(ax, df):
    """Draw candles as numeric x positions 0..n-1; return n."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    n = len(df)
    bw = 0.6
    for i in range(n):
        col = "#26a69a" if c[i] >= o[i] else "#ef5350"
        ax.plot([i, i], [l[i], h[i]], color=col, linewidth=0.7, zorder=2)
        body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
        rect = mpatches.Rectangle(
            (i - bw / 2, body_lo), bw, max(body_hi - body_lo, 0.05),
            facecolor=col, linewidth=0, zorder=3,
        )
        ax.add_patch(rect)
    ax.set_xlim(-1, n)
    return n


def _x_for(ts, chart_idx):
    """Numeric x-position for ts on the chart frame's index."""
    if pd.isna(ts) or ts is None:
        return None
    pos = chart_idx.searchsorted(ts)
    return float(min(pos, len(chart_idx) - 1))


def plot_trade(row, tag, idx_num):
    sweep_ts = row["setup_sweep_time"]
    mss_ts   = row["setup_mss_time"]
    fvg_ts   = row["fvg_time"] if row["entry_type"] == "fvg" else row.get("ob_time")
    entry_ts = row["entry_time"]
    exit_ts  = row["exit_time"] if not pd.isna(row["exit_time"]) else entry_ts

    anchor_ts = sweep_ts if not pd.isna(sweep_ts) else entry_ts
    anchor_pos = df1m.index.searchsorted(anchor_ts)
    exit_pos   = df1m.index.searchsorted(exit_ts)

    start_pos = max(0, anchor_pos - CONTEXT_BEFORE)
    end_pos   = min(len(df1m) - 1, exit_pos + CONTEXT_AFTER)

    chart = df1m.iloc[start_pos: end_pos + 1].copy()
    if chart.empty:
        return

    fig, ax = plt.subplots(figsize=(17, 8))
    n = _draw_candles(ax, chart)

    # FVG / OB zone (full width, semi-transparent)
    if row["entry_type"] == "fvg" and not pd.isna(row.get("fvg_top", np.nan)):
        zone_lo, zone_hi = float(row["fvg_bottom"]), float(row["fvg_top"])
        zone_color = "#42a5f5" if row["direction"] == "long" else "#ab47bc"
        zone_label = "FVG (bull)" if row["direction"] == "long" else "FVG (bear)"
        ax.axhspan(zone_lo, zone_hi, color=zone_color, alpha=0.18,
                   label=f"{zone_label} {zone_lo:.1f}–{zone_hi:.1f}")
    elif row["entry_type"] == "ob" and not pd.isna(row.get("ob_high", np.nan)):
        zone_lo, zone_hi = float(row["ob_low"]), float(row["ob_high"])
        ax.axhspan(zone_lo, zone_hi, color="#ffa726", alpha=0.20,
                   label=f"Order Block {zone_lo:.1f}–{zone_hi:.1f}")

    # Entry / Stop / Target horizontal lines
    ax.axhline(row["entry"],  color="dodgerblue", lw=1.6, ls="--",
               label=f"Entry  {row['entry']:.1f}", zorder=4)
    ax.axhline(row["stop"],   color="crimson",    lw=1.6, ls=":",
               label=f"Stop   {row['stop']:.1f}",  zorder=4)
    ax.axhline(row["target"], color="limegreen",  lw=1.6, ls="-.",
               label=f"Target {row['target']:.1f}", zorder=4)

    # Vertical event lines
    x_sweep = _x_for(sweep_ts, chart.index)
    x_mss   = _x_for(mss_ts, chart.index)
    x_fvg   = _x_for(fvg_ts, chart.index)
    x_entry = _x_for(entry_ts, chart.index)
    x_exit  = _x_for(exit_ts, chart.index)

    if x_sweep is not None:
        ax.axvline(x_sweep, color="mediumpurple", lw=1.4, ls="--", alpha=0.85,
                   label=f"15m Sweep  {sweep_ts.strftime('%H:%M')}")
    if x_mss is not None:
        ax.axvline(x_mss, color="darkorange", lw=1.4, ls="--", alpha=0.85,
                   label=f"15m MSS   {mss_ts.strftime('%H:%M')}")
    if x_fvg is not None and x_fvg != x_entry:
        ax.axvline(x_fvg, color="navy", lw=1.0, ls=":", alpha=0.6,
                   label=f"FVG forms {fvg_ts.strftime('%H:%M')}")
    if x_entry is not None:
        ax.axvline(x_entry, color="dodgerblue", lw=1.6, alpha=0.85,
                   label=f"Entry @ {entry_ts.strftime('%H:%M')}")
    if x_exit is not None:
        ax.axvline(x_exit, color="gray", lw=1.0, ls=":", alpha=0.6,
                   label=f"Exit  @ {exit_ts.strftime('%H:%M')}")

    # Shade trade duration
    if x_entry is not None and x_exit is not None:
        shade = ("limegreen" if row["outcome"] == "win"
                 else "crimson" if row["outcome"] == "loss"
                 else "gold")
        ax.axvspan(x_entry, max(x_exit, x_entry + 0.5), alpha=0.08, color=shade, zorder=1)

    # X tick labels
    tick_step = max(1, n // 14)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([chart.index[p].strftime("%m/%d %H:%M") for p in ticks],
                       rotation=30, ha="right", fontsize=8)

    # HTF bias at entry
    bias_at_entry = int(bias_1m.loc[bias_1m.index <= entry_ts].iloc[-1]) if (bias_1m.index <= entry_ts).any() else 0
    bias_label = "+1 BULL" if bias_at_entry == 1 else "-1 BEAR" if bias_at_entry == -1 else "0 FLAT"
    bias_color = "#2e7d32" if bias_at_entry == 1 else "#c62828" if bias_at_entry == -1 else "gray"

    # Title
    out_str = (row["outcome"] or "open").upper()
    title_color = ("#1b5e20" if row["outcome"] == "win"
                   else "#b71c1c" if row["outcome"] == "loss"
                   else "#bf6f00")
    dir_arrow = "▲ LONG" if row["direction"] == "long" else "▼ SHORT"
    risk_pts  = abs(row["entry"] - row["stop"])
    ax.set_title(
        f"#{idx_num:02d}  {tag}   |   {dir_arrow}   |   {out_str}   r={row['r']:+.2f}\n"
        f"Entry {entry_ts.strftime('%Y-%m-%d %H:%M ET')} → Exit "
        f"{exit_ts.strftime('%H:%M ET') if exit_ts else '—'}  |  "
        f"HTF Bias {bias_label}  |  Type {row['entry_type'].upper()}  |  "
        f"risk {risk_pts:.1f} pts  |  killzone {row['killzone']}",
        fontsize=11, fontweight="bold", color=title_color,
    )

    ax.set_ylabel("Price (NAS pts)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(alpha=0.25)

    # Y zoom
    pad = (chart["high"].max() - chart["low"].min()) * 0.08
    y_lo = min(chart["low"].min(), row["stop"]) - pad
    y_hi = max(chart["high"].max(), row["target"]) + pad
    ax.set_ylim(y_lo, y_hi)

    # Bias chip in upper-right
    ax.text(0.99, 0.97, f"4H Bias: {bias_label}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bias_color, alpha=0.85,
                      edgecolor="black"))

    plt.tight_layout()
    out = OUT_DIR / f"{idx_num:02d}_{tag}_{row['direction']}_{row['outcome']}_{row['r']:+.2f}R.png".replace("+", "p").replace("-", "n")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


print(f"\nWriting trade charts → {OUT_DIR}/")
for i, (tag, row) in enumerate(sel, start=1):
    plot_trade(row, tag, i)

print(f"\nDone. {len(sel)} screenshots in {OUT_DIR}/")
