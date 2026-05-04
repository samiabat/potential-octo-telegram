"""run_zarattini_orb.py – Paper implementation: Zarattini, Barbon & Aziz (2024)

"A Profitable Day Trading Strategy For The U.S. Equity Market"
SSRN: https://ssrn.com/abstract=4729284

Applied here to NQ Nasdaq futures 1m data (the closest available instrument
to the paper's universe; the authors' earlier work studied QQQ/TQQQ on the
same strategy).

Strategy rules — faithfully from Section 2 / Section 4 of the paper
═══════════════════════════════════════════════════════════════════
1.  Opening Range  : first 5 minutes of RTH, 09:30–09:34 ET
2.  Direction      : determined by the FIRST 5-min candle
                       bullish (close > open) → long candidates only
                       bearish (close < open) → short candidates only
                       doji   (close = open)  → no trade
3.  Entry          : stop-order at OR high (long) / OR low (short)
                     triggered only if price breaks that level
4.  Stop Loss      : 10% × ATR(14) from entry price
5.  Profit Target  : End of Day (EOD, 15:59 ET for NQ RTH close)
6.  Rel-Volume     : 5-min OR volume / 14-day trailing avg 5-min OR volume
                     must be ≥ REL_VOL_MIN (paper: ≥ 100%)
7.  Position size  : notional position so that a stop-loss hit = RISK_PER_R
                     (in points; expressed in R-multiples throughout)

No-look-ahead guarantees
────────────────────────
• ATR(14) is computed from *daily* bars whose close is strictly before the
  current session.  Lag = 1 day.
• Avg 5-min OR volume is a 14-day trailing mean computed strictly before
  the current session.  Lag = 1 session.
• The first 5-min candle [09:30, 09:34] is fully closed before any
  decision is made.
• Entry is triggered as soon as price pierces OR high/low on a *later* bar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_2022.data_loader import load_1m, resample

# ── output ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("zarattini_orb_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── strategy parameters ─────────────────────────────────────────────────────────
ATR_PERIOD      = 14       # rolling ATR window (daily bars)
STOP_ATR_MULT   = 0.10     # stop loss = 10% × ATR(14)
REL_VOL_MIN     = 1.00     # Relative Volume ≥ 100%  (1.0 = 100%)
ATR_VOL_WINDOW  = 14       # rolling window for avg 5-min OR volume

OR_START        = pd.Timestamp("09:30").time()
OR_END          = pd.Timestamp("09:34").time()   # 5-min candle close = 09:34:59
SESSION_END     = pd.Timestamp("15:59").time()   # EOD exit bar

MIN_ATR_PTS     = 5.0      # skip if daily ATR < this (data quality filter)


# ── data class ──────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    date:        str
    direction:   Literal["long", "short"]
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    stop:        float
    or_high:     float
    or_low:      float
    atr14:       float
    rel_vol:     float
    outcome:     str          # 'stop' | 'eod'
    pnl_pts:     float        # price-point P&L per unit (long: exit-entry, short: entry-exit)
    risk_pts:    float        # |entry - stop|
    r:           float        # pnl_pts / risk_pts


# ── helpers ─────────────────────────────────────────────────────────────────────

def _compute_daily_atr(df1m: pd.DataFrame, window: int = 14) -> dict:
    """Compute daily ATR(14) from 1m data, strictly no look-ahead.

    Returns a dict mapping date-string (YYYY-MM-DD) → ATR value where
    the ATR on date D is computed from *completed* sessions ending before D
    (shift(1) relative to that session).
    """
    daily = resample(df1m, "1D")
    tr1 = daily["high"] - daily["low"]
    tr2 = (daily["high"] - daily["close"].shift(1)).abs()
    tr3 = (daily["low"]  - daily["close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    # Shift by 1: today's row uses yesterday's ATR (no look-ahead)
    atr_shifted = atr.shift(1)
    # Return as dict keyed by date string for easy lookup
    return {
        ts.strftime("%Y-%m-%d"): float(val)
        for ts, val in atr_shifted.items()
        if not np.isnan(val)
    }


def _extract_or_bars(day_bars: pd.DataFrame) -> pd.DataFrame:
    """Return the 5-min opening range 1m bars (09:30–09:34 ET)."""
    t = day_bars.index.time
    return day_bars[(t >= OR_START) & (t <= OR_END)]


def _simulate_breakout(
    day_bars: pd.DataFrame,
    after_time: pd.Timestamp,
    trigger_level: float,
    direction: Literal["long", "short"],
    stop: float,
) -> Optional[tuple[float, pd.Timestamp, str, float, float]]:
    """Wait for price to breach trigger_level; then track to stop or EOD.

    Returns (entry_price, exit_price, exit_time, outcome, pnl_pts) or None
    if trigger never hit before EOD.
    """
    # bars strictly after the 5-min candle closes, within session
    sim = day_bars[
        (day_bars.index > after_time) &
        (day_bars.index.time <= SESSION_END)
    ]
    if sim.empty:
        return None

    highs  = sim["high"].to_numpy()
    lows   = sim["low"].to_numpy()
    closes = sim["close"].to_numpy()
    times  = sim.index

    entry_price: Optional[float] = None
    entry_bar_idx: int = -1

    # ── Phase 1: wait for breakout trigger ─────────────────────────────────
    for i in range(len(sim)):
        if direction == "long"  and highs[i] >= trigger_level:
            entry_price   = trigger_level  # stop-buy fills at trigger
            entry_bar_idx = i
            break
        if direction == "short" and lows[i]  <= trigger_level:
            entry_price   = trigger_level  # stop-sell fills at trigger
            entry_bar_idx = i
            break

    if entry_price is None:
        return None  # breakout never triggered

    entry_time = times[entry_bar_idx]

    # ── Phase 2: simulate from entry bar onwards ────────────────────────────
    # Remaining bars (we re-check the entry bar itself pessimistically)
    for i in range(entry_bar_idx, len(sim)):
        if direction == "long"  and lows[i]  <= stop:
            exit_px = stop
            return entry_price, exit_px, times[i], "stop", exit_px - entry_price
        if direction == "short" and highs[i] >= stop:
            exit_px = stop
            return entry_price, exit_px, times[i], "stop", entry_price - exit_px

    # EOD exit
    eod_px = closes[-1]
    pnl = eod_px - entry_price if direction == "long" else entry_price - eod_px
    return entry_price, eod_px, times[-1], "eod", pnl


# ── core backtest ────────────────────────────────────────────────────────────────

def run_zarattini_orb(
    df1m: pd.DataFrame,
    rel_vol_min: float = REL_VOL_MIN,
    atr_period: int    = ATR_PERIOD,
    stop_atr_mult: float = STOP_ATR_MULT,
    vol_window: int    = ATR_VOL_WINDOW,
) -> List[Trade]:
    """Run the Zarattini-Barbon-Aziz 5-min ORB with Relative Volume filter.

    All look-ahead is prevented:
    • ATR(14) uses lag=1 day.
    • 5-min OR volume average uses lag=1 session.
    """
    trades: List[Trade] = []

    # ── pre-compute daily ATR (no look-ahead, lag=1 day) ───────────────────
    daily_atr = _compute_daily_atr(df1m, window=atr_period)

    # ── group 1m bars by calendar date ─────────────────────────────────────
    grouped = df1m.groupby(df1m.index.normalize())

    # Keep a rolling deque of the last vol_window 5-min OR tick-volumes
    or_vol_history: list[float] = []

    for date, day_bars in grouped:
        date_str = date.strftime("%Y-%m-%d")

        # ── fetch no-look-ahead ATR ─────────────────────────────────────────
        atr14 = daily_atr.get(date_str, np.nan)
        if np.isnan(atr14) or atr14 < MIN_ATR_PTS:
            # Record this day's OR vol for future sessions even if we skip
            or_bars_raw = _extract_or_bars(day_bars)
            if not or_bars_raw.empty:
                or_vol_history.append(float(or_bars_raw["tick_volume"].sum()))
                if len(or_vol_history) > vol_window:
                    or_vol_history.pop(0)
            continue

        # ── 5-min opening range ─────────────────────────────────────────────
        or_bars = _extract_or_bars(day_bars)
        if len(or_bars) < 3:
            # Update vol history even on skip
            if not or_bars.empty:
                or_vol_history.append(float(or_bars["tick_volume"].sum()))
                if len(or_vol_history) > vol_window:
                    or_vol_history.pop(0)
            continue

        or_vol_today = float(or_bars["tick_volume"].sum())

        # ── Relative Volume filter (uses history strictly before today) ─────
        if len(or_vol_history) < vol_window:
            # Not enough history yet; record and skip
            or_vol_history.append(or_vol_today)
            if len(or_vol_history) > vol_window:
                or_vol_history.pop(0)
            continue

        avg_or_vol = float(np.mean(or_vol_history))
        rel_vol    = or_vol_today / avg_or_vol if avg_or_vol > 0 else 0.0

        # Record today's OR vol for future sessions BEFORE the filter check
        or_vol_history.append(or_vol_today)
        if len(or_vol_history) > vol_window:
            or_vol_history.pop(0)

        if rel_vol < rel_vol_min:
            continue

        # ── first 5-min candle direction ────────────────────────────────────
        # Candle open = first bar's open; candle close = last bar's close
        candle_open  = or_bars.iloc[0]["open"]
        candle_close = or_bars.iloc[-1]["close"]

        or_high = or_bars["high"].max()
        or_low  = or_bars["low"].min()

        if candle_close > candle_open:
            direction: Literal["long", "short"] = "long"
            trigger   = or_high
        elif candle_close < candle_open:
            direction = "short"
            trigger   = or_low
        else:
            continue  # doji → no trade

        # ── stop loss: 10% × ATR(14) from trigger ──────────────────────────
        stop_dist = stop_atr_mult * atr14
        if direction == "long":
            stop = trigger - stop_dist
        else:
            stop = trigger + stop_dist

        # ── time of last OR bar (= candle close time) ───────────────────────
        or_end_time = or_bars.index[-1]

        # ── simulate: wait for breakout, then track stop/EOD ───────────────
        result = _simulate_breakout(day_bars, or_end_time, trigger, direction, stop)
        if result is None:
            continue  # trigger never hit

        entry_price, exit_price, exit_time, outcome, pnl_pts = result
        risk_pts = abs(entry_price - stop)
        r = pnl_pts / risk_pts if risk_pts > 0 else 0.0

        trades.append(Trade(
            date=date.strftime("%Y-%m-%d"),
            direction=direction,
            entry_time=or_end_time,   # OR close time (filled at trigger on breakout)
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            stop=stop,
            or_high=or_high,
            or_low=or_low,
            atr14=atr14,
            rel_vol=rel_vol,
            outcome=outcome,
            pnl_pts=pnl_pts,
            risk_pts=risk_pts,
            r=r,
        ))

    return trades


# ── statistics ───────────────────────────────────────────────────────────────────

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["year"] = df["entry_time"].dt.year
    return df


def compute_stats(tdf: pd.DataFrame, pts_per_dollar: float = 20.0,
                  risk_pct: float = 0.01, start_equity: float = 25_000.0) -> dict:
    """Match paper's reporting format where feasible."""
    if tdf.empty:
        return {}

    n = len(tdf)
    wins   = tdf[tdf["pnl_pts"] > 0]
    losses = tdf[tdf["pnl_pts"] < 0]
    hit_ratio = len(wins) / n

    # dollar P&L using fixed 1-contract NQ sizing
    pnl_dollars = tdf["pnl_pts"] * pts_per_dollar
    total_pnl   = pnl_dollars.sum()

    gross_profit = wins["pnl_pts"].sum()   * pts_per_dollar
    gross_loss   = losses["pnl_pts"].sum() * pts_per_dollar
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # Equity curve
    equity = start_equity + pnl_dollars.cumsum()
    peak   = equity.cummax()
    mdd    = (equity - peak).min()

    # Annualised metrics (simple approximation)
    avg_r   = float(tdf["r"].mean())
    sum_r   = float(tdf["r"].sum())
    std_r   = float(tdf["r"].std())
    sharpe  = (avg_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    # Consecutive losses
    is_loss = (tdf["pnl_pts"] <= 0).astype(int).to_numpy()
    max_consec, cur = 0, 0
    for v in is_loss:
        cur = cur + 1 if v else 0
        max_consec = max(max_consec, cur)

    return {
        "total_trades":       n,
        "hit_ratio_%":        round(hit_ratio * 100, 2),
        "avg_r":              round(avg_r, 4),
        "sum_r":              round(sum_r, 3),
        "std_r":              round(std_r, 4),
        "sharpe_r":           round(sharpe, 3),
        "total_pnl_pts":      round(float(tdf["pnl_pts"].sum()), 2),
        "total_pnl_$":        round(float(total_pnl), 2),
        "gross_profit_$":     round(float(gross_profit), 2),
        "gross_loss_$":       round(float(gross_loss), 2),
        "profit_factor":      round(float(pf), 3),
        "max_drawdown_$":     round(float(mdd), 2),
        "max_consec_losses":  int(max_consec),
        "by_direction":       tdf["direction"].value_counts().to_dict(),
        "by_outcome":         tdf["outcome"].value_counts().to_dict(),
    }


def stats_by_year(tdf: pd.DataFrame, pts_per_dollar: float = 20.0) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()

    def _ys(g: pd.DataFrame) -> pd.Series:
        wins   = g[g["pnl_pts"] > 0]
        losses = g[g["pnl_pts"] < 0]
        return pd.Series({
            "trades":    len(g),
            "hit_%":     round(len(wins) / len(g) * 100, 1),
            "avg_r":     round(g["r"].mean(), 4),
            "sum_r":     round(g["r"].sum(), 3),
            "pnl_$":     round(g["pnl_pts"].sum() * pts_per_dollar, 0),
            "PF":        round(
                abs(wins["pnl_pts"].sum() / losses["pnl_pts"].sum())
                if losses["pnl_pts"].sum() != 0 else float("inf"), 3),
        })

    return tdf.groupby("year").apply(_ys)


def stats_by_direction(tdf: pd.DataFrame, pts_per_dollar: float = 20.0) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()

    def _ds(g: pd.DataFrame) -> pd.Series:
        wins = g[g["pnl_pts"] > 0]
        return pd.Series({
            "trades": len(g),
            "hit_%":  round(len(wins) / len(g) * 100, 1),
            "avg_r":  round(g["r"].mean(), 4),
            "sum_r":  round(g["r"].sum(), 3),
            "pnl_$":  round(g["pnl_pts"].sum() * pts_per_dollar, 0),
        })

    return tdf.groupby("direction").apply(_ds)


# ── plots ────────────────────────────────────────────────────────────────────────

def plot_equity(tdf: pd.DataFrame, path: str, pts_per_dollar: float = 20.0,
                start_equity: float = 25_000.0) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    eq = start_equity + (tdf["pnl_pts"] * pts_per_dollar).cumsum()
    ax.plot(eq.values, linewidth=1.5, color="#2196F3", label="Zarattini ORB (NQ, 1 contract)")
    ax.axhline(start_equity, color="gray", linewidth=0.8, linestyle="--", label="Starting equity")
    ax.set_title("Zarattini-Barbon-Aziz 5-min ORB + RelVol — Equity Curve (NQ futures)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Account equity ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_pnl(tdf: pd.DataFrame, path: str, pts_per_dollar: float = 20.0) -> None:
    annual = tdf.groupby("year")["pnl_pts"].sum() * pts_per_dollar
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in annual.values]
    ax.bar(annual.index.astype(str), annual.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Zarattini ORB — Annual P&L ($ per NQ contract)")
    ax.set_xlabel("Year")
    ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_r_distribution(tdf: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(tdf["r"], bins=50, color="#2196F3", edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="red", linewidth=1)
    axes[0].set_title("R-multiple distribution")
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("Count")

    # R by direction
    for dir_, color in [("long", "#4CAF50"), ("short", "#F44336")]:
        sub = tdf[tdf["direction"] == dir_]["r"]
        if not sub.empty:
            axes[1].hist(sub, bins=40, alpha=0.6, label=dir_, color=color, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("R by direction (long vs short)")
    axes[1].set_xlabel("R")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rel_vol_vs_r(tdf: pd.DataFrame, path: str) -> None:
    """Reproduce Figure 4 from the paper: avg R by Relative Volume bucket."""
    rv_bins = [0, 1, 2, 3, 5, 10, 20, 30, 1000]
    rv_labels = ["<1x", "1–2x", "2–3x", "3–5x", "5–10x", "10–20x", "20–30x", "30x+"]
    tdf = tdf.copy()
    tdf["rv_bucket"] = pd.cut(tdf["rel_vol"], bins=rv_bins, labels=rv_labels, right=False)
    avg_r = tdf.groupby("rv_bucket", observed=True)["r"].mean()
    counts = tdf.groupby("rv_bucket", observed=True)["r"].count()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in avg_r.values]
    axes[0].bar(avg_r.index.astype(str), avg_r.values, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Avg R by Relative Volume (Fig. 4 equivalent)")
    axes[0].set_xlabel("Relative Volume bucket")
    axes[0].set_ylabel("Avg R-multiple")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(counts.index.astype(str), counts.values, color="#2196F3")
    axes[1].set_title("Trade count by Relative Volume bucket")
    axes[1].set_xlabel("Relative Volume bucket")
    axes[1].set_ylabel("Trades")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Zarattini-Barbon-Aziz (2024) — 5-min ORB + Relative Volume")
    print("  Applied to NQ Nasdaq Futures  |  Long & Short  |  2016–2025")
    print("=" * 70)

    print("\nLoading 1m data …")
    df1m = load_1m("1m_data.csv")
    print(f"  {len(df1m):,} bars   {df1m.index[0]}  →  {df1m.index[-1]}")

    print(f"\nRunning 5-min ORB + RelVol backtest …")
    print(f"  ATR(14) stop   : {STOP_ATR_MULT*100:.0f}% × ATR")
    print(f"  Rel-Vol filter : ≥ {REL_VOL_MIN*100:.0f}% of 14-day avg 5-min OR volume")
    print(f"  Profit target  : End of Day (EOD)")
    print(f"  Look-ahead     : NONE (ATR lag=1 day, RelVol lag=1 session)\n")

    trades = run_zarattini_orb(df1m)
    print(f"  {len(trades)} trades generated")

    if not trades:
        print("No trades found — check data or parameters.")
        return

    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "zarattini_trades.csv", index=False)

    stats = compute_stats(tdf)
    by_year = stats_by_year(tdf)
    by_dir  = stats_by_direction(tdf)

    # ── print results ───────────────────────────────────────────────────────
    print("\n=== OVERALL STATS ===")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k:<22} {v}")
        elif isinstance(v, float):
            print(f"  {k:<22} {v:>14,.4f}")
        else:
            print(f"  {k:<22} {v:>14}")

    if not by_dir.empty:
        print("\n=== BY DIRECTION ===")
        print(by_dir.to_string())

    if not by_year.empty:
        print("\n=== BY YEAR ===")
        print(by_year.to_string())

    if len(tdf) > 0:
        span = (
            pd.to_datetime(tdf["exit_time"].iloc[-1]) -
            pd.to_datetime(tdf["entry_time"].iloc[0])
        ).days
        years = span / 365.25
        if years > 0:
            print(f"\n  Trades / year : {len(tdf) / years:.1f}")
            print(f"  Trades / week : {len(tdf) / years / 52:.1f}")

    # ── R vs Relative Volume sensitivity (Figure 4 equivalent) ─────────────
    print("\n=== AVG R BY RELATIVE VOLUME BUCKET ===")
    rv_bins   = [0, 1, 2, 3, 5, 10, 20, 30, 1000]
    rv_labels = ["<1x", "1–2x", "2–3x", "3–5x", "5–10x", "10–20x", "20–30x", "30x+"]
    tdf_rv    = tdf.copy()
    tdf_rv["rv_bucket"] = pd.cut(tdf_rv["rel_vol"], bins=rv_bins, labels=rv_labels, right=False)
    rv_table  = tdf_rv.groupby("rv_bucket", observed=True)["r"].agg(
        trades="count", avg_r="mean"
    ).round(4)
    print(rv_table.to_string())

    # ── save JSON ───────────────────────────────────────────────────────────
    json_stats = {
        k: (int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating, float)) else v)
        for k, v in stats.items()
    }
    by_year.to_csv(RESULTS_DIR / "zarattini_by_year.csv")
    by_dir.to_csv(RESULTS_DIR / "zarattini_by_direction.csv")
    rv_table.to_csv(RESULTS_DIR / "zarattini_relvol_vs_r.csv")
    with open(RESULTS_DIR / "zarattini_stats.json", "w") as f:
        json.dump(json_stats, f, indent=2)

    # ── plots ───────────────────────────────────────────────────────────────
    print(f"\nGenerating charts → {RESULTS_DIR}/")
    plot_equity(tdf,            str(RESULTS_DIR / "zarattini_equity_curve.png"))
    plot_annual_pnl(tdf,        str(RESULTS_DIR / "zarattini_annual_pnl.png"))
    plot_r_distribution(tdf,    str(RESULTS_DIR / "zarattini_r_dist.png"))
    plot_rel_vol_vs_r(tdf,      str(RESULTS_DIR / "zarattini_relvol_vs_r.png"))

    print(f"\nAll results written to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
