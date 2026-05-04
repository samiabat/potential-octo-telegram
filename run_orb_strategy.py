"""run_orb_strategy.py – Opening Range Breakout (ORB) Strategy Backtest

Rules (tradethatswing.com article, with March 2026 update):

  Opening Range : first 15 min of RTH  (09:30–09:44 ET, inclusive)
  Signal        : 5-min candle CLOSE above OR high  →  go long
                  5-min candle CLOSE below OR low   →  day invalidated (long-only)
  Filters       : OR size ≤ MAX_OR_PCT of day open; skip Thursdays
  Entry         : close of the triggering 5-min candle (no look-ahead)
  Stop          : OR low, but capped so risk ≤ MAX_RISK_PTS NQ points
  Target        : entry + TARGET_FRAC × OR range
  Limit         : one trade per day; EOD (SESSION_END) expiry

No look-ahead guarantee
  - OR is computed exclusively from 1m bars that close before 09:45.
  - The first eligible signal bar is the 5m candle [09:45, 09:49].
  - Entry price = close of signal bar (simulated on bars that open after the
    signal bar closes, i.e. strictly after bar_time + 4 minutes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_2022.data_loader import load_1m

# ── output directory ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("orb_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── strategy parameters ────────────────────────────────────────────────────────
MAX_OR_PCT    = 0.008   # skip day if OR > 0.8 % of open
MAX_RISK_PTS  = 50.0    # 50 NQ pts = $1 000/contract (1 pt = $20)
TARGET_FRAC   = 0.50    # target = entry + 50 % × OR range
SKIP_THURSDAY = True    # March-2026 update: Thursdays excluded
OR_START_T    = pd.Timestamp("09:30").time()
OR_END_T      = pd.Timestamp("09:44").time()
SESSION_END_T = pd.Timestamp("15:59").time()
MIN_OR_BARS   = 10      # require ≥ 10 of 15 OR bars to have data


# ── data structures ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    date:        str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    stop:        float
    target:      float
    or_high:     float
    or_low:      float
    outcome:     str    # 'target' | 'stop' | 'expiry'
    pnl_pts:     float  # NQ price-point P&L
    risk_pts:    float  # actual risk (entry – stop)
    r:           float  # R-multiple (pnl_pts / risk_pts)


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_signal_5m(day_1m: pd.DataFrame) -> pd.DataFrame:
    """Return 5-min OHLC bars for the post-OR window (≥ 09:45 ET)."""
    t = day_1m.index.time
    mask = (t >= pd.Timestamp("09:45").time()) & (t <= SESSION_END_T)
    signal_1m = day_1m[mask]
    if signal_1m.empty:
        return pd.DataFrame()
    return (
        signal_1m
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _simulate(
    sim_1m: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
) -> tuple[float, pd.Timestamp, str]:
    """Walk forward bar-by-bar on 1m data using vectorised numpy.

    Stop is checked before target on the same bar (pessimistic fill).
    Returns (exit_price, exit_time, outcome).
    """
    if sim_1m.empty:
        return entry, pd.NaT, "expiry"

    lows   = sim_1m["low"].to_numpy()
    highs  = sim_1m["high"].to_numpy()
    times  = sim_1m.index

    stop_hits   = np.where(lows  <= stop)[0]
    target_hits = np.where(highs >= target)[0]

    first_stop   = stop_hits[0]   if len(stop_hits)   else len(lows)
    first_target = target_hits[0] if len(target_hits) else len(lows)

    if first_stop <= first_target and first_stop < len(lows):
        return stop, times[first_stop], "stop"
    if first_target < len(lows):
        return target, times[first_target], "target"

    last = sim_1m.iloc[-1]
    return last["close"], times[-1], "expiry"


# ── core backtest ──────────────────────────────────────────────────────────────

def run_orb(df1m: pd.DataFrame) -> List[Trade]:
    """Run the ORB strategy on a 1-minute OHLCV DataFrame (NY tz).

    The DataFrame must be sorted ascending and tz-aware (America/New_York).
    No look-ahead: every decision uses only bars that have fully closed.
    """
    trades: List[Trade] = []

    # Group once by calendar date for O(n) day iteration
    grouped = df1m.groupby(df1m.index.normalize())

    for date, day_bars in grouped:
        if day_bars.empty:
            continue

        # Skip Thursday (weekday == 3)
        if SKIP_THURSDAY and date.weekday() == 3:
            continue

        # ── Opening Range (09:30–09:44 ET) ────────────────────────────────
        t = day_bars.index.time
        or_mask = (t >= OR_START_T) & (t <= OR_END_T)
        or_bars = day_bars[or_mask]

        if len(or_bars) < MIN_OR_BARS:
            continue

        or_high  = or_bars["high"].max()
        or_low   = or_bars["low"].min()
        or_open  = or_bars.iloc[0]["open"]
        or_range = or_high - or_low

        if or_range <= 0:
            continue

        # Filter: OR must be ≤ 0.8 % of open
        if or_range / or_open > MAX_OR_PCT:
            continue

        # ── Signal detection on 5-min candles (starting 09:45) ────────────
        signal_5m = _build_signal_5m(day_bars)
        if signal_5m.empty:
            continue

        for bar_time, bar in signal_5m.iterrows():
            # The 5-min bar [bar_time, bar_time+4min] is now closed.
            # Entry occurs at bar close; simulation uses strictly later bars.
            entry_cutoff = bar_time + pd.Timedelta(minutes=4)
            sim_1m = day_bars[
                (day_bars.index > entry_cutoff) &
                (day_bars.index.time <= SESSION_END_T)
            ]

            if bar["close"] > or_high:
                # ── Long breakout ──────────────────────────────────────────
                entry    = bar["close"]
                stop     = max(or_low, entry - MAX_RISK_PTS)
                target   = entry + TARGET_FRAC * or_range
                risk_pts = entry - stop

                if risk_pts <= 0:
                    break

                exit_px, exit_t, outcome = _simulate(sim_1m, entry, stop, target)
                pnl_pts = exit_px - entry
                r = pnl_pts / risk_pts

                trades.append(Trade(
                    date=date.strftime("%Y-%m-%d"),
                    entry_time=bar_time,
                    exit_time=exit_t,
                    entry_price=entry,
                    exit_price=exit_px,
                    stop=stop,
                    target=target,
                    or_high=or_high,
                    or_low=or_low,
                    outcome=outcome,
                    pnl_pts=pnl_pts,
                    risk_pts=risk_pts,
                    r=r,
                ))
                break  # one trade per day

            elif bar["close"] < or_low:
                # Downside break first → skip rest of day (long-only)
                break

    return trades


# ── statistics ─────────────────────────────────────────────────────────────────

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["year"] = df["entry_time"].dt.year
    return df


def compute_stats(tdf: pd.DataFrame, pts_per_dollar: float = 20.0) -> dict:
    """Compute summary statistics.

    NQ: 1 point = $20; MAX_RISK_PTS = 50 pts = $1 000.
    """
    if tdf.empty:
        return {}

    n       = len(tdf)
    wins    = tdf[tdf["pnl_pts"] > 0]
    losses  = tdf[tdf["pnl_pts"] < 0]
    win_pct = len(wins) / n * 100

    avg_win_pts  = wins["pnl_pts"].mean()  if len(wins)   else 0.0
    avg_loss_pts = losses["pnl_pts"].mean() if len(losses) else 0.0

    total_pts     = tdf["pnl_pts"].sum()
    total_dollars = total_pts * pts_per_dollar

    gross_profit = wins["pnl_pts"].sum()   * pts_per_dollar
    gross_loss   = losses["pnl_pts"].sum() * pts_per_dollar
    profit_factor = (
        abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    )

    avg_r   = tdf["r"].mean()
    sum_r   = tdf["r"].sum()

    # Equity curve (dollar) for drawdown calculation
    equity = (tdf["pnl_pts"] * pts_per_dollar).cumsum()
    peak   = equity.cummax()
    dd     = equity - peak
    max_dd = dd.min()

    # Consecutive losses
    is_loss = (tdf["pnl_pts"] <= 0).astype(int)
    max_consec_loss = 0
    cur = 0
    for v in is_loss:
        cur = cur + 1 if v else 0
        max_consec_loss = max(max_consec_loss, cur)

    return {
        "total_trades":        n,
        "win_pct":             round(win_pct, 2),
        "avg_win_pts":         round(avg_win_pts,  2),
        "avg_loss_pts":        round(avg_loss_pts, 2),
        "avg_r":               round(avg_r, 3),
        "sum_r":               round(sum_r, 3),
        "total_pts":           round(total_pts, 2),
        "total_dollars":       round(total_dollars, 2),
        "gross_profit_$":      round(gross_profit, 2),
        "gross_loss_$":        round(gross_loss, 2),
        "profit_factor":       round(profit_factor, 3),
        "max_drawdown_$":      round(max_dd, 2),
        "max_consec_losses":   max_consec_loss,
        "by_outcome":          tdf["outcome"].value_counts().to_dict(),
    }


def stats_by_year(tdf: pd.DataFrame, pts_per_dollar: float = 20.0) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()

    def year_stats(g):
        wins = g[g["pnl_pts"] > 0]
        losses = g[g["pnl_pts"] < 0]
        return pd.Series({
            "trades":    len(g),
            "win_%":     round(len(wins) / len(g) * 100, 1),
            "avg_r":     round(g["r"].mean(), 3),
            "sum_r":     round(g["r"].sum(), 3),
            "total_$":   round(g["pnl_pts"].sum() * pts_per_dollar, 0),
            "profit_factor": round(
                abs(wins["pnl_pts"].sum() / losses["pnl_pts"].sum())
                if losses["pnl_pts"].sum() != 0 else float("inf"), 3
            ),
        })

    return tdf.groupby("year").apply(year_stats)


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_equity(tdf: pd.DataFrame, path: str, pts_per_dollar: float = 20.0,
                starting_equity: float = 10_000.0) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    equity = starting_equity + (tdf["pnl_pts"] * pts_per_dollar).cumsum()
    ax.plot(equity.values, linewidth=1.5, color="#2196F3")
    ax.axhline(starting_equity, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("ORB Strategy – Equity Curve (1 NQ contract, $10k start)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Account equity ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_pnl(tdf: pd.DataFrame, path: str, pts_per_dollar: float = 20.0) -> None:
    annual = tdf.groupby("year")["pnl_pts"].sum() * pts_per_dollar
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in annual.values]
    ax.bar(annual.index.astype(str), annual.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("ORB Strategy – Annual P&L ($)")
    ax.set_xlabel("Year")
    ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcome_distribution(tdf: pd.DataFrame, path: str) -> None:
    outcome_r = tdf.groupby("outcome")["r"].mean()
    counts = tdf["outcome"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.index, counts.values, color=["#4CAF50", "#F44336", "#FF9800"])
    axes[0].set_title("Trade Count by Outcome")
    axes[0].set_ylabel("Count")
    axes[1].bar(outcome_r.index, outcome_r.values, color=["#4CAF50", "#F44336", "#FF9800"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Avg R-Multiple by Outcome")
    axes[1].set_ylabel("Avg R")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Opening Range Breakout (ORB) Strategy — NQ Futures")
    print("  Long-only | OR ≤ 0.8% | Skip Thu | Cap 50 pts | 1 trade/day")
    print("=" * 65)

    print(f"\nLoading 1m data …")
    df1m = load_1m("1m_data.csv")
    print(f"  {len(df1m):,} bars  {df1m.index[0]}  →  {df1m.index[-1]}")

    print("\nRunning ORB backtest (no look-ahead) …")
    trades = run_orb(df1m)
    print(f"  {len(trades)} trades generated")

    if not trades:
        print("No trades generated – check data or parameters.")
        return

    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "orb_trades.csv", index=False)

    stats = compute_stats(tdf)
    by_year = stats_by_year(tdf)

    # ── print summary ──────────────────────────────────────────────────────
    print("\n=== OVERALL STATS (full dataset, long-only ORB) ===")
    for k, v in stats.items():
        if k == "by_outcome":
            print(f"  {'by_outcome':<22} {v}")
        elif isinstance(v, float):
            print(f"  {k:<22} {v:>14,.2f}")
        else:
            print(f"  {k:<22} {v:>14}")

    if not by_year.empty:
        print("\n=== BY YEAR ===")
        print(by_year.to_string())
        by_year.to_csv(RESULTS_DIR / "orb_stats_by_year.csv")

    if len(tdf) > 0:
        span_days = (
            pd.to_datetime(tdf["exit_time"].iloc[-1]) -
            pd.to_datetime(tdf["entry_time"].iloc[0])
        ).days
        years = span_days / 365.25
        if years > 0:
            print(f"\n  Trades / year : {len(tdf) / years:.1f}")
            print(f"  Trades / week : {len(tdf) / years / 52:.1f}")

    # ── save JSON ──────────────────────────────────────────────────────────
    json_stats = {
        k: (list(v) if isinstance(v, (pd.Index,)) else
            int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating, float)) else v)
        for k, v in stats.items()
    }
    with open(RESULTS_DIR / "orb_stats.json", "w") as f:
        json.dump(json_stats, f, indent=2)

    # ── plots ──────────────────────────────────────────────────────────────
    print(f"\nGenerating charts → {RESULTS_DIR}/")
    plot_equity(tdf, str(RESULTS_DIR / "orb_equity_curve.png"))
    plot_annual_pnl(tdf, str(RESULTS_DIR / "orb_annual_pnl.png"))
    plot_outcome_distribution(tdf, str(RESULTS_DIR / "orb_outcome_dist.png"))

    print(f"\nAll results written to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
