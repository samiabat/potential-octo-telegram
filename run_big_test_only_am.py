"""2022 Model backtest — full 1m_data.csv (2016-2025), AM session only.

Runs the ICT 2022 Model on the full dataset with the AM killzone filter
(08:00–12:00 ET) and writes all results to big_test_only_am/.

Usage:
    python3 run_big_test_only_am.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from model_2022.data_loader import load_1m
from model_2022.strategy import run_2022_model
from model_2022.backtest import (
    trades_to_df,
    compute_stats,
    plot_equity,
    plot_breakdown,
    plot_monthly,
    plot_all_trade_charts,
    plot_trade_chart_5m,
    plot_trade_chart_15m,
)

DATA_PATH   = "1m_data.csv"
RESULTS_DIR = Path("big_test_only_am")
RESULTS_DIR.mkdir(exist_ok=True)

RISK_PER_TRADE = 100.0
RR             = 2.0
START_EQUITY   = 10_000.0


def per_year(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()
    t = tdf.copy()
    t["year"] = pd.to_datetime(t["entry_time"], utc=True).dt.year
    return t.groupby("year")["r"].agg([
        ("trades", "count"),
        ("mean_R", "mean"),
        ("sum_R",  "sum"),
        ("win_%",  lambda x: (x > 0).mean() * 100),
    ]).round(3)


def main() -> None:
    print("=" * 65)
    print(" ICT 2022 Model — Full Dataset (2016-2025)  |  AM Session Only")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"\nLoading 1m data from {DATA_PATH} …")
    df1m = load_1m(DATA_PATH)
    print(f"  {len(df1m):,} bars   {df1m.index[0]}  →  {df1m.index[-1]}")

    # ------------------------------------------------------------------
    # Run model — AM session only (08:00–12:00 ET)
    # ------------------------------------------------------------------
    print("\nRunning 2022 Model (AM killzone: 08:00–12:00 ET, 1m MSS) …")
    trades, diag = run_2022_model(
        df1m,
        rr=RR,
        ema_fast=20,
        ema_slow=50,
        sweep_window_15m=20,
        sweep_max_age=pd.Timedelta(hours=4),
        mss_tf="1m",                              # use 1m MSS for faster confirmation
        mss_lookback=50,
        arm_window=pd.Timedelta(minutes=90),      # 90 min window after MSS
        max_sweep_to_mss_atr=4.0,                 # skip if move already > 4×ATR
        min_fvg_size_atr=0.2,
        fvg_max_age_bars=20,                      # 20 min max wait for retrace
        trade_expiry_bars=120,
        max_trades_per_arm=1,
        cost_per_side_pts=1.0,
        min_stop_pts=5.0,
        min_stop_atr_mult=1.0,
        use_ob=False,
        use_fvg=True,
        use_killzones=True,
        allowed_killzones={"am"},
    )
    print(f"  {len(trades)} trades generated")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print("\n--- Cascade Diagnostics ---")
    for k, v in diag.__dict__.items():
        print(f"  {k:<26} {v:>10,}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "trades.csv", index=False)

    stats = compute_stats(tdf, risk_per_trade=RISK_PER_TRADE)
    by_kz   = stats.pop("by_killzone",   None)
    by_type = stats.pop("by_entry_type", None)
    by_dir  = stats.pop("by_direction",  None)

    print("\n=== OVERALL STATS (AM only, 2016-2025) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:>14,.2f}")
        else:
            print(f"  {k:<22} {v:>14}")

    if by_dir is not None:
        print("\nBy direction (long vs short):")
        print(by_dir.to_string())

    # ------------------------------------------------------------------
    # Per-year breakdown
    # ------------------------------------------------------------------
    py = per_year(tdf)
    if not py.empty:
        print("\n=== BY YEAR ===")
        print(py.to_string())
        py.to_csv(RESULTS_DIR / "stats_by_year.csv")

    # Trades per year / week
    if not tdf.empty:
        years = (
            pd.to_datetime(tdf["exit_time"].iloc[-1]) -
            pd.to_datetime(tdf["entry_time"].iloc[0])
        ).days / 365.25
        if years > 0:
            print(f"\n  Trades / year : {len(tdf) / years:.1f}")
            print(f"  Trades / week : {len(tdf) / years / 52:.1f}")

    # ------------------------------------------------------------------
    # Save JSON stats
    # ------------------------------------------------------------------
    out_json: dict = {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in stats.items()
    }
    out_json["diagnostics"] = diag.__dict__
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump(out_json, f, indent=2)

    if by_kz is not None:
        by_kz.to_csv(RESULTS_DIR / "stats_by_killzone.csv")
    if by_type is not None:
        by_type.to_csv(RESULTS_DIR / "stats_by_entry_type.csv")
    if by_dir is not None:
        by_dir.to_csv(RESULTS_DIR / "stats_by_direction.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print(f"\nGenerating charts → {RESULTS_DIR}/")
    plot_equity(tdf, str(RESULTS_DIR / "equity_curve.png"),
                risk_per_trade=RISK_PER_TRADE)
    plot_breakdown(tdf, str(RESULTS_DIR / "breakdown.png"),
                   risk_per_trade=RISK_PER_TRADE)
    plot_monthly(tdf, str(RESULTS_DIR / "monthly_pnl.png"),
                 risk_per_trade=RISK_PER_TRADE)

    print("Done.")
    print(f"\nAll results written to {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # Per-trade charts for 2020+ trades (3 charts each in own subfolder)
    # ------------------------------------------------------------------
    CHART_START = pd.Timestamp("2020-01-01", tz="America/New_York")
    trades_2020 = [t for t in trades if t.entry_time >= CHART_START]

    chart_dir = RESULTS_DIR / "trade_charts_2020"
    # Clear old charts before regenerating
    if chart_dir.exists():
        shutil.rmtree(chart_dir)
        print(f"\nCleared old charts from {chart_dir}/")

    if trades_2020:
        print(f"\nGenerating per-trade charts for {len(trades_2020)} "
              f"2020+ trades → {chart_dir}/")
        plot_all_trade_charts(
            trades_2020,
            df1m,
            chart_dir,
            context_bars_before=80,
            context_bars_after=20,
        )
    else:
        print("\nNo 2020+ trades found; skipping per-trade chart generation.")


if __name__ == "__main__":
    main()
