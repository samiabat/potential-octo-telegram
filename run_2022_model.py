"""Entry point — ICT 2022 Model backtest on the 1m NAS dataset.

Cascade (all gates required):
  HTF bias (4H EMA cross)
  → Liquidity sweep on 15m
  → Market Structure Shift (CHoCH) on 15m
  → FVG / Order Block entry on 1m

Usage:
    pip install pandas numpy matplotlib pytz
    python3 run_2022_model.py
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from model_2022.data_loader import load_1m
from model_2022.strategy import run_2022_model
from model_2022.backtest import (
    trades_to_df,
    compute_stats,
    plot_equity,
    plot_breakdown,
    plot_monthly,
)

DATA_PATH   = "1m_data.csv"
RESULTS_DIR = Path("results_2022_model")
RESULTS_DIR.mkdir(exist_ok=True)

RISK_PER_TRADE = 100.0   # $ risk per trade (1R)
RR             = 2.0     # reward-to-risk


def main() -> None:
    print("=" * 60)
    print(" ICT 2022 Model — 4H→15m→1m Cascade Backtest")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load 1m data
    # ------------------------------------------------------------------
    print(f"\nLoading 1m data from {DATA_PATH} ...")
    df1m = load_1m(DATA_PATH)
    print(f"  {len(df1m):,} bars   {df1m.index[0]}  →  {df1m.index[-1]}")

    # ------------------------------------------------------------------
    # Run the model
    # ------------------------------------------------------------------
    print("\nRunning 2022 Model (HTF bias → 15m sweep → 15m MSS → 1m entry)...")
    trades, diag = run_2022_model(
        df1m,
        rr=RR,
        # HTF (4H) EMA parameters
        ema_fast=20,
        ema_slow=50,
        # 15m sweep / MSS
        sweep_window_15m=20,
        sweep_max_age=pd.Timedelta(hours=4),   # tighter sweep recency
        mss_lookback_15m=50,
        arm_window=pd.Timedelta(hours=2),       # 2h entry window after MSS
        # 1m entry
        min_fvg_size_atr=0.2,
        fvg_max_age_bars=30,
        trade_expiry_bars=120,
        max_trades_per_arm=1,
        cost_per_side_pts=1.0,
        min_stop_pts=5.0,
        min_stop_atr_mult=1.0,
        use_ob=False,          # FVG-only; OB entries hurt on this dataset
        use_fvg=True,
        use_killzones=True,    # London, AM, PM session filter
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

    print("\n=== STATS ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:>14,.2f}")
        else:
            print(f"  {k:<18} {v:>14}")

    if by_kz is not None:
        print("\nBy killzone (session):")
        print(by_kz.to_string())

    if by_type is not None:
        print("\nBy entry type (FVG vs OB):")
        print(by_type.to_string())

    if by_dir is not None:
        print("\nBy direction (long vs short):")
        print(by_dir.to_string())

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
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
