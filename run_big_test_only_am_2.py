"""Same as run_big_test_only_am.py but uses 1m_data_2.csv (the smaller
in-repo sample, 2023-06 → 2024-07). Created so the leak-fix run could be
reproduced in environments without LFS access to the full 1m_data.csv.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from model_2022.data_loader import load_1m
from model_2022.strategy import run_2022_model
from model_2022.backtest import (
    trades_to_df, compute_stats, plot_equity, plot_breakdown, plot_monthly,
)

DATA_PATH   = "1m_data_2.csv"
RESULTS_DIR = Path("big_test_only_am")
RESULTS_DIR.mkdir(exist_ok=True)

RISK_PER_TRADE = 100.0
RR             = 2.0


def per_year(tdf):
    if tdf.empty: return pd.DataFrame()
    t = tdf.copy()
    t["year"] = pd.to_datetime(t["entry_time"], utc=True).dt.year
    return t.groupby("year")["r"].agg([
        ("trades", "count"), ("mean_R", "mean"),
        ("sum_R", "sum"), ("win_%", lambda x: (x > 0).mean() * 100),
    ]).round(3)


def main():
    print(f"Loading 1m data from {DATA_PATH} …")
    df1m = load_1m(DATA_PATH)
    print(f"  {len(df1m):,} bars   {df1m.index[0]}  →  {df1m.index[-1]}")

    trades, diag = run_2022_model(
        df1m, rr=RR,
        ema_fast=20, ema_slow=50,
        sweep_window_15m=20, sweep_max_age=pd.Timedelta(hours=4),
        mss_lookback_15m=50, arm_window=pd.Timedelta(hours=2),
        min_fvg_size_atr=0.2, fvg_max_age_bars=30,
        trade_expiry_bars=120, max_trades_per_arm=1,
        cost_per_side_pts=1.0, min_stop_pts=5.0, min_stop_atr_mult=1.0,
        use_ob=False, use_fvg=True,
        use_killzones=True, allowed_killzones={"am"},
    )
    print(f"  {len(trades)} trades generated")

    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "trades.csv", index=False)
    stats = compute_stats(tdf, risk_per_trade=RISK_PER_TRADE)
    for k, v in stats.items():
        if k.startswith("by_"): continue
        print(f"  {k:<22} {v}")
    py = per_year(tdf)
    if not py.empty:
        print(py.to_string()); py.to_csv(RESULTS_DIR / "stats_by_year.csv")

    out_json = {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in stats.items() if not k.startswith("by_")}
    out_json["diagnostics"] = diag.__dict__
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump(out_json, f, indent=2)

    plot_equity(tdf, str(RESULTS_DIR / "equity_curve.png"), risk_per_trade=RISK_PER_TRADE)
    plot_breakdown(tdf, str(RESULTS_DIR / "breakdown.png"), risk_per_trade=RISK_PER_TRADE)
    plot_monthly(tdf, str(RESULTS_DIR / "monthly_pnl.png"), risk_per_trade=RISK_PER_TRADE)


if __name__ == "__main__":
    main()
