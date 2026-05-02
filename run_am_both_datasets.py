"""AM-session-only backtest on BOTH 1m datasets.

Runs the exact same ICT 2022 Model strategy on:
  • 1m_data.csv   → results_am_ds1/
  • 1m_data_2.csv → results_am_ds2/

Changes vs. previous runs:
  • Only AM killzone (09:30-12:00 ET) is allowed — London & PM excluded
  • Per-trade candlestick charts generated for every trade
  • Weekly, monthly, and yearly PnL charts + breakdown CSV files

Old chart folders are cleaned before each run.

Usage:
    python3 run_am_both_datasets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from model_2022.data_loader import load_1m
from model_2022.strategy import run_2022_model
from model_2022.backtest import (
    trades_to_df,
    compute_stats,
    compute_period_stats,
    plot_equity,
    plot_breakdown,
    plot_monthly,
    plot_weekly,
    plot_yearly,
    plot_all_trade_charts,
)

RISK_PER_TRADE = 100.0
RR             = 2.0

# Datasets: (csv_file, results_dir, label)
DATASETS = [
    ("1m_data.csv",   Path("results_am_ds1"), "Dataset-1 (2020-2025)  AM-only"),
    ("1m_data_2.csv", Path("results_am_ds2"), "Dataset-2 (2023-2024)  AM-only"),
]


def _clean_charts(charts_dir: Path) -> None:
    """Remove and recreate the per-trade chart folder."""
    if charts_dir.exists():
        shutil.rmtree(charts_dir)
        print(f"  Cleaned old charts: {charts_dir}")
    charts_dir.mkdir(parents=True, exist_ok=True)


def run_one(csv_path: str, out_dir: Path, label: str) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)

    out_dir.mkdir(exist_ok=True)
    charts_dir = out_dir / "trade_charts"
    _clean_charts(charts_dir)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading {csv_path} ...")
    df1m = load_1m(csv_path)
    print(f"  {len(df1m):,} bars   {df1m.index[0]}  →  {df1m.index[-1]}")

    # ------------------------------------------------------------------
    # Run model — AM session only, all other params identical
    # ------------------------------------------------------------------
    print("Running model (AM session only) ...")
    trades, diag = run_2022_model(
        df1m,
        rr=RR,
        ema_fast=20,
        ema_slow=50,
        sweep_window_15m=20,
        sweep_max_age=pd.Timedelta(hours=4),
        mss_lookback_15m=50,
        arm_window=pd.Timedelta(hours=2),
        min_fvg_size_atr=0.2,
        fvg_max_age_bars=30,
        trade_expiry_bars=120,
        max_trades_per_arm=1,
        cost_per_side_pts=1.0,
        min_stop_pts=5.0,
        min_stop_atr_mult=1.0,
        use_ob=False,
        use_fvg=True,
        use_killzones=True,
        allowed_killzones={"am"},          # ← AM session only
    )
    print(f"  {len(trades)} AM trades generated")

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
    tdf.to_csv(out_dir / "trades.csv", index=False)

    stats   = compute_stats(tdf, risk_per_trade=RISK_PER_TRADE)
    by_kz   = stats.pop("by_killzone",   None)
    by_type = stats.pop("by_entry_type", None)
    by_dir  = stats.pop("by_direction",  None)

    print("\n=== STATS (AM only) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:>14,.2f}")
        else:
            print(f"  {k:<18} {v:>14}")

    if by_dir is not None:
        print("\nBy direction:")
        print(by_dir.to_string())

    # Trades-per-period cadence
    if not tdf.empty:
        span_days = (
            pd.to_datetime(tdf["exit_time"].iloc[-1]) -
            pd.to_datetime(tdf["entry_time"].iloc[0])
        ).days
        years = span_days / 365.25
        if years > 0:
            print(f"\n  Trades / year : {len(tdf) / years:.1f}")
            print(f"  Trades / week : {len(tdf) / years / 52:.1f}")

    # ------------------------------------------------------------------
    # Period breakdown (weekly / monthly / yearly)
    # ------------------------------------------------------------------
    periods = compute_period_stats(tdf, risk_per_trade=RISK_PER_TRADE)

    print("\n--- Weekly summary (first 10 rows) ---")
    print(periods["weekly"].head(10).to_string())

    print("\n--- Monthly summary ---")
    print(periods["monthly"].to_string())

    print("\n--- Yearly summary ---")
    print(periods["yearly"].to_string())

    periods["weekly"].to_csv(out_dir / "stats_by_week.csv")
    periods["monthly"].to_csv(out_dir / "stats_by_month.csv")
    periods["yearly"].to_csv(out_dir / "stats_by_year.csv")

    # ------------------------------------------------------------------
    # Save JSON stats
    # ------------------------------------------------------------------
    out_json: dict = {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in stats.items()
    }
    out_json["diagnostics"] = diag.__dict__
    with open(out_dir / "stats.json", "w") as f:
        json.dump(out_json, f, indent=2)

    if by_kz   is not None: by_kz.to_csv(out_dir / "stats_by_killzone.csv")
    if by_type is not None: by_type.to_csv(out_dir / "stats_by_entry_type.csv")
    if by_dir  is not None: by_dir.to_csv(out_dir / "stats_by_direction.csv")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print(f"\nGenerating charts → {out_dir}/")
    plot_equity(tdf, str(out_dir / "equity_curve.png"),
                risk_per_trade=RISK_PER_TRADE)
    plot_breakdown(tdf, str(out_dir / "breakdown.png"),
                   risk_per_trade=RISK_PER_TRADE)
    plot_monthly(tdf, str(out_dir / "monthly_pnl.png"),
                 risk_per_trade=RISK_PER_TRADE, title_suffix=f"{label}")
    plot_weekly(tdf, str(out_dir / "weekly_pnl.png"),
                risk_per_trade=RISK_PER_TRADE, title_suffix=f"{label}")
    plot_yearly(tdf, str(out_dir / "yearly_pnl.png"),
                risk_per_trade=RISK_PER_TRADE, title_suffix=f"{label}")

    # ------------------------------------------------------------------
    # Per-trade charts
    # ------------------------------------------------------------------
    print(f"\nGenerating per-trade candlestick charts → {charts_dir}/")
    plot_all_trade_charts(
        trades, df1m, charts_dir,
        context_bars_before=60,
        context_bars_after=15,
    )

    return tdf


def main() -> None:
    results: dict[str, pd.DataFrame] = {}
    for csv_path, out_dir, label in DATASETS:
        tdf = run_one(csv_path, out_dir, label)
        results[label] = tdf

    # ------------------------------------------------------------------
    # Cross-dataset comparison summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("  CROSS-DATASET COMPARISON  (AM session only)")
    print("=" * 65)
    rows = []
    for label, tdf in results.items():
        if tdf.empty:
            continue
        r = tdf["r"].values
        pnl = r * RISK_PER_TRADE
        eq  = pnl.cumsum() + 10_000
        peak = __import__("numpy").maximum.accumulate(eq)
        rows.append({
            "Dataset":       label,
            "Trades":        len(tdf),
            "Win%":          round(float((r > 0).mean() * 100), 1),
            "E[R]":          round(float(r.mean()), 3),
            "Total R":       round(float(r.sum()), 1),
            "Net PnL $":     round(float(pnl.sum()), 0),
            "PF":            round(float(pnl[pnl > 0].sum() / max(-pnl[pnl < 0].sum(), 1e-9)), 2),
            "Max DD $":      round(float((eq - peak).min()), 0),
            "Max DD %":      round(float(((eq - peak) / peak).min() * 100), 1),
        })
    if rows:
        cmp = pd.DataFrame(rows).set_index("Dataset")
        print(cmp.to_string())

    print("\nAll done.")
    for _, out_dir, _ in DATASETS:
        charts = list((out_dir / "trade_charts").glob("*.png"))
        print(f"  {out_dir}/  — {len(charts)} trade charts")


if __name__ == "__main__":
    main()
