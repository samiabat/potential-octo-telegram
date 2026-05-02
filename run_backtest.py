"""Entry point: loads data, runs Silver Bullet v2, writes stats + plots."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from silver_bullet.data_loader import load_5m, resample
from silver_bullet.strategy import run_silver_bullet
from silver_bullet.backtest import (
    trades_to_df, compute_stats, plot_equity, plot_killzone_breakdown,
)

DATA_PATH   = "5m_data.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RISK_PER_TRADE = 100.0
RR = 2.0


def main() -> None:
    print("Loading 5m data...")
    df5 = load_5m(DATA_PATH)
    print(f"  {len(df5):,} bars  {df5.index[0]} -> {df5.index[-1]}")

    print("Resampling to 1H for HTF bias...")
    df1h = resample(df5, "1h")

    print("Running Silver Bullet v2 (killzone + quality FVG)...")
    # Config selected from sweep_configs.py — bias/sweep/MSS gates were
    # found to FILTER OUT winners on this dataset. The simple model wins.
    trades, diag = run_silver_bullet(
        df5, df1h,
        rr=RR,
        bias_filter="off",          # EMA-cross bias hurts; needs proper HTF DOL
        require_sweep=False,        # liquidity-sweep gate hurts; reconsider when encoded properly
        require_mss=False,
        min_fvg_size_atr=0.3,       # FVG must be at least 0.3 ATR(14) wide
        only_first_half_kz=True,    # entries only in first 30 mins of killzone
        max_trades_per_kz=1,
        fvg_max_age_bars=12,
        trade_expiry_bars=24,
    )
    print(f"  {len(trades)} trades generated")
    print("\n--- Diagnostics ---")
    for k, v in diag.__dict__.items():
        print(f"  {k:<22} {v:>10,}")

    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "trades.csv", index=False)

    stats = compute_stats(tdf, risk_per_trade=RISK_PER_TRADE)
    by_kz = stats.pop("by_killzone", None)

    print("\n=== STATS ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<16} {v:>14,.2f}")
        else:
            print(f"  {k:<16} {v:>14}")
    if by_kz is not None:
        print("\nBy killzone:")
        print(by_kz.to_string())

    # Trades per year
    if not tdf.empty:
        years = (pd.to_datetime(tdf['entry_time'].iloc[-1]) -
                 pd.to_datetime(tdf['entry_time'].iloc[0])).days / 365.25
        print(f"\nTrades/year: {len(tdf)/years:.1f}    Trades/week: {len(tdf)/years/52:.1f}")

    with open(RESULTS_DIR / "stats.json", "w") as f:
        out = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in stats.items()}
        out["diagnostics"] = diag.__dict__
        json.dump(out, f, indent=2)
    if by_kz is not None:
        by_kz.to_csv(RESULTS_DIR / "stats_by_killzone.csv")

    plot_equity(tdf, str(RESULTS_DIR / "equity_curve.png"),
                risk_per_trade=RISK_PER_TRADE)
    plot_killzone_breakdown(tdf, str(RESULTS_DIR / "killzone_breakdown.png"),
                             risk_per_trade=RISK_PER_TRADE)

    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
