"""Entry point: loads data, runs Silver Bullet, writes stats + plots."""
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

RISK_PER_TRADE = 100.0   # $ per 1R, with $10k starting equity
RR = 2.0


def main() -> None:
    print("Loading 5m data...")
    df5 = load_5m(DATA_PATH)
    print(f"  {len(df5):,} bars  {df5.index[0]} -> {df5.index[-1]}")

    print("Resampling to 1H for HTF bias...")
    df1h = resample(df5, "1h")

    print("Running Silver Bullet strategy...")
    trades = run_silver_bullet(df5, df1h, rr=RR)
    print(f"  {len(trades)} trades generated")

    tdf = trades_to_df(trades)
    tdf.to_csv(RESULTS_DIR / "trades.csv", index=False)

    stats = compute_stats(tdf, risk_per_trade=RISK_PER_TRADE)
    by_kz = stats.pop("by_killzone", None)

    print("\n=== STATS ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<16} {v:>12,.2f}")
        else:
            print(f"  {k:<16} {v:>12}")
    if by_kz is not None:
        print("\nBy killzone:")
        print(by_kz.to_string())

    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump({k: (v if not isinstance(v, float) else round(v, 4))
                   for k, v in stats.items()}, f, indent=2)
    if by_kz is not None:
        by_kz.to_csv(RESULTS_DIR / "stats_by_killzone.csv")

    plot_equity(tdf, str(RESULTS_DIR / "equity_curve.png"),
                risk_per_trade=RISK_PER_TRADE)
    plot_killzone_breakdown(tdf, str(RESULTS_DIR / "killzone_breakdown.png"),
                             risk_per_trade=RISK_PER_TRADE)

    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
