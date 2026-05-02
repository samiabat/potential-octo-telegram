"""Sensitivity sweep: try several Silver Bullet configurations and compare."""
from __future__ import annotations
import pandas as pd

from silver_bullet.data_loader import load_5m, resample
from silver_bullet.strategy import run_silver_bullet
from silver_bullet.backtest import trades_to_df, compute_stats


def years_span(tdf):
    if tdf.empty: return 0
    return (pd.to_datetime(tdf['entry_time'].iloc[-1]) -
            pd.to_datetime(tdf['entry_time'].iloc[0])).days / 365.25


CONFIGS = [
    dict(name="A baseline (no bias, 1st-half, size 0.3)",
         bias_filter="off",  min_fvg_size_atr=0.3, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=False, require_mss=False),
    dict(name="B strict bias",
         bias_filter="strict", min_fvg_size_atr=0.3, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=False, require_mss=False),
    dict(name="C bias + sweep gate",
         bias_filter="strict", min_fvg_size_atr=0.3, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=True, require_mss=False),
    dict(name="D bias + sweep + MSS (purist)",
         bias_filter="strict", min_fvg_size_atr=0.3, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=True, require_mss=True,
         fvg_max_age_bars=18, trade_expiry_bars=36),
    dict(name="E bias + larger FVG (0.6 ATR)",
         bias_filter="strict", min_fvg_size_atr=0.6, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=False, require_mss=False),
    dict(name="F bias + full hour entries",
         bias_filter="strict", min_fvg_size_atr=0.3, only_first_half_kz=False,
         max_trades_per_kz=2, require_sweep=False, require_mss=False),
    dict(name="G bias + sweep + larger FVG",
         bias_filter="strict", min_fvg_size_atr=0.5, only_first_half_kz=True,
         max_trades_per_kz=1, require_sweep=True, require_mss=False),
]


def main():
    print("Loading data...")
    df5 = load_5m("5m_data.csv")
    df1h = resample(df5, "1h")
    df4h = resample(df5, "4h")

    rows = []
    for cfg in CONFIGS:
        name = cfg.pop("name")
        # use 4h for the strictest bias proxy
        htf = df1h
        trades, diag = run_silver_bullet(df5, htf, rr=2.0, **cfg)
        tdf = trades_to_df(trades)
        s = compute_stats(tdf)
        s.pop("by_killzone", None)
        yrs = years_span(tdf)
        rows.append({
            "config":   name,
            "trades":   s.get("trades", 0),
            "tr/yr":    round(s.get("trades", 0) / yrs, 1) if yrs else 0,
            "win_%":    round(s.get("win_rate", 0), 1),
            "exp_R":    round(s.get("expectancy_R", 0), 3),
            "PF":       round(s.get("profit_factor", 0), 2),
            "totR":     round(s.get("total_R", 0), 1),
            "MDD_%":    round(s.get("max_dd_%", 0), 1),
            "fin_$":    round(s.get("final_equity", 0), 0),
        })

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    out.to_csv("results/config_sweep.csv", index=False)


if __name__ == "__main__":
    main()
