"""
Performance analysis & dashboard for the 2022 Model on 1m_data_2.csv
(2023-06 → 2024-07), using the leak-free strategy/data_loader fixes.

Reads the trade log produced by run_big_test_only_am_2.py and writes a
multi-panel dashboard plus a folder of detail charts and a stats.json.

Pure trading analysis — no funded-account / challenge mechanics.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as mticker

# ── Inputs / outputs ──────────────────────────────────────────────────────────
TRADES_CSV   = Path("big_test_only_am/trades.csv")
OUT_DIR      = Path("analysis_2022_fixed")
OUT_DIR.mkdir(exist_ok=True)

START_EQUITY    = 10_000.0
RISK_PER_TRADE  = 100.0   # $ risked per trade (1% of $10k)
NY_TZ           = "America/New_York"

# ── Load ──────────────────────────────────────────────────────────────────────
tdf = pd.read_csv(TRADES_CSV)
tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True).dt.tz_convert(NY_TZ)
tdf["exit_time"]  = pd.to_datetime(tdf["exit_time"],  utc=True).dt.tz_convert(NY_TZ)
tdf = tdf.sort_values("entry_time").reset_index(drop=True)

if "mae_r" not in tdf.columns: tdf["mae_r"] = np.nan
if "mfe_r" not in tdf.columns: tdf["mfe_r"] = np.nan

n = len(tdf)
print(f"Loaded {n} trades  {tdf['entry_time'].min()} → {tdf['entry_time'].max()}")

# ── Derived columns ───────────────────────────────────────────────────────────
tdf["pnl"]         = tdf["r"] * RISK_PER_TRADE
tdf["equity"]      = START_EQUITY + tdf["pnl"].cumsum()
tdf["cum_R"]       = tdf["r"].cumsum()
tdf["peak_equity"] = tdf["equity"].cummax()
tdf["dd_$"]        = tdf["equity"] - tdf["peak_equity"]
tdf["dd_%"]        = tdf["dd_$"] / tdf["peak_equity"] * 100
tdf["win"]         = tdf["r"] > 0
tdf["dur_min"]     = (tdf["exit_time"] - tdf["entry_time"]).dt.total_seconds() / 60
tdf["entry_hour"]  = tdf["entry_time"].dt.hour
tdf["entry_minute"]= tdf["entry_time"].dt.minute
tdf["dow"]         = tdf["entry_time"].dt.day_name()
tdf["date"]        = tdf["entry_time"].dt.date
tdf["month"]       = tdf["entry_time"].dt.tz_localize(None).dt.to_period("M")
tdf["year"]        = tdf["entry_time"].dt.year
tdf["iso_week"]    = tdf["entry_time"].dt.tz_localize(None).dt.to_period("W")

# ── Headline stats ────────────────────────────────────────────────────────────
r       = tdf["r"].values
pnl     = tdf["pnl"].values
wins    = int((r > 0).sum())
losses  = int((r <= 0).sum())
win_rate = wins / n * 100 if n else 0
expectancy_R = float(r.mean()) if n else 0.0
total_R      = float(r.sum())
gross_win    = float(pnl[pnl > 0].sum())
gross_loss   = float(-pnl[pnl < 0].sum())
profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
avg_win  = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
payoff   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

# Risk-adjusted (per-trade, then annualised by trades/year)
trade_days = (tdf["entry_time"].max() - tdf["entry_time"].min()).days
years = trade_days / 365.25 if trade_days > 0 else 1.0
trades_per_year = n / years
sharpe = (r.mean() / r.std() * np.sqrt(trades_per_year)) if r.std() > 0 else 0.0
downside = r[r < 0]
sortino = (r.mean() / downside.std() * np.sqrt(trades_per_year)) if len(downside) and downside.std() > 0 else 0.0

# Drawdown
equity = tdf["equity"].values
peak = np.maximum.accumulate(equity)
dd = equity - peak
max_dd_dollars = float(dd.min())
max_dd_pct = float((dd / peak).min() * 100) if peak.max() > 0 else 0.0
max_dd_idx = int(dd.argmin())
peak_idx_before_dd = int(np.argmax(peak[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
recovery_idx = next((i for i in range(max_dd_idx, n)
                     if equity[i] >= peak[max_dd_idx]), None)
trades_underwater = (recovery_idx - peak_idx_before_dd) if recovery_idx else (n - peak_idx_before_dd)

# Streak analysis
streaks_w, streaks_l = [], []
cur, cur_kind = 0, None
for w in tdf["win"].values:
    kind = "w" if w else "l"
    if kind == cur_kind:
        cur += 1
    else:
        if cur_kind == "w": streaks_w.append(cur)
        if cur_kind == "l": streaks_l.append(cur)
        cur_kind, cur = kind, 1
if cur_kind == "w": streaks_w.append(cur)
if cur_kind == "l": streaks_l.append(cur)
max_win_streak  = max(streaks_w) if streaks_w else 0
max_loss_streak = max(streaks_l) if streaks_l else 0

# CAGR (annualised return on starting equity)
final_eq = float(equity[-1])
cagr = ((final_eq / START_EQUITY) ** (1 / years) - 1) * 100 if years > 0 else 0

# MAE / MFE summary (only meaningful if columns are populated)
have_excursion = tdf["mae_r"].notna().any()
if have_excursion:
    mae_mean = float(tdf["mae_r"].mean())
    mae_min  = float(tdf["mae_r"].min())
    mfe_mean = float(tdf["mfe_r"].mean())
    mfe_max  = float(tdf["mfe_r"].max())
    avg_mae_winners = float(tdf.loc[tdf["win"], "mae_r"].mean()) if tdf["win"].any() else 0
    avg_mfe_losers  = float(tdf.loc[~tdf["win"], "mfe_r"].mean()) if (~tdf["win"]).any() else 0

# Per-period breakdowns
def _agg(grp):
    return grp["r"].agg(
        trades="count",
        wins=lambda x: int((x > 0).sum()),
        losses=lambda x: int((x <= 0).sum()),
        win_rate_pct=lambda x: round(float((x > 0).mean() * 100), 1),
        mean_R="mean",
        total_R="sum",
        net_pnl=lambda x: round(float(x.sum() * RISK_PER_TRADE), 2),
    ).round(3)

monthly = _agg(tdf.groupby("month"))
weekly  = _agg(tdf.groupby("iso_week"))
yearly  = _agg(tdf.groupby("year"))
by_dow  = _agg(tdf.groupby("dow"))
by_hour = _agg(tdf.groupby("entry_hour"))
by_dir  = _agg(tdf.groupby("direction"))
by_type = _agg(tdf.groupby("entry_type"))

# Save tables
monthly.to_csv(OUT_DIR / "monthly.csv")
weekly.to_csv(OUT_DIR / "weekly.csv")
yearly.to_csv(OUT_DIR / "yearly.csv")
by_dow.to_csv(OUT_DIR / "by_day_of_week.csv")
by_hour.to_csv(OUT_DIR / "by_entry_hour.csv")
by_dir.to_csv(OUT_DIR / "by_direction.csv")
by_type.to_csv(OUT_DIR / "by_entry_type.csv")

# Stats JSON
stats = {
    "period_start":            str(tdf["entry_time"].min()),
    "period_end":              str(tdf["entry_time"].max()),
    "calendar_days":           int(trade_days),
    "years":                   round(years, 2),
    "trades":                  int(n),
    "trades_per_year":         round(trades_per_year, 1),
    "wins":                    wins,
    "losses":                  losses,
    "win_rate_%":              round(win_rate, 2),
    "expectancy_R":            round(expectancy_R, 4),
    "total_R":                 round(total_R, 2),
    "gross_win_$":             round(gross_win, 2),
    "gross_loss_$":            round(gross_loss, 2),
    "net_pnl_$":               round(gross_win - gross_loss, 2),
    "profit_factor":           round(profit_factor, 3),
    "avg_win_$":               round(avg_win, 2),
    "avg_loss_$":              round(avg_loss, 2),
    "payoff_ratio":            round(payoff, 2),
    "sharpe_R":                round(sharpe, 3),
    "sortino_R":               round(sortino, 3),
    "starting_equity":         START_EQUITY,
    "final_equity":            round(final_eq, 2),
    "cagr_%":                  round(cagr, 2),
    "max_dd_$":                round(max_dd_dollars, 2),
    "max_dd_%":                round(max_dd_pct, 2),
    "trades_underwater":       int(trades_underwater),
    "max_win_streak":          max_win_streak,
    "max_loss_streak":         max_loss_streak,
    "avg_trade_duration_min":  round(float(tdf["dur_min"].mean()), 1),
    "median_trade_dur_min":    round(float(tdf["dur_min"].median()), 1),
}
if have_excursion:
    stats.update({
        "avg_mae_R":             round(mae_mean, 3),
        "worst_mae_R":           round(mae_min, 3),
        "avg_mfe_R":             round(mfe_mean, 3),
        "best_mfe_R":            round(mfe_max, 3),
        "avg_mae_R_winners":     round(avg_mae_winners, 3),
        "avg_mfe_R_losers":      round(avg_mfe_losers, 3),
    })

with open(OUT_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── Console summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  ICT 2022 Model — Performance Analysis  (LEAK-FIXED)")
print(f"  {tdf['entry_time'].min().date()}  →  {tdf['entry_time'].max().date()}"
      f"   ({years:.2f} years,  {trades_per_year:.0f} trades/yr)")
print("=" * 70)
print(f"  Trades        : {n:>7}     Win-rate     : {win_rate:>7.2f}%")
print(f"  Expectancy    : {expectancy_R:>7.3f} R   Total R      : {total_R:>7.2f}")
print(f"  Profit factor : {profit_factor:>7.3f}     Payoff       : {payoff:>7.2f}")
print(f"  Sharpe (R)    : {sharpe:>7.2f}     Sortino (R)  : {sortino:>7.2f}")
print(f"  Net PnL       : ${gross_win-gross_loss:>9,.2f}  (start=${START_EQUITY:,.0f}, "
      f"end=${final_eq:,.2f})")
print(f"  Max DD        : ${max_dd_dollars:>9,.2f}  ({max_dd_pct:.2f}% of peak)")
print(f"  CAGR          : {cagr:>7.2f}%")
print(f"  Win streak    : {max_win_streak:>7}     Loss streak  : {max_loss_streak:>7}")
if have_excursion:
    print(f"  Avg MAE/MFE   : {mae_mean:>+7.3f}R / {mfe_mean:>+5.3f}R")
print("=" * 70)
print()
print("Yearly breakdown:")
print(yearly.to_string())
print()
print("Monthly breakdown (last 12):")
print(monthly.tail(12).to_string())

# ── Charts ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# 1. Master dashboard
fig = plt.figure(figsize=(20, 24))
g = gs.GridSpec(7, 4, figure=fig, hspace=0.55, wspace=0.30)

# Row 1: Equity + drawdown (full width)
ax_eq = fig.add_subplot(g[0, :])
ax_eq.plot(tdf["exit_time"], tdf["equity"], color="#1f77b4", lw=1.6, label="Equity")
ax_eq.fill_between(tdf["exit_time"], tdf["equity"], tdf["peak_equity"],
                   where=(tdf["equity"] < tdf["peak_equity"]),
                   color="red", alpha=0.18, label="Drawdown")
ax_eq.axhline(START_EQUITY, color="gray", ls="--", lw=0.8)
ax_eq.set_title(f"Equity Curve — start ${START_EQUITY:,.0f} → end ${final_eq:,.2f} "
                f"(net ${final_eq-START_EQUITY:+,.2f}, {((final_eq/START_EQUITY-1)*100):+.2f}%)")
ax_eq.set_ylabel("Equity ($)")
ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_eq.legend(loc="upper left"); ax_eq.grid(alpha=0.3)

# Row 2: Drawdown % (full width)
ax_dd = fig.add_subplot(g[1, :])
ax_dd.fill_between(tdf["exit_time"], tdf["dd_%"], 0, color="red", alpha=0.4)
ax_dd.plot(tdf["exit_time"], tdf["dd_%"], color="darkred", lw=0.8)
ax_dd.set_title(f"Drawdown (%)  — max {max_dd_pct:.2f}%, ${max_dd_dollars:,.2f}, "
                f"underwater {trades_underwater} trades")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.grid(alpha=0.3)

# Row 3: Cumulative R + R-distribution + monthly bars + win-rate-rolling
ax_cumR = fig.add_subplot(g[2, 0])
ax_cumR.plot(np.arange(1, n + 1), tdf["cum_R"], color="#2ca02c", lw=1.4)
ax_cumR.axhline(0, color="black", lw=0.8)
ax_cumR.set_title(f"Cumulative R  (total {total_R:+.2f}R)")
ax_cumR.set_xlabel("Trade #"); ax_cumR.set_ylabel("R"); ax_cumR.grid(alpha=0.3)

ax_rdist = fig.add_subplot(g[2, 1])
ax_rdist.hist(tdf["r"], bins=30, color="#1f77b4", edgecolor="black", alpha=0.75)
ax_rdist.axvline(0, color="black", lw=0.8)
ax_rdist.axvline(expectancy_R, color="red", lw=1.2, ls="--",
                 label=f"E[R]={expectancy_R:+.3f}")
ax_rdist.set_title("R-multiple distribution")
ax_rdist.set_xlabel("R"); ax_rdist.set_ylabel("Count")
ax_rdist.legend(); ax_rdist.grid(alpha=0.3)

ax_mo = fig.add_subplot(g[2, 2:])
mo_pnl = monthly["net_pnl"]
mo_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mo_pnl.values]
ax_mo.bar(range(len(mo_pnl)), mo_pnl.values, color=mo_colors, edgecolor="black")
ax_mo.axhline(0, color="black", lw=0.8)
ax_mo.set_title(f"Monthly P&L ($) — {(mo_pnl > 0).sum()} green / {(mo_pnl <= 0).sum()} red")
ax_mo.set_ylabel("P&L ($)")
ax_mo.set_xticks(range(len(mo_pnl)))
ax_mo.set_xticklabels([str(m) for m in mo_pnl.index], rotation=45, ha="right", fontsize=8)
ax_mo.grid(alpha=0.3, axis="y")

# Row 4: Rolling win-rate, rolling expectancy, payoff/PF, win/loss avg
roll_n = max(20, n // 15)
ax_rwr = fig.add_subplot(g[3, 0])
roll_wr = tdf["win"].rolling(roll_n).mean() * 100
ax_rwr.plot(np.arange(1, n + 1), roll_wr, color="purple", lw=1.4)
ax_rwr.axhline(50, color="gray", ls="--", lw=0.8)
ax_rwr.set_title(f"Rolling win-rate ({roll_n}-trade window)")
ax_rwr.set_xlabel("Trade #"); ax_rwr.set_ylabel("%"); ax_rwr.grid(alpha=0.3)

ax_rexp = fig.add_subplot(g[3, 1])
roll_exp = tdf["r"].rolling(roll_n).mean()
ax_rexp.plot(np.arange(1, n + 1), roll_exp, color="teal", lw=1.4)
ax_rexp.axhline(0, color="black", ls="--", lw=0.8)
ax_rexp.fill_between(np.arange(1, n + 1), roll_exp, 0, where=roll_exp >= 0,
                     alpha=0.15, color="green")
ax_rexp.fill_between(np.arange(1, n + 1), roll_exp, 0, where=roll_exp < 0,
                     alpha=0.15, color="red")
ax_rexp.set_title(f"Rolling expectancy ({roll_n}-trade window)")
ax_rexp.set_xlabel("Trade #"); ax_rexp.set_ylabel("E[R]"); ax_rexp.grid(alpha=0.3)

# Win vs loss bar
ax_wl = fig.add_subplot(g[3, 2])
ax_wl.bar(["Avg win", "|Avg loss|"], [avg_win, abs(avg_loss)],
          color=["#2ca02c", "#d62728"], edgecolor="black")
ax_wl.set_title(f"Avg Win/Loss  (payoff={payoff:.2f})")
ax_wl.set_ylabel("$")
for i, v in enumerate([avg_win, abs(avg_loss)]):
    ax_wl.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
ax_wl.grid(alpha=0.3, axis="y")

# Profit factor / win-rate metric tile
ax_kpi = fig.add_subplot(g[3, 3])
ax_kpi.axis("off")
kpi_text = (
    f"KPI snapshot\n\n"
    f"Profit factor   {profit_factor:>7.3f}\n"
    f"Win-rate        {win_rate:>6.2f}%\n"
    f"Expectancy      {expectancy_R:>+6.3f} R\n"
    f"Total R         {total_R:>+6.2f} R\n"
    f"Sharpe (R)      {sharpe:>6.2f}\n"
    f"Sortino (R)     {sortino:>6.2f}\n"
    f"Max DD          {max_dd_pct:>5.2f}%\n"
    f"CAGR            {cagr:>5.2f}%\n"
    f"Win streak      {max_win_streak:>6d}\n"
    f"Loss streak     {max_loss_streak:>6d}"
)
ax_kpi.text(0.0, 1.0, kpi_text, family="monospace", fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#999"))

# Row 5: by direction, by entry type, by day-of-week, by hour
def _bar_panel(ax, df, value_col, title, color="#1f77b4"):
    vals = df[value_col].values
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals] if value_col in ("net_pnl", "total_R", "mean_R") else [color] * len(vals)
    ax.bar(df.index.astype(str), vals, color=colors, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(alpha=0.3, axis="y")

ax_dir = fig.add_subplot(g[4, 0])
_bar_panel(ax_dir, by_dir, "net_pnl", "By direction — Net P&L ($)")
for x, v, c in zip(by_dir.index.astype(str), by_dir["net_pnl"].values, by_dir["trades"].values):
    ax_dir.text(x, v, f"{c}t\n${v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

ax_typ = fig.add_subplot(g[4, 1])
_bar_panel(ax_typ, by_type, "net_pnl", "By entry type — Net P&L ($)")
for x, v, c in zip(by_type.index.astype(str), by_type["net_pnl"].values, by_type["trades"].values):
    ax_typ.text(x, v, f"{c}t\n${v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

ax_dow = fig.add_subplot(g[4, 2])
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
by_dow = by_dow.reindex([d for d in dow_order if d in by_dow.index])
_bar_panel(ax_dow, by_dow, "net_pnl", "By day of week — Net P&L ($)")
ax_dow.tick_params(axis="x", rotation=30)

ax_hr = fig.add_subplot(g[4, 3])
_bar_panel(ax_hr, by_hour, "net_pnl", "By entry hour (NY) — Net P&L ($)")

# Row 6: Trade duration histogram, MAE histogram, MFE histogram, MAE-vs-R scatter
ax_dur = fig.add_subplot(g[5, 0])
ax_dur.hist(tdf["dur_min"], bins=30, color="#9467bd", edgecolor="black", alpha=0.8)
ax_dur.set_title(f"Trade duration (min) — median {tdf['dur_min'].median():.0f}m")
ax_dur.set_xlabel("Minutes"); ax_dur.grid(alpha=0.3)

if have_excursion:
    ax_mae = fig.add_subplot(g[5, 1])
    ax_mae.hist(tdf["mae_r"].dropna(), bins=30, color="#d62728", edgecolor="black", alpha=0.8)
    ax_mae.axvline(-1, color="black", ls="--", lw=0.8, label="-1R (stop)")
    ax_mae.axvline(tdf["mae_r"].mean(), color="navy", lw=1.2, label=f"avg {tdf['mae_r'].mean():.2f}R")
    ax_mae.set_title("MAE distribution (intraday worst R)")
    ax_mae.set_xlabel("R"); ax_mae.legend(fontsize=8); ax_mae.grid(alpha=0.3)

    ax_mfe = fig.add_subplot(g[5, 2])
    ax_mfe.hist(tdf["mfe_r"].dropna(), bins=30, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax_mfe.axvline(2, color="black", ls="--", lw=0.8, label="+2R (target)")
    ax_mfe.axvline(tdf["mfe_r"].mean(), color="navy", lw=1.2, label=f"avg {tdf['mfe_r'].mean():.2f}R")
    ax_mfe.set_title("MFE distribution (intraday best R)")
    ax_mfe.set_xlabel("R"); ax_mfe.legend(fontsize=8); ax_mfe.grid(alpha=0.3)

    ax_sc = fig.add_subplot(g[5, 3])
    colors_sc = ["#2ca02c" if w else "#d62728" for w in tdf["win"]]
    ax_sc.scatter(tdf["mae_r"], tdf["mfe_r"], c=colors_sc, alpha=0.6, s=15, edgecolors="black", linewidths=0.3)
    ax_sc.axhline(0, color="black", lw=0.6); ax_sc.axvline(0, color="black", lw=0.6)
    ax_sc.set_title("MAE vs MFE (green=win, red=loss)")
    ax_sc.set_xlabel("MAE (R)"); ax_sc.set_ylabel("MFE (R)"); ax_sc.grid(alpha=0.3)

# Row 7: Yearly breakdown table + monthly heatmap
ax_yr = fig.add_subplot(g[6, 0:2])
yr_pnl = yearly["net_pnl"]
yr_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in yr_pnl.values]
ax_yr.bar(yr_pnl.index.astype(str), yr_pnl.values, color=yr_colors, edgecolor="black")
ax_yr.axhline(0, color="black", lw=0.8)
ax_yr.set_title("Yearly P&L ($)  with trades / win-rate annotation")
for x, y, n_t, wr in zip(yr_pnl.index.astype(str), yr_pnl.values,
                          yearly["trades"].values, yearly["win_rate_pct"].values):
    ax_yr.text(x, y, f"{n_t}t  {wr:.0f}%wr\n${y:,.0f}",
               ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
ax_yr.grid(alpha=0.3, axis="y")

# Heatmap of month × year P&L
ax_hm = fig.add_subplot(g[6, 2:])
piv = tdf.copy()
piv["mo"] = piv["entry_time"].dt.month
piv["yr"] = piv["entry_time"].dt.year
heat = piv.pivot_table(index="yr", columns="mo", values="pnl", aggfunc="sum").fillna(0)
all_months = list(range(1, 13))
heat = heat.reindex(columns=all_months, fill_value=0)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
vmax = max(abs(heat.values.min()), abs(heat.values.max()), 1)
im = ax_hm.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
ax_hm.set_xticks(range(12)); ax_hm.set_xticklabels(month_labels, fontsize=9)
ax_hm.set_yticks(range(len(heat.index))); ax_hm.set_yticklabels(heat.index, fontsize=9)
ax_hm.set_title("Monthly P&L heatmap ($)")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if v != 0:
            ax_hm.text(j, i, f"{v:.0f}", ha="center", va="center",
                       fontsize=8, color="black" if abs(v) < 0.6 * vmax else "white")
plt.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)

# Title
fig.suptitle(
    f"ICT 2022 Model — Performance Dashboard (LEAK-FREE) | "
    f"NAS, AM session, RR=2  |  {tdf['entry_time'].min().date()} → {tdf['entry_time'].max().date()}",
    fontsize=14, fontweight="bold", y=0.995,
)
fig.savefig(OUT_DIR / "dashboard.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nDashboard → {OUT_DIR}/dashboard.png")

# ── Standalone detail charts ──────────────────────────────────────────────────

# Equity + DD detail (large)
fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
axes[0].plot(tdf["exit_time"], tdf["equity"], color="#1f77b4", lw=1.6, label="Equity")
axes[0].fill_between(tdf["exit_time"], tdf["equity"], tdf["peak_equity"],
                     where=tdf["equity"] < tdf["peak_equity"],
                     color="red", alpha=0.18)
axes[0].axhline(START_EQUITY, color="gray", ls="--", lw=0.8)
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("Equity ($)")
axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].fill_between(tdf["exit_time"], tdf["dd_%"], 0, color="red", alpha=0.4)
axes[1].plot(tdf["exit_time"], tdf["dd_%"], color="darkred", lw=0.8)
axes[1].set_ylabel("Drawdown (%)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "equity_and_drawdown.png", dpi=130)
plt.close(fig)

# Monthly bars
fig, ax = plt.subplots(figsize=(15, 4))
mo_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly["net_pnl"].values]
ax.bar(range(len(monthly)), monthly["net_pnl"].values, color=mo_colors, edgecolor="black")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Monthly P&L ($)")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right", fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_DIR / "monthly_pnl.png", dpi=130)
plt.close(fig)

# Weekly bars
fig, ax = plt.subplots(figsize=(max(15, len(weekly) // 2), 4))
wk_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in weekly["net_pnl"].values]
ax.bar(range(len(weekly)), weekly["net_pnl"].values, color=wk_colors, edgecolor="black")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Weekly P&L ($)")
tick_step = max(1, len(weekly) // 25)
ax.set_xticks(range(0, len(weekly), tick_step))
ax.set_xticklabels([str(weekly.index[i].start_time.date())
                    for i in range(0, len(weekly), tick_step)],
                   rotation=45, ha="right", fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_DIR / "weekly_pnl.png", dpi=130)
plt.close(fig)

# R distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(tdf["r"], bins=40, color="#1f77b4", edgecolor="black", alpha=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(expectancy_R, color="red", lw=1.4, ls="--", label=f"E[R]={expectancy_R:+.3f}")
ax.set_title("R-multiple distribution")
ax.set_xlabel("R"); ax.set_ylabel("Count"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "r_distribution.png", dpi=130)
plt.close(fig)

# Hour-of-day heatmap (entry hour × outcome)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
by_hour_disp = by_hour.reset_index()
axes[0].bar(by_hour_disp["entry_hour"].astype(str), by_hour_disp["net_pnl"],
            color=["#2ca02c" if v >= 0 else "#d62728" for v in by_hour_disp["net_pnl"]],
            edgecolor="black")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Net P&L by entry hour (NY)")
axes[0].set_xlabel("Hour"); axes[0].set_ylabel("$"); axes[0].grid(alpha=0.3, axis="y")
axes[1].bar(by_hour_disp["entry_hour"].astype(str), by_hour_disp["win_rate_pct"],
            color="#1f77b4", edgecolor="black", alpha=0.8)
axes[1].axhline(50, color="gray", ls="--", lw=0.8)
axes[1].set_title("Win-rate by entry hour (NY)")
axes[1].set_xlabel("Hour"); axes[1].set_ylabel("%"); axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_DIR / "by_hour.png", dpi=130)
plt.close(fig)

# Cumulative R
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(1, n + 1), tdf["cum_R"], color="#2ca02c", lw=1.6)
ax.axhline(0, color="black", lw=0.8)
ax.fill_between(np.arange(1, n + 1), tdf["cum_R"], 0,
                where=tdf["cum_R"] >= 0, color="green", alpha=0.15)
ax.fill_between(np.arange(1, n + 1), tdf["cum_R"], 0,
                where=tdf["cum_R"] < 0,  color="red", alpha=0.15)
ax.set_title(f"Cumulative R — total {total_R:+.2f}R over {n} trades")
ax.set_xlabel("Trade #"); ax.set_ylabel("R"); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "cumulative_r.png", dpi=130)
plt.close(fig)

# MAE / MFE charts (only if data present)
if have_excursion:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(tdf.loc[tdf["win"], "mae_r"].dropna(), bins=25, alpha=0.6,
                 color="#2ca02c", label="Winners' MAE", edgecolor="black")
    axes[0].hist(tdf.loc[~tdf["win"], "mae_r"].dropna(), bins=25, alpha=0.6,
                 color="#d62728", label="Losers' MAE", edgecolor="black")
    axes[0].axvline(-1, color="black", ls="--", lw=0.8)
    axes[0].set_title("MAE distribution by outcome")
    axes[0].set_xlabel("MAE (R)"); axes[0].set_ylabel("Count")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(tdf.loc[tdf["win"], "mfe_r"].dropna(), bins=25, alpha=0.6,
                 color="#2ca02c", label="Winners' MFE", edgecolor="black")
    axes[1].hist(tdf.loc[~tdf["win"], "mfe_r"].dropna(), bins=25, alpha=0.6,
                 color="#d62728", label="Losers' MFE", edgecolor="black")
    axes[1].axvline(2, color="black", ls="--", lw=0.8)
    axes[1].set_title("MFE distribution by outcome")
    axes[1].set_xlabel("MFE (R)"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mae_mfe_by_outcome.png", dpi=130)
    plt.close(fig)

print(f"\nAll detail charts → {OUT_DIR}/")
print(f"Stats → {OUT_DIR}/stats.json")
