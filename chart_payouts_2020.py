"""
Payout & Cash-Flow Charts — 2020-onward Funded Sim
====================================================
Produces four PNGs inside funded_sim_2020/:

  1. monthly_payouts.png   — bar chart: trader payout $ per calendar month
  2. annual_payouts.png    — bar chart: trader payout $ per year
  3. cumulative_cash.png   — line chart: cumulative net cash over time
  4. dashboard.png         — all three panels on one figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT = Path("funded_sim_2020")
OUT.mkdir(exist_ok=True)

# ── 1. Load the 2020-onward trade list (for date lookup) ──────────────────────
START_DATE = pd.Timestamp("2020-01-01", tz="UTC")
trades_raw = pd.read_csv("big_test_only_am/trades.csv")
trades_raw["entry_time"] = pd.to_datetime(trades_raw["entry_time"], utc=True)
trades = (trades_raw[trades_raw["entry_time"] >= START_DATE]
          .reset_index(drop=True))          # index = trade_idx used in sim
trades["date"] = trades["entry_time"].dt.date

# ── 2. Load payout log ────────────────────────────────────────────────────────
payouts = pd.read_csv(OUT / "payout_log.csv")
# trade_idx in the log is the index into the 2020+ filtered trade list
payouts["date"] = pd.to_datetime(
    payouts["trade_idx"].map(lambda i: trades.loc[i, "date"])
)
payouts["year"]       = payouts["date"].dt.year
payouts["month"]      = payouts["date"].dt.month
payouts["year_month"] = payouts["date"].dt.to_period("M")

# ── 3. Account events for cash-flow timeline ─────────────────────────────────
# We'll reconstruct a cumulative cash series keyed to actual dates.
# Each payout adds trader_gets.  Each fee paid/reimbursed uses the account's
# start trade index to find the date.
acct_log = pd.read_csv(OUT / "account_log.csv")
acct_log["start_date"] = pd.to_datetime(
    acct_log["start_trade"].map(lambda i: trades.loc[i, "date"])
)

# Build an event series: (date, cash_delta)
events = []

for _, row in acct_log.iterrows():
    d = row["start_date"]
    events.append((d, -row["fee_paid"]))
    if row["fee_reimbursed"] > 0:
        events.append((d, row["fee_reimbursed"]))
    if row["challenge_bonus"] > 0:
        events.append((d, row["challenge_bonus"]))

for _, row in payouts.iterrows():
    events.append((row["date"], row["trader_gets"]))

ev = (pd.DataFrame(events, columns=["date", "delta"])
        .sort_values("date")
        .reset_index(drop=True))
ev["cumulative"] = ev["delta"].cumsum()

# ── Palette ───────────────────────────────────────────────────────────────────
GREEN  = "#2ecc71"
BLUE   = "#3498db"
ORANGE = "#e67e22"
RED    = "#e74c3c"
DARK   = "#2c3e50"
GRID   = "#ecf0f1"

def _money(x, _):
    return f"${x:,.0f}"

# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Monthly Payouts
# ══════════════════════════════════════════════════════════════════════════════
monthly = (payouts.groupby("year_month")["trader_gets"]
                   .sum()
                   .reset_index())
monthly["label"] = monthly["year_month"].dt.strftime("%b\n%Y")

fig1, ax1 = plt.subplots(figsize=(16, 6))
fig1.patch.set_facecolor(DARK)
ax1.set_facecolor(DARK)

bars = ax1.bar(range(len(monthly)), monthly["trader_gets"],
               color=GREEN, edgecolor="white", linewidth=0.4, width=0.7)

# colour months with zero (gap months) differently — none here, but future-proof
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly["label"], fontsize=7.5, color="white", rotation=0)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#4a4a4a")
ax1.set_facecolor(DARK)

for bar, val in zip(bars, monthly["trader_gets"]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 500,
             f"${val:,.0f}", ha="center", va="bottom",
             fontsize=6.5, color="white", fontweight="bold")

ax1.set_title("Monthly Trader Payouts — 2020-Onward Funded Sim",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax1.set_ylabel("Trader Payout ($)", color="white", fontsize=11)
ax1.yaxis.label.set_color("white")
ax1.grid(axis="y", color="#444", linewidth=0.5, linestyle="--")

total_label = f"Total payouts: ${monthly['trader_gets'].sum():,.0f}"
ax1.text(0.99, 0.97, total_label, transform=ax1.transAxes,
         ha="right", va="top", color=GREEN, fontsize=11, fontweight="bold")

plt.tight_layout()
fig1.savefig(OUT / "monthly_payouts.png", dpi=140, bbox_inches="tight",
             facecolor=DARK)
plt.close(fig1)
print("✓ monthly_payouts.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Annual Payouts
# ══════════════════════════════════════════════════════════════════════════════
annual = payouts.groupby("year")["trader_gets"].sum().reset_index()
annual["count"] = payouts.groupby("year")["trader_gets"].count().values

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor(DARK)
ax2.set_facecolor(DARK)

bars2 = ax2.bar(annual["year"].astype(str), annual["trader_gets"],
                color=BLUE, edgecolor="white", linewidth=0.5, width=0.55)

for bar, row in zip(bars2, annual.itertuples()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1500,
             f"${row.trader_gets:,.0f}\n({row.count} payouts)",
             ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
ax2.tick_params(colors="white", labelsize=11)
ax2.spines[:].set_color("#4a4a4a")
ax2.set_title("Annual Trader Payouts — 2020-Onward Funded Sim",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax2.set_ylabel("Annual Payout ($)", color="white", fontsize=11)
ax2.grid(axis="y", color="#444", linewidth=0.5, linestyle="--")

plt.tight_layout()
fig2.savefig(OUT / "annual_payouts.png", dpi=140, bbox_inches="tight",
             facecolor=DARK)
plt.close(fig2)
print("✓ annual_payouts.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Cumulative Net Cash Curve
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(14, 6))
fig3.patch.set_facecolor(DARK)
ax3.set_facecolor(DARK)

# Shade below/above zero
ax3.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax3.fill_between(ev["date"], ev["cumulative"], 0,
                 where=(ev["cumulative"] >= 0),
                 alpha=0.15, color=GREEN, interpolate=True)
ax3.fill_between(ev["date"], ev["cumulative"], 0,
                 where=(ev["cumulative"] < 0),
                 alpha=0.25, color=RED, interpolate=True)

ax3.plot(ev["date"], ev["cumulative"],
         color=GREEN, linewidth=2.2, zorder=3)

# Mark each payout event
payout_ev = ev[ev["delta"] > 0]
ax3.scatter(payout_ev["date"], payout_ev["cumulative"],
            color=ORANGE, s=35, zorder=4, label="Payout received", linewidths=0)

# Mark fee events (negative deltas)
fee_ev = ev[ev["delta"] < 0]
ax3.scatter(fee_ev["date"], fee_ev["cumulative"],
            color=RED, s=55, marker="x", zorder=5, label="Fee paid", linewidths=1.5)

# Annotate final value
final_val = ev["cumulative"].iloc[-1]
ax3.annotate(f"Final: ${final_val:,.0f}",
             xy=(ev["date"].iloc[-1], final_val),
             xytext=(-100, 20), textcoords="offset points",
             color=GREEN, fontsize=11, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
ax3.tick_params(colors="white", labelsize=9)
ax3.spines[:].set_color("#4a4a4a")
ax3.set_title("Cumulative Net Cash to Trader — 2020-Onward Funded Sim",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax3.set_ylabel("Cumulative Net Cash ($)", color="white", fontsize=11)
ax3.set_xlabel("Date", color="white", fontsize=11)
ax3.legend(facecolor="#333", edgecolor="#555", labelcolor="white", fontsize=10)
ax3.grid(color="#444", linewidth=0.4, linestyle="--")

plt.tight_layout()
fig3.savefig(OUT / "cumulative_cash.png", dpi=140, bbox_inches="tight",
             facecolor=DARK)
plt.close(fig3)
print("✓ cumulative_cash.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Dashboard (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(20, 14))
fig4.patch.set_facecolor(DARK)
gs = fig4.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

axA = fig4.add_subplot(gs[0, :])   # full width — monthly
axB = fig4.add_subplot(gs[1, 0])   # annual
axC = fig4.add_subplot(gs[1, 1])   # cumulative cash

for ax in [axA, axB, axC]:
    ax.set_facecolor(DARK)

# ── Panel A: Monthly ─────────────────────────────────────────────────────────
barsA = axA.bar(range(len(monthly)), monthly["trader_gets"],
                color=GREEN, edgecolor="white", linewidth=0.4, width=0.7)
axA.set_xticks(range(len(monthly)))
axA.set_xticklabels(monthly["label"], fontsize=7, color="white", rotation=0)
axA.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
axA.tick_params(colors="white")
axA.spines[:].set_color("#4a4a4a")
for bar, val in zip(barsA, monthly["trader_gets"]):
    axA.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 500,
             f"${val:,.0f}", ha="center", va="bottom",
             fontsize=6, color="white", fontweight="bold")
axA.set_title("Monthly Trader Payouts", color="white", fontsize=13, fontweight="bold")
axA.set_ylabel("Payout ($)", color="white", fontsize=10)
axA.grid(axis="y", color="#444", linewidth=0.5, linestyle="--")
axA.text(0.99, 0.97, f"Total: ${monthly['trader_gets'].sum():,.0f}",
         transform=axA.transAxes, ha="right", va="top",
         color=GREEN, fontsize=11, fontweight="bold")

# ── Panel B: Annual ──────────────────────────────────────────────────────────
barsB = axB.bar(annual["year"].astype(str), annual["trader_gets"],
                color=BLUE, edgecolor="white", linewidth=0.5, width=0.55)
for bar, row in zip(barsB, annual.itertuples()):
    axB.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1500,
             f"${row.trader_gets:,.0f}\n({row.count}×)",
             ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")
axB.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
axB.tick_params(colors="white", labelsize=10)
axB.spines[:].set_color("#4a4a4a")
axB.set_title("Annual Trader Payouts", color="white", fontsize=13, fontweight="bold")
axB.set_ylabel("Annual Payout ($)", color="white", fontsize=10)
axB.grid(axis="y", color="#444", linewidth=0.5, linestyle="--")

# ── Panel C: Cumulative cash ──────────────────────────────────────────────────
axC.axhline(0, color="#888", linewidth=0.8, linestyle="--")
axC.fill_between(ev["date"], ev["cumulative"], 0,
                 where=(ev["cumulative"] >= 0),
                 alpha=0.15, color=GREEN, interpolate=True)
axC.fill_between(ev["date"], ev["cumulative"], 0,
                 where=(ev["cumulative"] < 0),
                 alpha=0.25, color=RED, interpolate=True)
axC.plot(ev["date"], ev["cumulative"], color=GREEN, linewidth=2, zorder=3)
payout_ev2 = ev[ev["delta"] > 0]
fee_ev2    = ev[ev["delta"] < 0]
axC.scatter(payout_ev2["date"], payout_ev2["cumulative"],
            color=ORANGE, s=25, zorder=4, label="Payout")
axC.scatter(fee_ev2["date"], fee_ev2["cumulative"],
            color=RED, s=40, marker="x", zorder=5, label="Fee", linewidths=1.2)
axC.annotate(f"${final_val:,.0f}",
             xy=(ev["date"].iloc[-1], final_val),
             xytext=(-90, 18), textcoords="offset points",
             color=GREEN, fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
axC.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
axC.tick_params(colors="white", labelsize=9)
axC.spines[:].set_color("#4a4a4a")
axC.set_title("Cumulative Net Cash Curve", color="white", fontsize=13, fontweight="bold")
axC.set_ylabel("Net Cash ($)", color="white", fontsize=10)
axC.set_xlabel("Date", color="white", fontsize=9)
axC.legend(facecolor="#333", edgecolor="#555", labelcolor="white", fontsize=9)
axC.grid(color="#444", linewidth=0.4, linestyle="--")

# ── Super-title ───────────────────────────────────────────────────────────────
fig4.suptitle("Funded Account Sim — 2020-Onward  |  $100k Account  |  90% Payout Split",
              color="white", fontsize=15, fontweight="bold", y=0.98)

fig4.savefig(OUT / "dashboard.png", dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close(fig4)
print("✓ dashboard.png")

print(f"\nAll charts saved to {OUT}/")
print("\n── Monthly breakdown ──────────────────────────────")
for _, row in monthly.iterrows():
    print(f"  {row['year_month']}  ${row['trader_gets']:>10,.2f}")
print("\n── Annual breakdown ───────────────────────────────")
for _, row in annual.iterrows():
    print(f"  {int(row['year'])}   ${row['trader_gets']:>10,.2f}  ({int(row['count'])} payouts)")
print(f"\n  TOTAL NET CASH: ${final_val:,.2f}")
