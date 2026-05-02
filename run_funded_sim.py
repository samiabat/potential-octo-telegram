"""
Funded Account Simulator — ICT 2022 Model, AM Session (2016-2025)

Uses the real chronological AM trade sequence (big_test_only_am/trades.csv)
to simulate repeatedly buying a $100k prop-firm challenge account and trading
through challenge phases until all trades are exhausted.

Prop-firm rules assumed:
  Account size          : $100,000
  Challenge fee         : $550 (per attempt)
  Phase 1 profit target : 8%  = $8,000   |  max DD (trailing) : 8%  = $8,000
  Phase 2 profit target : 5%  = $5,000   |  max DD (trailing) : 5%  = $5,000
  Funded max DD         : 10% = $10,000  (static — floor = $90,000)
  Daily DD limit        : 5%  = $5,000   (all phases, from SOD equity)
  Fee reimbursed        : yes, when both challenge phases are passed
  Challenge bonus       : 15% of net challenge P&L (P1+P2), paid on funding
  Profit split          : 90% to trader
  Payout trigger        : $5,000 funded profit (equity ≥ $105,000)

Risk sizing: 1% of $100k = $1,000 per R  (scaled from original $10k backtests)
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ──────────────────────────────────────────────────────────────────
ACCOUNT_SIZE      = 100_000
RISK_PER_R        = 1_000       # 1% of $100k
CHALLENGE_FEE     = 550
PROFIT_SPLIT      = 0.90
CHALLENGE_BONUS_RATE = 0.15

# Challenge thresholds
P1_PROFIT_TARGET  = 8_000       # 8% of $100k
P1_TRAIL_DD       = 8_000       # 8% trailing from peak
P2_PROFIT_TARGET  = 5_000       # 5% of $100k
P2_TRAIL_DD       = 5_000       # 5% trailing from peak

# Funded thresholds
FUNDED_STATIC_DD  = 10_000      # 10% below $100k → floor = $90k
FUNDED_FLOOR      = ACCOUNT_SIZE - FUNDED_STATIC_DD   # $90,000
DAILY_DD_LIMIT    = 5_000       # 5% of $100k, reset each day

PAYOUT_TRIGGER    = 5_000       # request payout when funded profit ≥ $5,000

RESULTS_DIR = Path("funded_sim")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Load trades ────────────────────────────────────────────────────────────────
df = pd.read_csv("big_test_only_am/trades.csv")
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["date"]       = df["entry_time"].dt.date
df["pnl"]        = df["r"] * RISK_PER_R      # $ P&L per trade
trades           = df.reset_index(drop=True)
N                = len(trades)

print(f"Loaded {N} trades across {df['date'].nunique()} trading days "
      f"({df['date'].min()} → {df['date'].max()})")
print()

# ── Helper — simulate one "phase" from a given trade index ─────────────────────
def run_phase(trade_idx, start_equity, trail_dd_limit, profit_target,
              daily_dd_limit=DAILY_DD_LIMIT):
    """
    Simulate one challenge phase (or funded period with profit_target=None).

    Returns dict with keys:
      end_idx       : next trade index after this phase
      end_equity    : final equity
      blown         : bool — hit a DD limit
      target_hit    : bool — hit the profit target
      trades_log    : list of (date, pnl, equity) tuples
      daily_blown   : bool — blown by daily DD specifically
    """
    equity = start_equity
    peak   = start_equity          # for trailing DD
    blown  = False
    daily_blown = False
    target_hit  = False
    trades_log  = []

    # Group trades by day for daily-DD enforcement
    remaining = trades.iloc[trade_idx:].copy()
    if remaining.empty:
        return dict(end_idx=trade_idx, end_equity=equity, blown=False,
                    target_hit=False, trades_log=[], daily_blown=False)

    idx = trade_idx
    for date, day_grp in remaining.groupby("date"):
        sod_equity = equity          # start-of-day for daily DD check
        day_blown  = False

        for row_idx in day_grp.index:
            row = trades.loc[row_idx]
            pnl = row["pnl"]
            equity += pnl
            peak = max(peak, equity)
            idx += 1
            trades_log.append((date, pnl, equity))

            # ── Daily DD check ─────────────────────────────────────
            if sod_equity - equity > daily_dd_limit:
                blown = True; daily_blown = True; day_blown = True
                break

            # ── Phase DD check (trailing from peak) ────────────────
            if trail_dd_limit is not None:
                if peak - equity > trail_dd_limit:
                    blown = True; day_blown = True
                    break

            # ── Funded static floor check ───────────────────────────
            if trail_dd_limit is None:          # funded — static floor
                if equity < FUNDED_FLOOR:
                    blown = True; day_blown = True
                    break

        if day_blown or blown:
            break

        # ── End-of-day profit target check ─────────────────────────
        if profit_target is not None:
            if equity - ACCOUNT_SIZE >= profit_target:
                target_hit = True
                break

        # ── Funded payout check (end of day) ───────────────────────
        if profit_target is None:
            if equity - ACCOUNT_SIZE >= PAYOUT_TRIGGER:
                target_hit = True   # signals "payout ready"
                break

    return dict(end_idx=idx, end_equity=equity, blown=blown,
                target_hit=target_hit, trades_log=trades_log,
                daily_blown=daily_blown)


# ── Main simulation ────────────────────────────────────────────────────────────
account_log = []   # one entry per account attempt
payout_log  = []   # one entry per payout event
equity_curve = []  # (account_num, phase, date, pnl, running_equity) for global curve

trade_idx = 0
account_num = 0

# Running financial totals
total_fees_paid        = 0.0
total_fees_reimbursed  = 0.0
total_challenge_bonus  = 0.0
total_payouts_received = 0.0   # trader's 90% take
total_payouts_gross    = 0.0   # full profit withdrawn
global_equity          = 0.0   # running net cash position of trader

print("=" * 60)
print("  Running funded account simulation …")
print("=" * 60)

while trade_idx < N:
    account_num += 1
    entry = dict(
        account=account_num,
        start_trade=trade_idx,
        fee_paid=CHALLENGE_FEE,
        p1_pnl=0.0, p2_pnl=0.0, funded_gross_pnl=0.0,
        funded_payouts_count=0, funded_payouts_gross=0.0,
        challenge_bonus=0.0, fee_reimbursed=0.0,
        blown_phase=None, reason=None,
        p1_trades=0, p2_trades=0, funded_trades=0,
    )
    total_fees_paid += CHALLENGE_FEE
    global_equity   -= CHALLENGE_FEE

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    res1 = run_phase(trade_idx, ACCOUNT_SIZE, P1_TRAIL_DD, P1_PROFIT_TARGET)
    entry["p1_pnl"]    = res1["end_equity"] - ACCOUNT_SIZE
    entry["p1_trades"] = res1["end_idx"] - trade_idx
    for date, pnl, eq in res1["trades_log"]:
        equity_curve.append((account_num, "p1", date, pnl, eq))
    trade_idx = res1["end_idx"]

    if res1["blown"]:
        entry["blown_phase"] = "p1"
        entry["reason"]      = "daily_dd" if res1["daily_blown"] else "max_dd"
        account_log.append(entry)
        continue   # buy new account

    if not res1["target_hit"]:
        entry["blown_phase"] = "incomplete"; entry["reason"] = "trades_exhausted"
        account_log.append(entry)
        break

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    res2 = run_phase(trade_idx, ACCOUNT_SIZE, P2_TRAIL_DD, P2_PROFIT_TARGET)
    entry["p2_pnl"]    = res2["end_equity"] - ACCOUNT_SIZE
    entry["p2_trades"] = res2["end_idx"] - trade_idx
    for date, pnl, eq in res2["trades_log"]:
        equity_curve.append((account_num, "p2", date, pnl, eq))
    trade_idx = res2["end_idx"]

    if res2["blown"]:
        entry["blown_phase"] = "p2"
        entry["reason"]      = "daily_dd" if res2["daily_blown"] else "max_dd"
        account_log.append(entry)
        continue

    if not res2["target_hit"]:
        entry["blown_phase"] = "incomplete"; entry["reason"] = "trades_exhausted"
        account_log.append(entry)
        break

    # ── Challenge passed → fee reimbursed + challenge bonus ──────────────────
    challenge_pnl = entry["p1_pnl"] + entry["p2_pnl"]
    bonus = max(0, challenge_pnl * CHALLENGE_BONUS_RATE)
    entry["fee_reimbursed"]   = CHALLENGE_FEE
    entry["challenge_bonus"]  = bonus
    total_fees_reimbursed    += CHALLENGE_FEE
    total_challenge_bonus    += bonus
    global_equity            += CHALLENGE_FEE + bonus   # cash received

    # ── Funded phase (loop until blown or trades run out) ─────────────────────
    funded_equity = ACCOUNT_SIZE   # reset to fresh $100k
    funded_gross  = 0.0
    payouts_count = 0
    payouts_gross = 0.0

    while trade_idx < N:
        res_f = run_phase(trade_idx, funded_equity, None, None)
        for date, pnl, eq in res_f["trades_log"]:
            equity_curve.append((account_num, "funded", date, pnl, eq))
        entry["funded_trades"] += res_f["end_idx"] - trade_idx
        trade_idx  = res_f["end_idx"]
        funded_equity = res_f["end_equity"]

        if res_f["blown"]:
            # Record partial funded pnl before blowing
            funded_gross += funded_equity - ACCOUNT_SIZE  # this will be negative
            entry["blown_phase"] = "funded"
            entry["reason"]      = "daily_dd" if res_f["daily_blown"] else "max_dd"
            break

        if res_f["target_hit"]:
            # ── Payout ──────────────────────────────────────────────────────
            profit      = funded_equity - ACCOUNT_SIZE
            trader_cut  = profit * PROFIT_SPLIT
            payouts_count          += 1
            payouts_gross          += profit
            funded_gross           += profit
            total_payouts_received += trader_cut
            total_payouts_gross    += profit
            global_equity          += trader_cut
            payout_log.append(dict(
                account=account_num,
                payout_num=payouts_count,
                gross_profit=round(profit, 2),
                trader_gets=round(trader_cut, 2),
                trade_idx=trade_idx,
            ))
            funded_equity = ACCOUNT_SIZE   # reset equity after payout
            continue

        # Trades exhausted during funded phase
        entry["blown_phase"] = "incomplete"; entry["reason"] = "trades_exhausted"
        funded_gross += funded_equity - ACCOUNT_SIZE
        break

    entry["funded_gross_pnl"]     = funded_gross
    entry["funded_payouts_count"] = payouts_count
    entry["funded_payouts_gross"] = payouts_gross
    account_log.append(entry)

    if entry["blown_phase"] == "incomplete" or entry["blown_phase"] is None:
        break   # no more trades


# ── Compile results ────────────────────────────────────────────────────────────
al = pd.DataFrame(account_log)
pl = pd.DataFrame(payout_log) if payout_log else pd.DataFrame(
    columns=["account","payout_num","gross_profit","trader_gets","trade_idx"])

blown_p1     = (al["blown_phase"] == "p1").sum()
blown_p2     = (al["blown_phase"] == "p2").sum()
blown_funded = (al["blown_phase"] == "funded").sum()
total_blown  = blown_p1 + blown_p2 + blown_funded
passed_chal  = ((al["blown_phase"] != "p1") &
                (al["blown_phase"] != "incomplete") &
                (al["p2_pnl"] > 0)).sum()   # got through both phases at least once

total_accounts = len(al)
total_fees     = total_fees_paid
total_payouts  = total_payouts_received
net_cash       = global_equity    # final net position of trader

print()
print("=" * 60)
print("  FUNDED ACCOUNT SIMULATION — RESULTS")
print("=" * 60)
print()
print("── Account Attempts ──────────────────────────────────────")
print(f"  Total accounts bought      : {total_accounts}")
print(f"  Blown in Phase 1           : {blown_p1}")
print(f"  Blown in Phase 2           : {blown_p2}")
print(f"  Blown while funded         : {blown_funded}")
print(f"  Total accounts blown       : {total_blown}")
print(f"  Passed both phases         : {passed_chal}")
print()
print("── Financials ────────────────────────────────────────────")
print(f"  Total fees paid            : ${total_fees:>12,.0f}")
print(f"  Fees reimbursed            : ${total_fees_reimbursed:>12,.0f}")
print(f"  Challenge bonuses (15%)    : ${total_challenge_bonus:>12,.2f}")
print(f"  Gross payouts extracted    : ${total_payouts_gross:>12,.2f}")
print(f"  Trader payouts (90% split) : ${total_payouts_received:>12,.2f}")
print(f"  Net cash position          : ${net_cash:>12,.2f}")
print()
print("── Payout Summary ────────────────────────────────────────")
print(f"  Total payouts made         : {len(pl)}")
if len(pl) > 0:
    print(f"  Avg payout (trader cut)    : ${pl['trader_gets'].mean():>10,.2f}")
    print(f"  Largest payout             : ${pl['trader_gets'].max():>10,.2f}")
    print(f"  Smallest payout            : ${pl['trader_gets'].min():>10,.2f}")
print()

# ── Per-account table ──────────────────────────────────────────────────────────
summary_cols = ["account","fee_paid","fee_reimbursed","challenge_bonus",
                "p1_pnl","p2_pnl","funded_gross_pnl","funded_payouts_count",
                "funded_payouts_gross","blown_phase","reason"]
al_display = al[summary_cols].copy()
al_display = al_display.round(2)

print("── Per-Account Detail ────────────────────────────────────")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.2f}".format)
print(al_display.to_string(index=False))

# ── Save outputs ───────────────────────────────────────────────────────────────
al.to_csv(RESULTS_DIR / "account_log.csv", index=False)
pl.to_csv(RESULTS_DIR / "payout_log.csv", index=False)

stats = dict(
    total_accounts_bought=int(total_accounts),
    blown_p1=int(blown_p1),
    blown_p2=int(blown_p2),
    blown_funded=int(blown_funded),
    total_blown=int(total_blown),
    passed_challenge=int(passed_chal),
    total_fees_paid=round(total_fees, 2),
    total_fees_reimbursed=round(total_fees_reimbursed, 2),
    total_challenge_bonus=round(total_challenge_bonus, 2),
    gross_payouts_extracted=round(total_payouts_gross, 2),
    trader_payouts_received=round(total_payouts_received, 2),
    total_payouts_count=len(pl),
    net_cash_position=round(net_cash, 2),
)
with open(RESULTS_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── Equity curve chart ─────────────────────────────────────────────────────────
if equity_curve:
    ec_df = pd.DataFrame(equity_curve, columns=["account","phase","date","pnl","equity"])
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    # Build a monotonic running equity across all accounts
    # (cash perspective: track trader's net cash, not account equity)
    # Re-derive cash flow per trade event
    cash_events = []
    running_cash = 0.0

    for _, row in al.iterrows():
        running_cash -= CHALLENGE_FEE
        cash_events.append(running_cash)

    # Simple approach: plot account equity for the longest funded run
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Panel 1: Equity within each account attempt (overlay for funded phases)
    ax = axes[0]
    colors = {"p1": "#aaaaff", "p2": "#ffaaaa", "funded": "#44cc88"}
    ec_df_funded = ec_df[ec_df["phase"] == "funded"]
    for acct_id, grp in ec_df.groupby("account"):
        phase_grp = grp[grp["phase"].isin(["p1","p2"])]
        for ph, sg in phase_grp.groupby("phase"):
            ax.plot(sg["date"], sg["equity"], lw=0.5, color=colors[ph], alpha=0.4)
        fgrp = grp[grp["phase"] == "funded"]
        if not fgrp.empty:
            ax.plot(fgrp["date"], fgrp["equity"], lw=0.7, color=colors["funded"], alpha=0.6)
    ax.axhline(ACCOUNT_SIZE, color="gray", lw=0.8, ls="--", label="$100k baseline")
    ax.axhline(FUNDED_FLOOR, color="red", lw=0.8, ls=":", label="Funded floor $90k")
    ax.set_title("Equity per Account Attempt (blue=P1, red=P2, green=Funded)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Trader's running net cash position
    ax2 = axes[1]
    cash_timeline = []
    cash = 0.0
    for _, row in al.iterrows():
        cash -= row["fee_paid"]
        cash += row["fee_reimbursed"] + row["challenge_bonus"]
    # Rebuild per-payout timeline
    cash2 = 0.0
    cash_timeline = [(pd.Timestamp("2016-01-01"), 0.0)]
    acct_fees = al.set_index("account")["fee_paid"]
    acct_reimb = al.set_index("account")["fee_reimbursed"]
    acct_bonus = al.set_index("account")["challenge_bonus"]

    # Use payout log dates where possible
    if len(pl) > 0:
        pl2 = pl.copy()
        pl2["trade_date"] = pd.to_datetime(
            trades.loc[pl2["trade_idx"].clip(0, N-1), "date"].values)
        for _, p in pl2.iterrows():
            cash2 += p["trader_gets"]
            cash_timeline.append((p["trade_date"], cash2))
    cumcash = [0]
    for _, row in al.iterrows():
        cumcash.append(cumcash[-1] - row["fee_paid"] + row["fee_reimbursed"] + row["challenge_bonus"])
    cumcash_net = []
    run = 0.0
    for _, row in al.iterrows():
        run -= row["fee_paid"]
        run += row["fee_reimbursed"] + row["challenge_bonus"]
        cumcash_net.append(run)

    # Simple bar: net cash per account
    ax2.bar(al["account"], al["funded_payouts_gross"] * PROFIT_SPLIT
            + al["fee_reimbursed"] + al["challenge_bonus"]
            - al["fee_paid"],
            color=["#44cc88" if v >= 0 else "#ee4444"
                   for v in (al["funded_payouts_gross"] * PROFIT_SPLIT
                              + al["fee_reimbursed"] + al["challenge_bonus"]
                              - al["fee_paid"])])
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Net Cash P&L per Account Attempt", fontsize=11)
    ax2.set_xlabel("Account #")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Cumulative net cash to trader
    ax3 = axes[2]
    per_acct_net = (al["funded_payouts_gross"] * PROFIT_SPLIT
                    + al["fee_reimbursed"] + al["challenge_bonus"]
                    - al["fee_paid"])
    cum_net = per_acct_net.cumsum().values
    ax3.plot(range(1, len(cum_net)+1), cum_net, marker="o", ms=4, color="#2255cc", lw=1.5)
    ax3.fill_between(range(1, len(cum_net)+1), cum_net,
                     where=[v >= 0 for v in cum_net], alpha=0.15, color="green")
    ax3.fill_between(range(1, len(cum_net)+1), cum_net,
                     where=[v < 0 for v in cum_net], alpha=0.15, color="red")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Cumulative Net Cash to Trader (across all accounts)", fontsize=11)
    ax3.set_xlabel("Account #")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "funded_sim_overview.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {RESULTS_DIR}/funded_sim_overview.png")

print(f"\nAll results written to {RESULTS_DIR}/")

# ── Final summary line ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  NET CASH TO TRADER  : ${net_cash:>10,.2f}")
print(f"  (fees ${total_fees:,.0f} − reimbursed ${total_fees_reimbursed:,.0f} + "
      f"payouts ${total_payouts_received:,.2f} + bonus ${total_challenge_bonus:,.2f})")
print("=" * 60)
