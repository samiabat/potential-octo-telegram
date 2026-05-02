"""
Funded Account Simulator — ICT 2022 Model, AM Session (2020-2025)

Uses the real chronological AM trade sequence (big_test_only_am/trades.csv)
filtered to trades from 2020-01-01 onwards, to simulate repeatedly buying a
$100k prop-firm challenge account and trading through challenge phases until
all trades are exhausted.

Prop-firm rules assumed:
  Account size          : $100,000
  Challenge fee         : $550 (per attempt)
  Phase 1 profit target : 8%  = $8,000   |  max DD (trailing) : 8%  = $8,000
  Phase 2 profit target : 5%  = $5,000   |  max DD (trailing) : 5%  = $5,000
  Funded max DD         : 10% = $10,000  (static — floor = $90,000)
  Daily DD limit        : 5%  = $5,000   (prop-firm rule — all phases, from SOD equity)
  Fee reimbursed        : yes, when both challenge phases are passed
  Challenge bonus       : 15% of net challenge P&L (P1+P2), paid on funding
  Profit split          : 90% to trader
  Payout trigger        : $5,000 funded profit (equity ≥ $105,000)

Personal risk management:
  Personal daily stop   : $2,000/day  — trader stops trading for the rest of the day
                          once the day's loss reaches $2k. The account is NOT blown;
                          prop-firm daily limit ($5k) is the actual account-killer.

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
DAILY_DD_LIMIT       = 5_000       # 5% of $100k — prop-firm rule (account blown if breached)
PERSONAL_DAILY_STOP  = 2_000       # trader's self-imposed stop: quit the day at -$2k (no blow)

PAYOUT_TRIGGER    = 5_000       # request payout when funded profit ≥ $5,000

RESULTS_DIR = Path("funded_sim_2020")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Load trades (2020-01-01 onwards) ──────────────────────────────────────────
START_DATE = pd.Timestamp("2020-01-01", tz="UTC")

df = pd.read_csv("big_test_only_am/trades.csv")
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df            = df[df["entry_time"] >= START_DATE].copy()
df["date"]    = df["entry_time"].dt.date
df["pnl"]     = df["r"] * RISK_PER_R      # $ P&L per trade
trades        = df.reset_index(drop=True)
N             = len(trades)

print(f"Loaded {N} trades (2020+) across {df['date'].nunique()} trading days "
      f"({df['date'].min()} → {df['date'].max()})")
print()

# ── Helper — simulate one "phase" from a given trade index ─────────────────────
def run_phase(trade_idx, start_equity, trail_dd_limit, profit_target,
              daily_dd_limit=DAILY_DD_LIMIT,
              personal_daily_stop=PERSONAL_DAILY_STOP):
    """
    Simulate one challenge phase (or funded period with profit_target=None).

    Prop-firm blow conditions (account lost):
      • daily_dd_limit  : day's loss from SOD equity exceeds prop-firm limit ($5k)
      • trail_dd_limit  : equity drops more than X below trailing peak (challenge phases)
      • funded floor    : equity falls below $90k (funded phase)

    Personal risk management:
      • personal_daily_stop : if day's loss reaches this amount ($2k), skip the rest of
                              today's trades and move to the next trading day (no blow).

    Returns dict with keys:
      end_idx       : next trade index after this phase
      end_equity    : final equity
      blown         : bool — hit a prop-firm DD limit (account gone)
      target_hit    : bool — hit the profit target
      trades_log    : list of (date, pnl, equity) tuples
      daily_blown   : bool — blown by the prop-firm daily DD specifically
      days_stopped  : int  — number of days trading was cut short by personal stop
    """
    equity = start_equity
    peak   = start_equity          # for trailing DD
    blown  = False
    daily_blown  = False
    target_hit   = False
    trades_log   = []
    days_stopped = 0               # count of days ended early by personal stop

    # Group trades by day for daily-DD enforcement
    remaining = trades.iloc[trade_idx:].copy()
    if remaining.empty:
        return dict(end_idx=trade_idx, end_equity=equity, blown=False,
                    target_hit=False, trades_log=[], daily_blown=False,
                    days_stopped=0)

    idx = trade_idx
    for date, day_grp in remaining.groupby("date"):
        sod_equity      = equity          # start-of-day for daily DD / personal stop
        day_blown       = False
        day_personally_stopped = False

        for row_idx in day_grp.index:
            row = trades.loc[row_idx]
            pnl = row["pnl"]
            equity += pnl
            peak = max(peak, equity)
            idx += 1
            trades_log.append((date, pnl, equity))

            day_loss = sod_equity - equity   # positive = loss on the day

            # ── Prop-firm daily DD check (account blown) ────────────
            if day_loss >= daily_dd_limit:
                blown = True; daily_blown = True; day_blown = True
                break

            # ── Phase DD check (trailing from peak) ─────────────────
            if trail_dd_limit is not None:
                if peak - equity > trail_dd_limit:
                    blown = True; day_blown = True
                    break

            # ── Funded static floor check ────────────────────────────
            if trail_dd_limit is None:          # funded — static floor
                if equity < FUNDED_FLOOR:
                    blown = True; day_blown = True
                    break

            # ── Personal daily stop (not a blow — just done for today)
            if personal_daily_stop is not None and day_loss >= personal_daily_stop:
                day_personally_stopped = True
                break

        if day_blown or blown:
            break

        if day_personally_stopped:
            days_stopped += 1
            # continue to next day — account still live

        # ── End-of-day profit target check ──────────────────────────
        if profit_target is not None:
            if equity - ACCOUNT_SIZE >= profit_target:
                target_hit = True
                break

        # ── Funded payout check (end of day) ─────────────────────────
        if profit_target is None:
            if equity - ACCOUNT_SIZE >= PAYOUT_TRIGGER:
                target_hit = True   # signals "payout ready"
                break

    return dict(end_idx=idx, end_equity=equity, blown=blown,
                target_hit=target_hit, trades_log=trades_log,
                daily_blown=daily_blown, days_stopped=days_stopped)


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
        days_stopped=0,    # total days cut short by personal $2k stop
    )
    total_fees_paid += CHALLENGE_FEE
    global_equity   -= CHALLENGE_FEE

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    res1 = run_phase(trade_idx, ACCOUNT_SIZE, P1_TRAIL_DD, P1_PROFIT_TARGET)
    entry["p1_pnl"]    = res1["end_equity"] - ACCOUNT_SIZE
    entry["p1_trades"] = res1["end_idx"] - trade_idx
    entry["days_stopped"] += res1["days_stopped"]
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
    entry["days_stopped"] += res2["days_stopped"]
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
        entry["funded_trades"]  += res_f["end_idx"] - trade_idx
        entry["days_stopped"]   += res_f["days_stopped"]
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
total_days_stopped = int(al["days_stopped"].sum())   # days cut short by personal stop

# ── Bankroll / Runway analysis ─────────────────────────────────────────────────
# Build the cumulative cash curve, one step per account outcome
al["net_per_account"] = (
    al["funded_payouts_gross"] * PROFIT_SPLIT
    + al["fee_reimbursed"]
    + al["challenge_bonus"]
    - al["fee_paid"]
)
cum_cash = al["net_per_account"].cumsum().values

# Peak, trough, and drawdown in cumulative cash
cum_peak = np.maximum.accumulate(cum_cash)
cash_dd   = cum_cash - cum_peak          # always ≤ 0

max_cash_dd        = float(cash_dd.min())           # worst drawdown (negative)
max_cash_dd_pct    = (max_cash_dd / cum_peak[np.argmin(cash_dd)]) * 100 if cum_peak[np.argmin(cash_dd)] > 0 else float("nan")

# How many consecutive blown accounts are there at worst?
consecutive_blown = []
run = 0
for _, row in al.iterrows():
    if row["blown_phase"] in ("p1", "p2"):
        run += 1
    else:
        if run > 0:
            consecutive_blown.append(run)
        run = 0
if run > 0:
    consecutive_blown.append(run)

max_consec_blown   = max(consecutive_blown) if consecutive_blown else 0
avg_consec_blown   = float(np.mean(consecutive_blown)) if consecutive_blown else 0.0

# Minimum cash reserve needed = deepest trough of cumulative cash curve
# (starting from 0, how far in the red did the trader go?)
min_cum_cash = float(cum_cash.min())
# "bankroll needed" is how much you must start with so you never go broke
bankroll_needed = max(0.0, -min_cum_cash)

# How many accounts in a row before first net positive?
breakeven_idx = next((i for i, v in enumerate(cum_cash) if v > 0), None)
accts_before_positive = (breakeven_idx + 1) if breakeven_idx is not None else total_accounts

# How long (accounts) did each "negative patch" last?
neg_patches = []
neg_start = None
for i, v in enumerate(cum_cash):
    if v < 0 and neg_start is None:
        neg_start = i
    elif v >= 0 and neg_start is not None:
        neg_patches.append(i - neg_start)
        neg_start = None
if neg_start is not None:
    neg_patches.append(len(cum_cash) - neg_start)

longest_negative_run = max(neg_patches) if neg_patches else 0

# Rolling worst 10-account window (simulate "I just had a bad streak")
window = min(10, total_accounts)
rolling_10 = [sum(al["net_per_account"].iloc[i:i+window]) for i in range(total_accounts - window + 1)]
worst_10_window = min(rolling_10) if rolling_10 else 0.0

# Fee cost for consecutive blown accounts (e.g. worst streak)
fee_per_streak = max_consec_blown * CHALLENGE_FEE

print()
print("=" * 60)
print(f"  FUNDED ACCOUNT SIMULATION — RESULTS")
print(f"  Prop-firm daily DD : $5,000  |  Personal daily stop : $2,000")
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
print("── Bankroll & Runway Analysis ────────────────────────────")
print(f"  Min cash ever needed       : ${bankroll_needed:>10,.2f}  ← starting capital to never go broke")
print(f"  Deepest cum. cash drawdown : ${max_cash_dd:>10,.2f}  ({max_cash_dd_pct:.1f}% of peak)" if not np.isnan(max_cash_dd_pct) else f"  Deepest cum. cash drawdown : ${max_cash_dd:>10,.2f}")
print(f"  Accounts before breakeven  : {accts_before_positive}  (first time cumulative cash went +)")
print(f"  Longest stretch in red     : {longest_negative_run} consecutive accounts")
print(f"  Max consecutive blown      : {max_consec_blown}  (cost ${fee_per_streak:,.0f} in fees alone)")
print(f"  Avg consecutive blown      : {avg_consec_blown:.1f}")
print(f"  Worst 10-account window    : ${worst_10_window:>10,.2f}  (net over any 10-account stretch)")
print(f"  Days stopped by $2k rule   : {total_days_stopped}  (trading halted early, account kept)")
print()

# ── Per-account table ──────────────────────────────────────────────────────────
summary_cols = ["account","fee_paid","fee_reimbursed","challenge_bonus",
                "p1_pnl","p2_pnl","funded_gross_pnl","funded_payouts_count",
                "funded_payouts_gross","net_per_account","blown_phase","reason"]
al_display = al[summary_cols].copy()
al_display["cumulative_cash"] = al_display["net_per_account"].cumsum().round(2)
al_display = al_display.round(2)

print("── Per-Account Detail (with running cumulative cash) ─────")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.2f}".format)
print(al_display.to_string(index=False))

# ── Save outputs ───────────────────────────────────────────────────────────────
al.to_csv(RESULTS_DIR / "account_log.csv", index=False)
pl.to_csv(RESULTS_DIR / "payout_log.csv", index=False)

stats = dict(
    prop_firm_daily_dd_limit=DAILY_DD_LIMIT,
    personal_daily_stop=PERSONAL_DAILY_STOP,
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
    bankroll_min_needed=round(bankroll_needed, 2),
    deepest_cash_drawdown=round(max_cash_dd, 2),
    accounts_before_breakeven=int(accts_before_positive),
    longest_stretch_in_red_accounts=int(longest_negative_run),
    max_consecutive_blown=int(max_consec_blown),
    avg_consecutive_blown=round(avg_consec_blown, 1),
    worst_10_account_window=round(worst_10_window, 2),
    total_days_stopped_by_personal_rule=int(total_days_stopped),
)
with open(RESULTS_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── Charts ─────────────────────────────────────────────────────────────────────
if equity_curve:
    ec_df = pd.DataFrame(equity_curve, columns=["account","phase","date","pnl","equity"])
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 20))

    # Panel 1: Account equity curves
    ax = axes[0]
    colors = {"p1": "#aaaaff", "p2": "#ffaaaa", "funded": "#44cc88"}
    for acct_id, grp in ec_df.groupby("account"):
        for ph, sg in grp.groupby("phase"):
            ax.plot(sg["date"], sg["equity"], lw=0.5 if ph != "funded" else 0.8,
                    color=colors[ph], alpha=0.4)
    ax.axhline(ACCOUNT_SIZE, color="gray", lw=0.8, ls="--", label="$100k baseline")
    ax.axhline(FUNDED_FLOOR, color="red",  lw=0.8, ls=":", label="Funded floor $90k")
    ax.set_title("Equity per Account Attempt  (blue=P1, red=P2, green=Funded)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Net cash P&L per account (bar chart)
    ax2 = axes[1]
    net_vals = al["net_per_account"].values
    bar_colors = ["#44cc88" if v >= 0 else "#ee4444" for v in net_vals]
    ax2.bar(al["account"], net_vals, color=bar_colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Net Cash P&L per Account Attempt  (green=profit, red=loss)", fontsize=11)
    ax2.set_xlabel("Account #")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Cumulative cash curve + drawdown shading
    ax3 = axes[2]
    x_idx = np.arange(1, len(cum_cash) + 1)
    ax3.plot(x_idx, cum_cash, marker="o", ms=4, color="#2255cc", lw=1.5, label="Cumulative net cash")
    ax3.fill_between(x_idx, cum_cash, 0,
                     where=(cum_cash >= 0), alpha=0.15, color="green", label="In profit")
    ax3.fill_between(x_idx, cum_cash, 0,
                     where=(cum_cash < 0), alpha=0.25, color="red", label="In drawdown")
    # Annotate deepest trough
    trough_idx = int(np.argmin(cum_cash))
    ax3.annotate(f"Deepest trough\n${cum_cash[trough_idx]:,.0f}",
                 xy=(x_idx[trough_idx], cum_cash[trough_idx]),
                 xytext=(x_idx[trough_idx]+1.5, cum_cash[trough_idx]-500),
                 fontsize=8, color="darkred",
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axhline(-bankroll_needed, color="orange", lw=1, ls="--",
                label=f"Min bankroll needed ${bankroll_needed:,.0f}")
    ax3.set_title("Cumulative Net Cash to Trader (across all accounts)", fontsize=11)
    ax3.set_xlabel("Account #")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel 4: Cash drawdown curve (how far below peak at each point)
    ax4 = axes[3]
    ax4.fill_between(x_idx, cash_dd, 0, alpha=0.4, color="red")
    ax4.plot(x_idx, cash_dd, color="darkred", lw=1)
    ax4.set_title("Cumulative Cash Drawdown from Peak (per account)", fontsize=11)
    ax4.set_xlabel("Account #")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.grid(True, alpha=0.3)

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
print(f"  MIN BANKROLL NEEDED : ${bankroll_needed:>10,.2f}  (to never run out of cash)")
print("=" * 60)
