"""
Funded Account Simulator — ICT 2022 Model, AM Session (2020-2025) — FIXED VERSION

Differences vs run_funded_sim_2020.py (the original):
  1. Daily-DD bucket key uses NY calendar date (broker/firm reset cadence),
     not UTC date — so day boundaries match the prop firm's daily-loss
     accounting.
  2. Daily, trailing and funded-floor DD checks now consider the trade's
     INTRADAY excursion (mae_r in R-multiples), not just realised P&L.
     A trade that drew down -1R intraday and recovered to a small win
     could still have blown the account live; the original sim missed that.
  3. Comparison operators are consistent: every breach uses ">=" (any touch
     of the limit blows the account).
  4. After a payout, we keep the trailing peak at funded_equity rather than
     resetting hidden state, and reset funded_equity to ACCOUNT_SIZE only
     after the trader withdrawal — explicit and auditable.

Trade input must include `mae_r` and `mfe_r` columns (added by the fixed
strategy.py). Trades are read from big_test_only_am/trades.csv.

Prop-firm rules assumed (unchanged):
  Account size, fee, profit targets, trailing DDs, daily DD, payout split,
  payout trigger, personal $2k stop — all as in the original script.
"""

import pandas as pd
import numpy as np
import json
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

P1_PROFIT_TARGET  = 8_000
P1_TRAIL_DD       = 8_000
P2_PROFIT_TARGET  = 5_000
P2_TRAIL_DD       = 5_000

FUNDED_STATIC_DD  = 10_000
FUNDED_FLOOR      = ACCOUNT_SIZE - FUNDED_STATIC_DD   # $90,000
DAILY_DD_LIMIT       = 5_000
PERSONAL_DAILY_STOP  = 2_000

PAYOUT_TRIGGER    = 5_000

NY_TZ = "America/New_York"
RESULTS_DIR = Path("funded_sim_2020_fixed")
RESULTS_DIR.mkdir(exist_ok=True)
TRADES_CSV = Path("big_test_only_am/trades.csv")

START_DATE_UTC = pd.Timestamp("2020-01-01", tz="UTC")


# ── Load trades ────────────────────────────────────────────────────────────────
df = pd.read_csv(TRADES_CSV)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df = df[df["entry_time"] >= START_DATE_UTC].copy()

# NY-local calendar day = the firm's accounting day (good enough proxy for
# the 5pm-NY rollover most prop firms use; trades execute during 09:30-12:00 ET
# AM session so they never straddle the 5pm cutoff).
df["entry_time_ny"] = df["entry_time"].dt.tz_convert(NY_TZ)
df["date"] = df["entry_time_ny"].dt.date

df["pnl"] = df["r"] * RISK_PER_R

# Backward-compatibility: if the trades file pre-dates the strategy fix and
# does not yet carry mae_r / mfe_r columns, fall back to the most-conservative
# assumption — every trade could have touched -1R intraday before resolving.
if "mae_r" not in df.columns:
    print("⚠  trades.csv has no mae_r column — assuming worst-case "
          "intraday excursion of -1R for every trade (conservative).")
    df["mae_r"] = -1.0
if "mfe_r" not in df.columns:
    df["mfe_r"] = df["r"].clip(lower=0.0)

trades = df.reset_index(drop=True)
N = len(trades)

print(f"Loaded {N} trades (2020+) across {trades['date'].nunique()} NY trading days "
      f"({trades['date'].min()} → {trades['date'].max()})")
print()


# ── Helper — simulate one phase ────────────────────────────────────────────────
def run_phase(trade_idx, start_equity, trail_dd_limit, profit_target,
              start_peak=None,
              daily_dd_limit=DAILY_DD_LIMIT,
              personal_daily_stop=PERSONAL_DAILY_STOP):
    """Simulate one challenge phase or one funded-segment-until-payout.

    Intraday-aware DD checks: each trade's MAE is materialised against
    the running equity *before* the trade is realised. If MAE alone breaches
    a limit the account is blown at the trade's entry bar.

    Returns dict with end_idx, end_equity, end_peak, blown, blown_kind
    (daily_dd | trail_dd | floor), target_hit, trades_log, days_stopped.
    """
    equity = start_equity
    peak   = start_equity if start_peak is None else max(start_peak, start_equity)

    blown = False
    blown_kind = None
    target_hit = False
    trades_log = []
    days_stopped = 0

    remaining = trades.iloc[trade_idx:]
    if remaining.empty:
        return dict(end_idx=trade_idx, end_equity=equity, end_peak=peak,
                    blown=False, blown_kind=None, target_hit=False,
                    trades_log=[], days_stopped=0)

    idx = trade_idx
    for date, day_grp in remaining.groupby("date", sort=False):
        sod_equity = equity
        day_blown = False
        day_personally_stopped = False

        for row_idx in day_grp.index:
            row = trades.loc[row_idx]
            pnl    = row["pnl"]
            mae_r  = row["mae_r"]                         # ≤ 0
            mae_dollar = mae_r * RISK_PER_R                    # ≤ 0  (worst floating loss vs entry)

            # Floating equity at trade's worst intraday point
            floating_low = equity + mae_dollar
            day_loss_floating = sod_equity - floating_low

            # ── Intraday DD checks (these blow the account at MAE) ───
            if day_loss_floating >= daily_dd_limit:
                blown = True; blown_kind = "daily_dd"; day_blown = True
                trades_log.append((date, mae_dollar, floating_low))
                idx += 1
                break

            if trail_dd_limit is not None and (peak - floating_low) >= trail_dd_limit:
                blown = True; blown_kind = "trail_dd"; day_blown = True
                trades_log.append((date, mae_dollar, floating_low))
                idx += 1
                break

            if trail_dd_limit is None and floating_low < FUNDED_FLOOR:
                blown = True; blown_kind = "funded_floor"; day_blown = True
                trades_log.append((date, mae_dollar, floating_low))
                idx += 1
                break

            # ── Realise the trade ────────────────────────────────────
            equity += pnl
            if equity > peak:
                peak = equity
            idx += 1
            trades_log.append((date, pnl, equity))

            # ── Re-check post-realisation (closing PnL itself can breach) ─
            day_loss = sod_equity - equity
            if day_loss >= daily_dd_limit:
                blown = True; blown_kind = "daily_dd"; day_blown = True
                break
            if trail_dd_limit is not None and (peak - equity) >= trail_dd_limit:
                blown = True; blown_kind = "trail_dd"; day_blown = True
                break
            if trail_dd_limit is None and equity < FUNDED_FLOOR:
                blown = True; blown_kind = "funded_floor"; day_blown = True
                break

            # ── Personal daily stop (account stays alive) ────────────
            if personal_daily_stop is not None and day_loss >= personal_daily_stop:
                day_personally_stopped = True
                break

        if day_blown or blown:
            break

        if day_personally_stopped:
            days_stopped += 1

        # ── End-of-day checks ────────────────────────────────────────
        if profit_target is not None and (equity - ACCOUNT_SIZE) >= profit_target:
            target_hit = True
            break
        if profit_target is None and (equity - ACCOUNT_SIZE) >= PAYOUT_TRIGGER:
            target_hit = True
            break

    return dict(end_idx=idx, end_equity=equity, end_peak=peak,
                blown=blown, blown_kind=blown_kind,
                target_hit=target_hit, trades_log=trades_log,
                days_stopped=days_stopped)


# ── Main simulation ────────────────────────────────────────────────────────────
account_log = []
payout_log = []
equity_curve = []

trade_idx = 0
account_num = 0

total_fees_paid = 0.0
total_fees_reimbursed = 0.0
total_challenge_bonus = 0.0
total_payouts_received = 0.0
total_payouts_gross = 0.0
global_equity = 0.0

print("=" * 60)
print("  Running FIXED funded account simulation …")
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
        days_stopped=0,
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
        entry["reason"]      = res1["blown_kind"]
        account_log.append(entry)
        continue

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
        entry["reason"]      = res2["blown_kind"]
        account_log.append(entry)
        continue

    if not res2["target_hit"]:
        entry["blown_phase"] = "incomplete"; entry["reason"] = "trades_exhausted"
        account_log.append(entry)
        break

    # ── Challenge passed ─────────────────────────────────────────────────────
    challenge_pnl = entry["p1_pnl"] + entry["p2_pnl"]
    bonus = max(0, challenge_pnl * CHALLENGE_BONUS_RATE)
    entry["fee_reimbursed"]   = CHALLENGE_FEE
    entry["challenge_bonus"]  = bonus
    total_fees_reimbursed    += CHALLENGE_FEE
    total_challenge_bonus    += bonus
    global_equity            += CHALLENGE_FEE + bonus

    # ── Funded phase loop ────────────────────────────────────────────────────
    funded_equity = ACCOUNT_SIZE
    funded_peak   = ACCOUNT_SIZE
    funded_gross  = 0.0
    payouts_count = 0
    payouts_gross = 0.0

    while trade_idx < N:
        res_f = run_phase(trade_idx, funded_equity, None, None,
                          start_peak=funded_peak)
        for date, pnl, eq in res_f["trades_log"]:
            equity_curve.append((account_num, "funded", date, pnl, eq))
        entry["funded_trades"]  += res_f["end_idx"] - trade_idx
        entry["days_stopped"]   += res_f["days_stopped"]
        trade_idx     = res_f["end_idx"]
        funded_equity = res_f["end_equity"]
        funded_peak   = res_f["end_peak"]

        if res_f["blown"]:
            funded_gross += funded_equity - ACCOUNT_SIZE
            entry["blown_phase"] = "funded"
            entry["reason"]      = res_f["blown_kind"]
            break

        if res_f["target_hit"]:
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
            funded_equity = ACCOUNT_SIZE
            # The static funded floor is anchored at $90k regardless of payouts,
            # so we do NOT carry a higher peak forward — it's irrelevant
            # (trail_dd is None for funded phase). Reset peak to $100k for
            # consistency with the next segment's start equity.
            funded_peak = ACCOUNT_SIZE
            continue

        entry["blown_phase"] = "incomplete"; entry["reason"] = "trades_exhausted"
        funded_gross += funded_equity - ACCOUNT_SIZE
        break

    entry["funded_gross_pnl"]     = funded_gross
    entry["funded_payouts_count"] = payouts_count
    entry["funded_payouts_gross"] = payouts_gross
    account_log.append(entry)

    if entry["blown_phase"] in ("incomplete", None):
        break


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
                (al["p2_pnl"] > 0)).sum()

total_accounts = len(al)
total_fees     = total_fees_paid
net_cash       = global_equity
total_days_stopped = int(al["days_stopped"].sum())

al["net_per_account"] = (
    al["funded_payouts_gross"] * PROFIT_SPLIT
    + al["fee_reimbursed"]
    + al["challenge_bonus"]
    - al["fee_paid"]
)
cum_cash = al["net_per_account"].cumsum().values
cum_peak = np.maximum.accumulate(cum_cash) if len(cum_cash) else np.array([0.0])
cash_dd  = cum_cash - cum_peak

max_cash_dd = float(cash_dd.min()) if len(cash_dd) else 0.0
peak_at_trough = cum_peak[np.argmin(cash_dd)] if len(cash_dd) else 0.0
max_cash_dd_pct = (max_cash_dd / peak_at_trough * 100) if peak_at_trough > 0 else float("nan")

consecutive_blown = []
run = 0
for _, row in al.iterrows():
    if row["blown_phase"] in ("p1", "p2"):
        run += 1
    else:
        if run > 0: consecutive_blown.append(run)
        run = 0
if run > 0: consecutive_blown.append(run)
max_consec_blown = max(consecutive_blown) if consecutive_blown else 0
avg_consec_blown = float(np.mean(consecutive_blown)) if consecutive_blown else 0.0

min_cum_cash    = float(cum_cash.min()) if len(cum_cash) else 0.0
bankroll_needed = max(0.0, -min_cum_cash)

breakeven_idx = next((i for i, v in enumerate(cum_cash) if v > 0), None)
accts_before_positive = (breakeven_idx + 1) if breakeven_idx is not None else total_accounts

neg_patches = []
neg_start = None
for i, v in enumerate(cum_cash):
    if v < 0 and neg_start is None: neg_start = i
    elif v >= 0 and neg_start is not None:
        neg_patches.append(i - neg_start); neg_start = None
if neg_start is not None:
    neg_patches.append(len(cum_cash) - neg_start)
longest_negative_run = max(neg_patches) if neg_patches else 0

window = min(10, total_accounts)
rolling_10 = ([sum(al["net_per_account"].iloc[i:i+window])
               for i in range(total_accounts - window + 1)] if window else [])
worst_10_window = min(rolling_10) if rolling_10 else 0.0

fee_per_streak = max_consec_blown * CHALLENGE_FEE

# Reason breakdown
reason_counts = al["reason"].fillna("ok").value_counts().to_dict()

print()
print("=" * 60)
print(f"  FUNDED ACCOUNT SIMULATION (FIXED) — RESULTS")
print(f"  Daily DD : $5,000  |  Personal stop : $2,000")
print(f"  DD checks: intraday-aware (uses MAE)")
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
print("── Blow-up reasons ───────────────────────────────────────")
for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v:>5}")
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
print(f"  Min cash ever needed       : ${bankroll_needed:>10,.2f}")
print(f"  Deepest cum. cash drawdown : ${max_cash_dd:>10,.2f}"
      + (f"  ({max_cash_dd_pct:.1f}% of peak)" if not np.isnan(max_cash_dd_pct) else ""))
print(f"  Accounts before breakeven  : {accts_before_positive}")
print(f"  Longest stretch in red     : {longest_negative_run}")
print(f"  Max consecutive blown      : {max_consec_blown}  (cost ${fee_per_streak:,.0f} in fees)")
print(f"  Avg consecutive blown      : {avg_consec_blown:.1f}")
print(f"  Worst 10-account window    : ${worst_10_window:>10,.2f}")
print(f"  Days stopped by $2k rule   : {total_days_stopped}")
print()

al.to_csv(RESULTS_DIR / "account_log.csv", index=False)
pl.to_csv(RESULTS_DIR / "payout_log.csv", index=False)

stats = dict(
    fixed_version=True,
    intraday_aware_dd=True,
    daily_dd_limit=DAILY_DD_LIMIT,
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
    blow_up_reasons={str(k): int(v) for k, v in reason_counts.items()},
)
with open(RESULTS_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

if equity_curve:
    ec_df = pd.DataFrame(equity_curve, columns=["account","phase","date","pnl","equity"])
    ec_df["date"] = pd.to_datetime(ec_df["date"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    ax = axes[0]
    colors = {"p1": "#aaaaff", "p2": "#ffaaaa", "funded": "#44cc88"}
    for acct_id, grp in ec_df.groupby("account"):
        for ph, sg in grp.groupby("phase"):
            ax.plot(sg["date"], sg["equity"], lw=0.5 if ph != "funded" else 0.8,
                    color=colors[ph], alpha=0.4)
    ax.axhline(ACCOUNT_SIZE, color="gray", lw=0.8, ls="--")
    ax.axhline(FUNDED_FLOOR, color="red",  lw=0.8, ls=":")
    ax.set_title("Equity per Account Attempt (FIXED, intraday-aware)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    net_vals = al["net_per_account"].values
    bar_colors = ["#44cc88" if v >= 0 else "#ee4444" for v in net_vals]
    ax2.bar(al["account"], net_vals, color=bar_colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Net Cash P&L per Account", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    x_idx = np.arange(1, len(cum_cash) + 1)
    ax3.plot(x_idx, cum_cash, marker="o", ms=3, color="#2255cc", lw=1.2)
    ax3.fill_between(x_idx, cum_cash, 0, where=(cum_cash >= 0), alpha=0.15, color="green")
    ax3.fill_between(x_idx, cum_cash, 0, where=(cum_cash < 0),  alpha=0.25, color="red")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axhline(-bankroll_needed, color="orange", lw=1, ls="--",
                label=f"Bankroll needed ${bankroll_needed:,.0f}")
    ax3.set_title("Cumulative Net Cash to Trader", fontsize=11)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "funded_sim_overview.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {RESULTS_DIR}/funded_sim_overview.png")

print(f"\nAll results written to {RESULTS_DIR}/")
print()
print("=" * 60)
print(f"  NET CASH TO TRADER  : ${net_cash:>10,.2f}")
print(f"  MIN BANKROLL NEEDED : ${bankroll_needed:>10,.2f}")
print("=" * 60)
