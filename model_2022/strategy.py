"""ICT 2022 Model — fully rule-based multi-timeframe cascade.

Cascade (all gates are required; none is optional):
  1. HTF Bias (4H EMA cross) → determines direction (+1 bull / -1 bear)
  2. Liquidity Sweep (15m)   → price sweeps a recent 15m swing H/L and
                               rejects back inside
  3. MSS / CHoCH (15m)       → after the sweep, price breaks the opposing
                               swing structure on 15m (market-structure shift)
  4. FVG / OB Entry (1m)     → once the 15m MSS is confirmed, wait for
                               price to retrace into a bullish/bearish FVG
                               or Order Block on the 1m chart

Entry, stop, and target:
  - Entry = FVG inner edge (top of bull FVG, bottom of bear FVG)
            OR Order Block edge (ob_high for bull, ob_low for bear)
            — whichever the 1m candle touches first
  - Stop  = below OB low (bull) / above OB high (bear), with ATR floor
  - Target = Entry ± RR × risk  (fixed-RR exit, default 2.0)
  - Expiry: trade closes at close-of-bar if neither target nor stop hit
            within *trade_expiry_bars* 1m bars

No look-ahead bias: every signal uses only data available at that bar.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import time as dtime

import numpy as np
import pandas as pd

from .data_loader import resample
from .ict_primitives import (
    htf_bias, align_bias_to_ltf,
    liquidity_sweep, detect_mss,
    detect_fvgs, detect_order_blocks,
    build_event_list,
    most_recent_event_before,
    first_event_between,
    FVG, OrderBlock,
)

# ---------------------------------------------------------------------------
# Optional killzone filter (ET / NY time)
# ---------------------------------------------------------------------------
KILLZONES_ET: dict[str, tuple[dtime, dtime]] = {
    "london": (dtime(3, 0),  dtime(5, 0)),
    "am":     (dtime(9, 30), dtime(12, 0)),
    "pm":     (dtime(13, 0), dtime(16, 0)),
}


def _in_killzone(ts: pd.Timestamp) -> str | None:
    t = ts.time()
    for name, (start, end) in KILLZONES_ET.items():
        if start <= t < end:
            return name
    return None


# ---------------------------------------------------------------------------
# Trade & Diagnostics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    direction: str          # 'long' | 'short'
    entry: float
    stop: float
    target: float
    exit_price: float | None = None
    outcome: str | None = None
    killzone: str = "none"
    r_multiple: float = 0.0
    entry_type: str = "fvg"  # 'fvg' | 'ob'
    setup_sweep_time: pd.Timestamp | None = None
    setup_mss_time: pd.Timestamp | None = None


@dataclass
class Diag:
    total_1m_bars: int = 0
    bars_with_htf_bias: int = 0
    bars_sweep_found: int = 0
    bars_mss_found: int = 0       # armed bars
    fvg_candidates: int = 0
    ob_candidates: int = 0
    fvg_blocked_size: int = 0
    fvg_no_retrace: int = 0
    ob_no_retrace: int = 0
    trades_taken: int = 0
    setups_expired: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(highs: np.ndarray, lows: np.ndarray, i: int, n: int = 14) -> float:
    if i < n:
        return 0.0
    return float((highs[i - n: i] - lows[i - n: i]).mean())


def _close_trade(
    t: Trade,
    ts: pd.Timestamp,
    price: float,
    outcome: str,
    r: float,
) -> None:
    t.exit_time = ts
    t.exit_price = price
    t.outcome = outcome
    t.r_multiple = r


def _r_multiple(t: Trade, exit_price: float) -> float:
    risk = abs(t.entry - t.stop)
    if risk == 0:
        return 0.0
    if t.direction == "long":
        return (exit_price - t.entry) / risk
    return (t.entry - exit_price) / risk


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

def run_2022_model(
    df1m: pd.DataFrame,
    *,
    rr: float = 2.0,
    # HTF bias (4H EMA cross)
    ema_fast: int = 20,
    ema_slow: int = 50,
    # 15m sweep / MSS parameters
    sweep_window_15m: int = 20,       # bars on 15m for sweep detection (~5h)
    sweep_max_age: pd.Timedelta = pd.Timedelta(hours=6),
    mss_lookback_15m: int = 50,       # bars on 15m for MSS detection
    arm_window: pd.Timedelta = pd.Timedelta(hours=3),  # how long after MSS to look for entry
    # 1m entry parameters
    min_fvg_size_atr: float = 0.2,    # FVG must be ≥ N × ATR(14) on 1m
    fvg_max_age_bars: int = 30,       # max 1m bars to wait for retrace (30 min)
    trade_expiry_bars: int = 120,     # max 1m bars to hold a trade (2h)
    max_trades_per_arm: int = 1,      # max entries per armed setup
    cost_per_side_pts: float = 1.0,   # spread + slip (NAS points)
    min_stop_pts: float = 5.0,        # absolute minimum stop distance
    min_stop_atr_mult: float = 1.0,   # min stop = N × ATR(14) on 1m
    use_ob: bool = True,              # enable Order Block entries
    use_fvg: bool = True,             # enable FVG entries
    use_killzones: bool = False,      # restrict entries to NY/London sessions
    allowed_killzones: set[str] | None = None,  # subset of killzones; None = all
) -> tuple[list[Trade], Diag]:
    """Run the 2022 model on *df1m* and return (trades, diagnostics)."""

    # ------------------------------------------------------------------
    # 1. Build higher-timeframe frames
    # ------------------------------------------------------------------
    df15m = resample(df1m, "15min")
    df4h  = resample(df1m, "4h")

    # ------------------------------------------------------------------
    # 2. HTF bias on 4H (EMA cross: +1 bull, -1 bear, 0 flat)
    #
    # Leak-free: the EMA of a 4H bar uses that bar's *close*, which is
    # only known when the bar finishes.  We shift by 1 bar so the bias
    # becomes visible to the 1m walk only from the *next* 4H bar's open
    # (i.e. after the bar that computed it has actually closed).
    # ------------------------------------------------------------------
    bias_4h = htf_bias(df4h, ema_fast=ema_fast, ema_slow=ema_slow)
    bias_4h = bias_4h.shift(1).fillna(0).astype(int)   # no look-ahead
    bias_1m = align_bias_to_ltf(bias_4h, df1m.index)

    # ------------------------------------------------------------------
    # 3. 15m liquidity sweep events
    #
    # Leak-free: sweep detection uses the bar's *high* AND *close*, so
    # the signal is only known after the 15m bar closes.  Shifting by 1
    # moves each event to the *next* 15m bar's timestamp, which is the
    # earliest moment the signal is actually observable.
    # ------------------------------------------------------------------
    df15m_sw = liquidity_sweep(df15m, window=sweep_window_15m)
    sweep_low_events  = build_event_list(
        df15m_sw["sweep_low"].shift(1).fillna(False))   # bull setup
    sweep_high_events = build_event_list(
        df15m_sw["sweep_high"].shift(1).fillna(False))  # bear setup

    # ------------------------------------------------------------------
    # 4. 15m MSS events (computed on sweep-annotated frame so the same
    #    copy of the frame is used; MSS logic only reads OHLC)
    #
    # Leak-free: MSS uses close-of-bar data on the 15m frame; shift by 1
    # so events are only visible after the bar that generated them closes.
    # ------------------------------------------------------------------
    df15m_mss = detect_mss(df15m_sw, lookback=mss_lookback_15m)
    mss_up_events = build_event_list(
        df15m_mss["mss_up"].shift(1).fillna(False))   # bull MSS (CHoCH up)
    mss_dn_events = build_event_list(
        df15m_mss["mss_dn"].shift(1).fillna(False))   # bear MSS (CHoCH dn)

    # ------------------------------------------------------------------
    # 5. 1m FVGs and Order Blocks (full pre-computation, index lookup)
    # ------------------------------------------------------------------
    fvgs_all = detect_fvgs(df1m) if use_fvg else []
    obs_all  = detect_order_blocks(df1m) if use_ob else []

    fvgs_by_idx: dict[int, list[FVG]] = {}
    for f in fvgs_all:
        fvgs_by_idx.setdefault(f.idx, []).append(f)

    obs_by_idx: dict[int, list[OrderBlock]] = {}
    for o in obs_all:
        obs_by_idx.setdefault(o.idx, []).append(o)

    # ------------------------------------------------------------------
    # 6. Walk 1m bars
    # ------------------------------------------------------------------
    h = df1m["high"].values
    l = df1m["low"].values
    c = df1m["close"].values
    times = df1m.index

    diag = Diag()
    trades: list[Trade] = []

    open_trade: Trade | None = None
    open_trade_bars: int = 0

    # Armed-setup state
    arm_dir: str | None = None           # 'long' | 'short'
    arm_expires: pd.Timestamp | None = None
    arm_sweep_time: pd.Timestamp | None = None
    arm_mss_time: pd.Timestamp | None = None
    arm_trades: int = 0                  # trades taken in current arm

    # Pending 1m entries (FVGs / OBs waiting for a retrace)
    pending: list[dict] = []

    for i in range(len(df1m)):
        ts = times[i]
        diag.total_1m_bars += 1

        # --------------------------------------------------------------
        # A. Manage open trade
        # --------------------------------------------------------------
        if open_trade is not None:
            open_trade_bars += 1
            risk = abs(open_trade.entry - open_trade.stop)
            cost_r = (2 * cost_per_side_pts) / risk if risk > 0 else 0.0

            hit_sl = (
                (open_trade.direction == "long"  and l[i] <= open_trade.stop) or
                (open_trade.direction == "short" and h[i] >= open_trade.stop)
            )
            hit_tp = (
                (open_trade.direction == "long"  and h[i] >= open_trade.target) or
                (open_trade.direction == "short" and l[i] <= open_trade.target)
            )
            if hit_sl:
                _close_trade(open_trade, ts, open_trade.stop, "loss", -1.0 - cost_r)
                trades.append(open_trade)
                open_trade = None
                open_trade_bars = 0
            elif hit_tp:
                _close_trade(open_trade, ts, open_trade.target, "win", rr - cost_r)
                trades.append(open_trade)
                open_trade = None
                open_trade_bars = 0
            elif open_trade_bars >= trade_expiry_bars:
                _close_trade(open_trade, ts, c[i], "expired",
                             _r_multiple(open_trade, c[i]) - cost_r)
                trades.append(open_trade)
                open_trade = None
                open_trade_bars = 0

        # --------------------------------------------------------------
        # B. Update armed-setup state from cascade
        # --------------------------------------------------------------
        bias = bias_1m.iat[i]

        if bias != 0:
            diag.bars_with_htf_bias += 1

        # Bull cascade: bias +1, sweep-low, MSS-up
        if bias == 1:
            sw = most_recent_event_before(sweep_low_events, ts, sweep_max_age)
            if sw is not None:
                diag.bars_sweep_found += 1
                mss = first_event_between(mss_up_events, sw, ts)
                if mss is not None:
                    # New or refreshed bull arm
                    if arm_dir != "long" or mss != arm_mss_time:
                        arm_dir = "long"
                        arm_expires = mss + arm_window
                        arm_sweep_time = sw
                        arm_mss_time = mss
                        arm_trades = 0
                        pending = []   # reset pending from old arm

        # Bear cascade: bias -1, sweep-high, MSS-dn
        elif bias == -1:
            sw = most_recent_event_before(sweep_high_events, ts, sweep_max_age)
            if sw is not None:
                diag.bars_sweep_found += 1
                mss = first_event_between(mss_dn_events, sw, ts)
                if mss is not None:
                    if arm_dir != "short" or mss != arm_mss_time:
                        arm_dir = "short"
                        arm_expires = mss + arm_window
                        arm_sweep_time = sw
                        arm_mss_time = mss
                        arm_trades = 0
                        pending = []

        # Check arm expiry
        is_armed = (
            arm_dir is not None
            and arm_expires is not None
            and ts <= arm_expires
            and arm_trades < max_trades_per_arm
        )

        if is_armed:
            diag.bars_mss_found += 1
        else:
            # Arm expired — purge stale pending entries
            if pending and arm_expires is not None and ts > arm_expires:
                diag.setups_expired += len(pending)
                pending = []
            arm_dir = None

        # --------------------------------------------------------------
        # C. Collect new 1m FVG / OB candidates if armed
        # --------------------------------------------------------------
        if is_armed:
            atr = _atr(h, l, i)

            for fvg in fvgs_by_idx.get(i, []):
                wanted = "long" if fvg.direction == "bull" else "short"
                if wanted != arm_dir:
                    continue
                diag.fvg_candidates += 1
                fvg_size = fvg.top - fvg.bottom
                if atr > 0 and fvg_size < min_fvg_size_atr * atr:
                    diag.fvg_blocked_size += 1
                    continue
                pending.append({
                    "type": "fvg",
                    "fvg": fvg,
                    "dir": wanted,
                    "born": i,
                })

            if use_ob:
                for ob in obs_by_idx.get(i, []):
                    wanted = "long" if ob.direction == "bull" else "short"
                    if wanted != arm_dir:
                        continue
                    diag.ob_candidates += 1
                    pending.append({
                        "type": "ob",
                        "ob": ob,
                        "dir": wanted,
                        "born": i,
                    })

        # --------------------------------------------------------------
        # D. Try to fill from pending (only after arm is confirmed)
        # --------------------------------------------------------------
        if open_trade is None and is_armed:
            kz = _in_killzone(ts) if use_killzones else "no_filter"
            kz_allowed = (
                not use_killzones
                or (kz is not None
                    and (allowed_killzones is None or kz in allowed_killzones))
            )
            if kz_allowed:
                atr = _atr(h, l, i)
                min_stop = max(min_stop_atr_mult * atr, min_stop_pts)

                still_pending: list[dict] = []
                for p in pending:
                    age = i - p["born"]

                    if age > fvg_max_age_bars:
                        if p["type"] == "fvg":
                            diag.fvg_no_retrace += 1
                        else:
                            diag.ob_no_retrace += 1
                        continue
                    if age < 1:          # don't fill on formation bar
                        still_pending.append(p)
                        continue

                    entry_taken = False

                    if p["type"] == "fvg" and open_trade is None:
                        fvg = p["fvg"]
                        if p["dir"] == "long" and l[i] <= fvg.top and h[i] >= fvg.bottom:
                            entry_px = min(c[i], fvg.top)
                            stop_px  = fvg.bottom - 0.1 * atr
                            if entry_px - stop_px < min_stop:
                                stop_px = entry_px - min_stop
                            risk = entry_px - stop_px
                            if risk > 0:
                                tgt = entry_px + rr * risk
                                kz_label = kz if use_killzones else _in_killzone(ts) or "none"
                                open_trade = Trade(
                                    ts, None, "long", entry_px, stop_px, tgt,
                                    killzone=kz_label, entry_type="fvg",
                                    setup_sweep_time=arm_sweep_time,
                                    setup_mss_time=arm_mss_time,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                diag.trades_taken += 1
                                entry_taken = True

                        elif p["dir"] == "short" and h[i] >= fvg.bottom and l[i] <= fvg.top:
                            entry_px = max(c[i], fvg.bottom)
                            stop_px  = fvg.top + 0.1 * atr
                            if stop_px - entry_px < min_stop:
                                stop_px = entry_px + min_stop
                            risk = stop_px - entry_px
                            if risk > 0:
                                tgt = entry_px - rr * risk
                                kz_label = kz if use_killzones else _in_killzone(ts) or "none"
                                open_trade = Trade(
                                    ts, None, "short", entry_px, stop_px, tgt,
                                    killzone=kz_label, entry_type="fvg",
                                    setup_sweep_time=arm_sweep_time,
                                    setup_mss_time=arm_mss_time,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                diag.trades_taken += 1
                                entry_taken = True

                    elif p["type"] == "ob" and open_trade is None:
                        ob = p["ob"]
                        if p["dir"] == "long" and l[i] <= ob.ob_high and h[i] >= ob.ob_low:
                            entry_px = min(c[i], ob.ob_high)
                            stop_px  = ob.ob_low - 0.1 * atr
                            if entry_px - stop_px < min_stop:
                                stop_px = entry_px - min_stop
                            risk = entry_px - stop_px
                            if risk > 0:
                                tgt = entry_px + rr * risk
                                kz_label = kz if use_killzones else _in_killzone(ts) or "none"
                                open_trade = Trade(
                                    ts, None, "long", entry_px, stop_px, tgt,
                                    killzone=kz_label, entry_type="ob",
                                    setup_sweep_time=arm_sweep_time,
                                    setup_mss_time=arm_mss_time,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                diag.trades_taken += 1
                                entry_taken = True

                        elif p["dir"] == "short" and h[i] >= ob.ob_low and l[i] <= ob.ob_high:
                            entry_px = max(c[i], ob.ob_low)
                            stop_px  = ob.ob_high + 0.1 * atr
                            if stop_px - entry_px < min_stop:
                                stop_px = entry_px + min_stop
                            risk = stop_px - entry_px
                            if risk > 0:
                                tgt = entry_px - rr * risk
                                kz_label = kz if use_killzones else _in_killzone(ts) or "none"
                                open_trade = Trade(
                                    ts, None, "short", entry_px, stop_px, tgt,
                                    killzone=kz_label, entry_type="ob",
                                    setup_sweep_time=arm_sweep_time,
                                    setup_mss_time=arm_mss_time,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                diag.trades_taken += 1
                                entry_taken = True

                    if not entry_taken:
                        still_pending.append(p)

                pending = still_pending

        elif not is_armed:
            pending = [p for p in pending if (i - p["born"]) <= fvg_max_age_bars]

    return trades, diag
