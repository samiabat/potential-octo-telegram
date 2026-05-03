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
    swing_points,
    htf_bias, align_bias_to_ltf,
    liquidity_sweep, detect_mss,
    detect_fvgs, detect_order_blocks,
    build_event_list, build_level_event_list,
    most_recent_event_before, most_recent_level_event_before,
    first_event_between,
    FVG, OrderBlock,
)

# ---------------------------------------------------------------------------
# Optional killzone filter (ET / NY time)
# ---------------------------------------------------------------------------
KILLZONES_ET: dict[str, tuple[dtime, dtime]] = {
    "london": (dtime(3, 0),  dtime(5, 0)),
    "am":     (dtime(8, 0), dtime(12, 0)),   # expanded: 08:00–12:00 ET
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
    # Price-level metadata (for charts and SL analysis)
    sweep_level: float | None = None   # the 15m swing price that was swept (horizontal level)
    fvg_top: float | None = None       # top of the entry FVG on 1m
    fvg_bottom: float | None = None    # bottom of the entry FVG on 1m
    fvg_time: pd.Timestamp | None = None   # when the entry FVG formed
    setup_mss_tf: str = "15m"              # timeframe where MSS was detected


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


def _collect_displacement_pending(
    *,
    df1m_index: pd.DatetimeIndex,
    fvgs_by_idx: dict[int, list],
    obs_by_idx: dict[int, list],
    sweep_time: pd.Timestamp,
    mss_time: pd.Timestamp,
    direction: str,                 # 'long' | 'short'
    current_bar_i: int,
    atr: float,
    min_fvg_size_atr: float,
    fvg_max_age_bars: int,
    use_fvg: bool,
    use_ob: bool,
    diag: "Diag",
) -> list[dict]:
    """Scan the displacement leg [sweep_idx ... min(mss_idx, i)] for
    bias-aligned FVGs / OBs and return them as `pending` dicts.

    This is the canonical ICT 2022-Model entry: the FVG that lives inside
    the move which caused the MSS.  Without this back-scan we'd only see
    FVGs forming AFTER the arm fired, missing the actual displacement FVG.
    """
    # 1m bar indices for the sweep and MSS timestamps.  searchsorted with
    # side='left' picks the first 1m bar at-or-after the timestamp.
    sweep_idx = int(df1m_index.searchsorted(sweep_time, side="left"))
    mss_idx   = int(df1m_index.searchsorted(mss_time,   side="left"))
    upper     = min(mss_idx, current_bar_i)
    # Don't reach back further than fvg_max_age_bars from now — FVGs older
    # than the standard age cap would be purged on the very next iteration.
    lower     = max(sweep_idx, current_bar_i - fvg_max_age_bars)
    if lower > upper:
        return []

    wanted_fvg_dir = "bull" if direction == "long" else "bear"
    wanted_ob_dir  = wanted_fvg_dir
    out: list[dict] = []

    for j in range(lower, upper + 1):
        if use_fvg:
            for fvg in fvgs_by_idx.get(j, []):
                if fvg.direction != wanted_fvg_dir:
                    continue
                diag.fvg_candidates += 1
                fvg_size = fvg.top - fvg.bottom
                if atr > 0 and fvg_size < min_fvg_size_atr * atr:
                    diag.fvg_blocked_size += 1
                    continue
                out.append({"type": "fvg", "fvg": fvg, "dir": direction, "born": j})
        if use_ob:
            for ob in obs_by_idx.get(j, []):
                if ob.direction != wanted_ob_dir:
                    continue
                diag.ob_candidates += 1
                out.append({"type": "ob", "ob": ob, "dir": direction, "born": j})
    return out


def _nearest_swing_sl(
    sw_low_prices: np.ndarray,
    sw_high_prices: np.ndarray,
    direction: str,
    entry_px: float,
    bar_i: int,
    lookback: int = 60,
) -> float | None:
    """Find the nearest 1m swing low (for long) or swing high (for short)
    that is below/above the entry price within *lookback* bars.

    Returns the swing price level, or None if no qualifying swing is found.
    """
    if direction == "long":
        best: float | None = None
        for j in range(bar_i, max(bar_i - lookback, -1), -1):
            lvl = sw_low_prices[j]
            if not np.isnan(lvl) and lvl < entry_px:
                if best is None or lvl > best:   # closest (highest) swing low below entry
                    best = lvl
                    break   # fractal swings are in ascending index order; first one found is nearest
        return best
    else:  # short
        best = None
        for j in range(bar_i, max(bar_i - lookback, -1), -1):
            lvl = sw_high_prices[j]
            if not np.isnan(lvl) and lvl > entry_px:
                if best is None or lvl < best:
                    best = lvl
                    break
        return best


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
    # 15m sweep parameters
    sweep_window_15m: int = 20,       # bars on 15m for sweep detection (~5h)
    sweep_max_age: pd.Timedelta = pd.Timedelta(hours=6),
    # MSS parameters — default to 1m for faster confirmation
    mss_tf: str = "1m",               # timeframe for MSS: "1m", "5m", or "15m"
    mss_lookback: int = 50,           # bars on the MSS timeframe for lookback
    arm_window: pd.Timedelta = pd.Timedelta(minutes=90),   # time after MSS to look for entry
    # Entry filter: skip if price has already moved too far from sweep before MSS
    max_sweep_to_mss_atr: float = 4.0,   # max |close − sweep_level| in ATR(14, 1m) at arm time
    # 1m entry parameters
    min_fvg_size_atr: float = 0.2,    # FVG must be ≥ N × ATR(14) on 1m
    fvg_max_age_bars: int = 20,       # max 1m bars to wait for retrace (20 min)
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
    #
    # We build LEVEL event lists so the strategy knows the exact 15m
    # swing price that was swept — used for proper SL placement.
    # ------------------------------------------------------------------
    df15m_sw = liquidity_sweep(df15m, window=sweep_window_15m)
    # Shift both the flag and the level by 1 bar (leak-free)
    sweep_low_flag    = df15m_sw["sweep_low"].shift(1).fillna(False)
    sweep_low_lvl     = df15m_sw["sweep_low_level"].shift(1)
    sweep_high_flag   = df15m_sw["sweep_high"].shift(1).fillna(False)
    sweep_high_lvl    = df15m_sw["sweep_high_level"].shift(1)

    sweep_low_events  = build_level_event_list(sweep_low_flag,  sweep_low_lvl)   # bull setup
    sweep_high_events = build_level_event_list(sweep_high_flag, sweep_high_lvl)  # bear setup

    # ------------------------------------------------------------------
    # 4. MSS / CHoCH events on the requested timeframe (mss_tf)
    #
    # Leak-free: MSS uses close-of-bar data; shift by 1 so events are
    # only visible after the bar that generated them closes.
    # ------------------------------------------------------------------
    if mss_tf == "1m":
        df_mss = detect_mss(df1m, lookback=mss_lookback)
        mss_up_events = build_event_list(
            df_mss["mss_up"].shift(1).fillna(False))
        mss_dn_events = build_event_list(
            df_mss["mss_dn"].shift(1).fillna(False))
    elif mss_tf == "5m":
        df5m = resample(df1m, "5min")
        df5m_mss = detect_mss(df5m, lookback=mss_lookback)
        mss_up_events = build_event_list(
            df5m_mss["mss_up"].shift(1).fillna(False))
        mss_dn_events = build_event_list(
            df5m_mss["mss_dn"].shift(1).fillna(False))
    else:  # "15m"
        df15m_mss = detect_mss(df15m_sw, lookback=mss_lookback)
        mss_up_events = build_event_list(
            df15m_mss["mss_up"].shift(1).fillna(False))
        mss_dn_events = build_event_list(
            df15m_mss["mss_dn"].shift(1).fillna(False))

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
    # 5b. 1m swing points — used for SL placement below swing low /
    #     above swing high (no look-ahead: recorded at confirmation bar)
    # ------------------------------------------------------------------
    df1m_sw        = swing_points(df1m, n=2)
    sw_low_prices  = df1m_sw["swing_low_price"].values   # NaN where no swing
    sw_high_prices = df1m_sw["swing_high_price"].values

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
    arm_sweep_level: float | None = None  # 15m price level that was swept (for SL)
    arm_mss_time: pd.Timestamp | None = None
    arm_trades: int = 0                  # trades taken in current arm

    # Track which sweep timestamps have already produced a trade.
    # One sweep → at most one trade; once used the sweep is consumed.
    used_sweep_times: set[pd.Timestamp] = set()

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
            sw_result = most_recent_level_event_before(sweep_low_events, ts, sweep_max_age)
            if sw_result is not None:
                sw, sw_level = sw_result
                # Skip sweeps that already produced a trade (one sweep = one trade)
                if sw not in used_sweep_times:
                    diag.bars_sweep_found += 1
                    mss = first_event_between(mss_up_events, sw, ts)
                    if mss is not None:
                        # Move-size filter: skip if price has already run too far from
                        # the swept level before we'd enter (= stale move)
                        atr_now = _atr(h, l, i)
                        move_ok = (
                            atr_now == 0
                            or abs(c[i] - sw_level) <= max_sweep_to_mss_atr * atr_now
                        )
                        # New or refreshed bull arm
                        if move_ok and (arm_dir != "long" or mss != arm_mss_time):
                            same_sweep = (arm_dir == "long" and sw == arm_sweep_time)
                            arm_dir = "long"
                            arm_expires = mss + arm_window
                            arm_sweep_time = sw
                            arm_sweep_level = sw_level
                            arm_mss_time = mss
                            arm_trades = 0
                            # Keep pending FVGs when refreshing within the same sweep
                            # so existing retrace opportunities are not lost.
                            if not same_sweep:
                                pending = []
                            # Back-scan the displacement leg for the canonical
                            # ICT-2022 entry FVG (the FVG inside the move that
                            # caused the MSS).  Without this we'd silently miss
                            # FVGs whose 3rd candle sits between the sweep and
                            # the bar where the arm becomes visible.
                            pending.extend(_collect_displacement_pending(
                                df1m_index=df1m.index,
                                fvgs_by_idx=fvgs_by_idx,
                                obs_by_idx=obs_by_idx,
                                sweep_time=sw,
                                mss_time=mss,
                                direction="long",
                                current_bar_i=i,
                                atr=_atr(h, l, i),
                                min_fvg_size_atr=min_fvg_size_atr,
                                fvg_max_age_bars=fvg_max_age_bars,
                                use_fvg=use_fvg,
                                use_ob=use_ob,
                                diag=diag,
                            ))

        # Bear cascade: bias -1, sweep-high, MSS-dn
        elif bias == -1:
            sw_result = most_recent_level_event_before(sweep_high_events, ts, sweep_max_age)
            if sw_result is not None:
                sw, sw_level = sw_result
                # Skip sweeps that already produced a trade (one sweep = one trade)
                if sw not in used_sweep_times:
                    diag.bars_sweep_found += 1
                    mss = first_event_between(mss_dn_events, sw, ts)
                    if mss is not None:
                        atr_now = _atr(h, l, i)
                        move_ok = (
                            atr_now == 0
                            or abs(c[i] - sw_level) <= max_sweep_to_mss_atr * atr_now
                        )
                        if move_ok and (arm_dir != "short" or mss != arm_mss_time):
                            same_sweep = (arm_dir == "short" and sw == arm_sweep_time)
                            arm_dir = "short"
                            arm_expires = mss + arm_window
                            arm_sweep_time = sw
                            arm_sweep_level = sw_level
                            arm_mss_time = mss
                            arm_trades = 0
                            if not same_sweep:
                                pending = []
                            pending.extend(_collect_displacement_pending(
                                df1m_index=df1m.index,
                                fvgs_by_idx=fvgs_by_idx,
                                obs_by_idx=obs_by_idx,
                                sweep_time=sw,
                                mss_time=mss,
                                direction="short",
                                current_bar_i=i,
                                atr=_atr(h, l, i),
                                min_fvg_size_atr=min_fvg_size_atr,
                                fvg_max_age_bars=fvg_max_age_bars,
                                use_fvg=use_fvg,
                                use_ob=use_ob,
                                diag=diag,
                            ))

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
                            # SL: find the nearest 1m swing low below entry.
                            # Fall back to sweep level, then FVG bottom.
                            sw_sl = _nearest_swing_sl(
                                sw_low_prices, sw_high_prices,
                                "long", entry_px, i,
                            )
                            if sw_sl is not None and sw_sl < entry_px:
                                stop_px = sw_sl - 0.5 * atr
                            elif arm_sweep_level is not None and arm_sweep_level < entry_px:
                                stop_px = arm_sweep_level - 0.1 * atr
                            else:
                                stop_px = fvg.bottom - 0.1 * atr
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
                                    sweep_level=arm_sweep_level,
                                    fvg_top=fvg.top,
                                    fvg_bottom=fvg.bottom,
                                    fvg_time=fvg.time,
                                    setup_mss_tf=mss_tf,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                used_sweep_times.add(arm_sweep_time)  # consume sweep
                                diag.trades_taken += 1
                                entry_taken = True

                        elif p["dir"] == "short" and h[i] >= fvg.bottom and l[i] <= fvg.top:
                            entry_px = max(c[i], fvg.bottom)
                            # SL: find the nearest 1m swing high above entry.
                            sw_sl = _nearest_swing_sl(
                                sw_low_prices, sw_high_prices,
                                "short", entry_px, i,
                            )
                            if sw_sl is not None and sw_sl > entry_px:
                                stop_px = sw_sl + 0.5 * atr
                            elif arm_sweep_level is not None and arm_sweep_level > entry_px:
                                stop_px = arm_sweep_level + 0.1 * atr
                            else:
                                stop_px = fvg.top + 0.1 * atr
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
                                    sweep_level=arm_sweep_level,
                                    fvg_top=fvg.top,
                                    fvg_bottom=fvg.bottom,
                                    fvg_time=fvg.time,
                                    setup_mss_tf=mss_tf,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                used_sweep_times.add(arm_sweep_time)  # consume sweep
                                diag.trades_taken += 1
                                entry_taken = True

                    elif p["type"] == "ob" and open_trade is None:
                        ob = p["ob"]
                        if p["dir"] == "long" and l[i] <= ob.ob_high and h[i] >= ob.ob_low:
                            entry_px = min(c[i], ob.ob_high)
                            sw_sl = _nearest_swing_sl(
                                sw_low_prices, sw_high_prices,
                                "long", entry_px, i,
                            )
                            if sw_sl is not None and sw_sl < entry_px:
                                stop_px = sw_sl - 0.5 * atr
                            elif arm_sweep_level is not None and arm_sweep_level < entry_px:
                                stop_px = arm_sweep_level - 0.1 * atr
                            else:
                                stop_px = ob.ob_low - 0.1 * atr
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
                                    sweep_level=arm_sweep_level,
                                    fvg_top=None, fvg_bottom=None,
                                    fvg_time=None, setup_mss_tf=mss_tf,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                used_sweep_times.add(arm_sweep_time)  # consume sweep
                                diag.trades_taken += 1
                                entry_taken = True

                        elif p["dir"] == "short" and h[i] >= ob.ob_low and l[i] <= ob.ob_high:
                            entry_px = max(c[i], ob.ob_low)
                            sw_sl = _nearest_swing_sl(
                                sw_low_prices, sw_high_prices,
                                "short", entry_px, i,
                            )
                            if sw_sl is not None and sw_sl > entry_px:
                                stop_px = sw_sl + 0.5 * atr
                            elif arm_sweep_level is not None and arm_sweep_level > entry_px:
                                stop_px = arm_sweep_level + 0.1 * atr
                            else:
                                stop_px = ob.ob_high + 0.1 * atr
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
                                    sweep_level=arm_sweep_level,
                                    fvg_top=None, fvg_bottom=None,
                                    fvg_time=None, setup_mss_tf=mss_tf,
                                )
                                open_trade_bars = 0
                                arm_trades += 1
                                used_sweep_times.add(arm_sweep_time)  # consume sweep
                                diag.trades_taken += 1
                                entry_taken = True

                    if not entry_taken:
                        still_pending.append(p)

                pending = still_pending

        elif not is_armed:
            pending = [p for p in pending if (i - p["born"]) <= fvg_max_age_bars]

    return trades, diag
