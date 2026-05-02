"""ICT Silver Bullet v2 — higher-frequency mechanical implementation.

The v1 model required: bias-aligned liquidity sweep + MSS + FVG retrace
inside a 1h window. That alignment is rare (~14 trades/yr).

v2 keeps the spirit of Silver Bullet but matches how it is actually
day-traded mechanically:

  killzone → FVG forms aligned with bias → retrace into FVG → fixed RR

Optional gates (toggle in config):
  - require_sweep:  prior-N-bar liquidity sweep before the FVG
  - require_mss:    market-structure shift before the FVG
  - bias_filter:    'strict' (block opposing), 'soft' (no filter), 'off'

Entry  = FVG entry edge (top for shorts, bottom for longs)
Stop   = FVG opposite edge + ATR pad   (tight, FVG-defined)
Target = entry +/- RR * risk           (fixed RR)

Per-trade R is reduced by 2 * cost_per_side_pts / risk to model spread + slip.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import time
import pandas as pd
import numpy as np

from .ict_primitives import (
    detect_fvgs, detect_mss, liquidity_sweep, htf_bias, align_bias_to_ltf,
)

KILLZONES_ET = {
    "london": (time(3, 0), time(4, 0)),
    "am":     (time(10, 0), time(11, 0)),
    "pm":     (time(14, 0), time(15, 0)),
}


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    direction: str
    entry: float
    stop: float
    target: float
    exit_price: float | None = None
    outcome: str | None = None
    killzone: str = ""
    r_multiple: float = 0.0


@dataclass
class Diag:
    killzone_bars: int = 0
    fvgs_in_kz: int = 0
    fvg_blocked_bias: int = 0
    fvg_blocked_sweep: int = 0
    fvg_blocked_mss: int = 0
    fvg_blocked_size: int = 0
    fvg_no_retrace: int = 0
    trades_taken: int = 0


def in_killzone(ts: pd.Timestamp) -> str | None:
    t = ts.time()
    for name, (start, end) in KILLZONES_ET.items():
        if start <= t < end:
            return name
    return None


def run_silver_bullet(
    df5: pd.DataFrame,
    df_htf: pd.DataFrame,
    rr: float = 2.0,
    bias_filter: str = "strict",        # 'strict' | 'soft' | 'off'
    require_sweep: bool = False,
    require_mss: bool = False,
    sweep_lookback: int = 24,
    fvg_max_age_bars: int = 12,         # 1h to retrace
    trade_expiry_bars: int = 24,        # 2h to resolve
    max_trades_per_kz: int = 1,
    cost_per_side_pts: float = 1.0,
    min_stop_atr_mult: float = 1.0,        # min stop = N * ATR(14)
    min_stop_pts_floor: float = 5.0,       # absolute floor in NAS points
    min_fvg_size_atr: float = 0.3,         # FVG must be >= N * ATR
    only_first_half_kz: bool = True,       # only enter in first 30m of killzone
) -> tuple[list[Trade], Diag]:
    bias = align_bias_to_ltf(htf_bias(df_htf), df5.index)

    df = liquidity_sweep(df5, window=sweep_lookback) if require_sweep else df5.copy()
    if require_sweep:
        sweep_high = df["sweep_high"].values
        sweep_low  = df["sweep_low"].values
    if require_mss:
        df = detect_mss(df, lookback=60)
        mss_up = df["mss_up"].values
        mss_dn = df["mss_dn"].values

    fvgs_all = detect_fvgs(df)
    fvgs_by_idx: dict[int, list] = {}
    for f in fvgs_all:
        fvgs_by_idx.setdefault(f.idx, []).append(f)

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    times = df.index

    diag = Diag()
    trades: list[Trade] = []
    open_trade: Trade | None = None
    open_trade_bars = 0

    # Per-killzone state
    last_kz: str | None = None
    kz_trades_count = 0
    pending_fvgs: list = []        # FVGs awaiting retrace, with metadata
    sweep_dir: str | None = None   # 'up'/'down' if a sweep just happened
    mss_dir: str | None = None

    for i in range(len(df)):
        ts = times[i]
        # ---- Manage open trade (works in or out of killzone) ----
        if open_trade is not None:
            open_trade_bars += 1
            risk = abs(open_trade.entry - open_trade.stop)
            cost_r = (2 * cost_per_side_pts) / risk if risk > 0 else 0.0
            hit_stop = (open_trade.direction == "long" and l[i] <= open_trade.stop) or \
                       (open_trade.direction == "short" and h[i] >= open_trade.stop)
            hit_tp   = (open_trade.direction == "long" and h[i] >= open_trade.target) or \
                       (open_trade.direction == "short" and l[i] <= open_trade.target)
            if hit_stop:
                _close(open_trade, ts, open_trade.stop, "loss", -1.0 - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0
            elif hit_tp:
                _close(open_trade, ts, open_trade.target, "win", rr - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0
            elif open_trade_bars >= trade_expiry_bars:
                _close(open_trade, ts, c[i], "expired", _r(open_trade, c[i]) - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0

        kz = in_killzone(ts)

        # ---- killzone transition ----
        if kz != last_kz:
            pending_fvgs = []
            sweep_dir = None
            mss_dir = None
            kz_trades_count = 0
            last_kz = kz
        if kz is None:
            continue

        diag.killzone_bars += 1
        b = bias.iat[i]

        # Track sweep/MSS state inside killzone
        if require_sweep:
            if sweep_low[i]:
                sweep_dir = "up"   # sell-side swept => expect bull reversal
            if sweep_high[i]:
                sweep_dir = "down"
        if require_mss:
            if mss_up[i]:
                mss_dir = "up"
            if mss_dn[i]:
                mss_dir = "down"

        # Add new FVGs formed this bar to pending list
        atr_now = _atr(df, i)
        for fvg in fvgs_by_idx.get(i, []):
            diag.fvgs_in_kz += 1
            wanted = "long" if fvg.direction == "bull" else "short"

            # FVG quality: size relative to ATR
            fvg_size = fvg.top - fvg.bottom
            if atr_now > 0 and fvg_size < min_fvg_size_atr * atr_now:
                diag.fvg_blocked_size += 1
                continue

            # Bias filter
            if bias_filter == "strict":
                if (wanted == "long" and b < 0) or (wanted == "short" and b > 0):
                    diag.fvg_blocked_bias += 1
                    continue
            elif bias_filter == "soft":
                pass  # informational only

            # Sweep gate
            if require_sweep:
                if (wanted == "long" and sweep_dir != "up") or \
                   (wanted == "short" and sweep_dir != "down"):
                    diag.fvg_blocked_sweep += 1
                    continue

            # MSS gate
            if require_mss:
                if (wanted == "long" and mss_dir != "up") or \
                   (wanted == "short" and mss_dir != "down"):
                    diag.fvg_blocked_mss += 1
                    continue

            pending_fvgs.append({"fvg": fvg, "dir": wanted, "born": i})

        # Optional: only allow entries in first half of killzone window
        in_entry_window = True
        if only_first_half_kz:
            kz_start, kz_end = KILLZONES_ET[kz]
            kz_minutes = (ts.time().hour - kz_start.hour) * 60 + \
                         (ts.time().minute - kz_start.minute)
            in_entry_window = kz_minutes < 30

        # Try to fill from pending FVGs (only one open trade at a time).
        # IMPORTANT: entry must be on a bar AFTER the FVG formation bar,
        # otherwise the FVG-formation bar trivially "touches" its own edge.
        if open_trade is None and kz_trades_count < max_trades_per_kz and in_entry_window:
            still_pending = []
            atr = _atr(df, i)
            min_stop_pts = max(min_stop_atr_mult * atr, min_stop_pts_floor)
            swing_lookback = 12  # 12 5m bars = 1h
            lo_window = df["low"].iloc[max(0, i - swing_lookback): i + 1].min()
            hi_window = df["high"].iloc[max(0, i - swing_lookback): i + 1].max()

            for p in pending_fvgs:
                fvg = p["fvg"]
                age = i - p["born"]
                if age > fvg_max_age_bars:
                    diag.fvg_no_retrace += 1
                    continue
                if age < 1:        # don't fill on formation bar
                    still_pending.append(p)
                    continue

                if p["dir"] == "long":
                    if l[i] <= fvg.top and h[i] >= fvg.bottom:
                        entry = min(c[i], fvg.top)
                        # Stop = below local swing low (ICT: below leg origin)
                        stop_raw = min(fvg.bottom, lo_window) - 0.1 * atr
                        # Enforce minimum stop distance
                        if (entry - stop_raw) < min_stop_pts:
                            stop_raw = entry - min_stop_pts
                        risk = entry - stop_raw
                        if risk > 0:
                            target = entry + rr * risk
                            open_trade = Trade(ts, None, "long", entry, stop_raw,
                                               target, killzone=kz)
                            open_trade_bars = 0
                            kz_trades_count += 1
                            diag.trades_taken += 1
                            continue
                else:
                    if h[i] >= fvg.bottom and l[i] <= fvg.top:
                        entry = max(c[i], fvg.bottom)
                        stop_raw = max(fvg.top, hi_window) + 0.1 * atr
                        if (stop_raw - entry) < min_stop_pts:
                            stop_raw = entry + min_stop_pts
                        risk = stop_raw - entry
                        if risk > 0:
                            target = entry - rr * risk
                            open_trade = Trade(ts, None, "short", entry, stop_raw,
                                               target, killzone=kz)
                            open_trade_bars = 0
                            kz_trades_count += 1
                            diag.trades_taken += 1
                            continue
                still_pending.append(p)
            pending_fvgs = still_pending
        else:
            pending_fvgs = [p for p in pending_fvgs if (i - p["born"]) <= fvg_max_age_bars]

    return trades, diag


def _close(t: Trade, ts: pd.Timestamp, price: float, outcome: str, r: float) -> None:
    t.exit_time = ts
    t.exit_price = price
    t.outcome = outcome
    t.r_multiple = r


def _r(t: Trade, exit_price: float) -> float:
    if t.direction == "long":
        risk = t.entry - t.stop
        return (exit_price - t.entry) / risk if risk else 0.0
    risk = t.stop - t.entry
    return (t.entry - exit_price) / risk if risk else 0.0


def _atr_pad(df: pd.DataFrame, i: int, n: int = 14, mult: float = 0.1) -> float:
    if i < n:
        return 0.0
    tr = (df["high"].iloc[i - n : i] - df["low"].iloc[i - n : i]).mean()
    return float(tr) * mult


def _atr(df: pd.DataFrame, i: int, n: int = 14) -> float:
    if i < n:
        return 0.0
    return float((df["high"].iloc[i - n : i] - df["low"].iloc[i - n : i]).mean())
