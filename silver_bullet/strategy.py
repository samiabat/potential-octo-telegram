"""ICT Silver Bullet implementation.

Rules (per Michael Huddleston):
  1. Trade only inside one of three killzones (NY local time):
        - London open SB:  03:00 - 04:00
        - AM SB:           10:00 - 11:00
        - PM SB:           14:00 - 15:00
  2. Establish HTF bias (1H EMA cross used here as a mechanical proxy for
     premium/discount + draw-on-liquidity).
  3. Inside the killzone, wait for liquidity to be swept against the bias
     (e.g. sell-side liquidity grabbed below a recent low when bias is bullish).
  4. Wait for displacement that shifts market structure (MSS / CHoCH) in the
     bias direction.
  5. Identify the FVG created by that displacement.
  6. Limit-entry on retrace into the FVG; stop beyond the swept extreme;
     target the next opposing liquidity pool with a minimum 2R fixed RR.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import time
import pandas as pd

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
    direction: str           # 'long' or 'short'
    entry: float
    stop: float
    target: float
    exit_price: float | None = None
    outcome: str | None = None   # 'win' / 'loss' / 'expired'
    killzone: str = ""
    r_multiple: float = 0.0


def in_killzone(ts: pd.Timestamp) -> str | None:
    t = ts.time()
    for name, (start, end) in KILLZONES_ET.items():
        if start <= t < end:
            return name
    return None


def run_silver_bullet(
    df5: pd.DataFrame,
    df1h: pd.DataFrame,
    rr: float = 2.0,
    sweep_lookback: int = 24,         # 24*5m = 2h prior context
    fvg_max_age_bars: int = 6,        # FVG must fill within ~30m of formation
    trade_expiry_bars: int = 36,      # max 3h to resolve
    risk_per_trade: float = 100.0,    # $ per R for sizing display
    cost_per_side_pts: float = 1.0,   # NAS points slippage+spread per side
) -> list[Trade]:
    """Walk forward over 5m bars, generate Silver Bullet trades."""
    bias_1h = htf_bias(df1h)
    bias = align_bias_to_ltf(bias_1h, df5.index)

    df = liquidity_sweep(df5, window=sweep_lookback)
    df = detect_mss(df, lookback=60)
    fvgs_all = detect_fvgs(df)
    # Index FVGs by formation bar for fast lookup
    fvgs_by_idx: dict[int, list] = {}
    for f in fvgs_all:
        fvgs_by_idx.setdefault(f.idx, []).append(f)

    trades: list[Trade] = []
    open_trade: Trade | None = None
    open_trade_bars = 0

    # Per-killzone state machine
    state: dict[str, dict] = {k: _empty_kz_state() for k in KILLZONES_ET}
    last_kz: str | None = None

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    o = df["open"].values
    times = df.index

    for i in range(len(df)):
        ts = times[i]
        # ---- Manage open trade ----
        if open_trade is not None:
            open_trade_bars += 1
            hit_stop = (open_trade.direction == "long" and l[i] <= open_trade.stop) or \
                       (open_trade.direction == "short" and h[i] >= open_trade.stop)
            hit_tp   = (open_trade.direction == "long" and h[i] >= open_trade.target) or \
                       (open_trade.direction == "short" and l[i] <= open_trade.target)
            risk = abs(open_trade.entry - open_trade.stop)
            cost_r = (2 * cost_per_side_pts) / risk if risk > 0 else 0.0
            if hit_stop and hit_tp:
                _close(open_trade, ts, open_trade.stop, "loss", -1.0 - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0
            elif hit_stop:
                _close(open_trade, ts, open_trade.stop, "loss", -1.0 - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0
            elif hit_tp:
                _close(open_trade, ts, open_trade.target, "win", rr - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0
            elif open_trade_bars >= trade_expiry_bars:
                _close(open_trade, ts, c[i], "expired",
                       _r(open_trade, c[i]) - cost_r)
                trades.append(open_trade); open_trade = None; open_trade_bars = 0

        kz = in_killzone(ts)
        if kz is None:
            last_kz = None
            continue

        # Reset killzone state on entry
        if kz != last_kz:
            state[kz] = _empty_kz_state()
            last_kz = kz

        s = state[kz]
        b = bias.iat[i]

        # 1) Detect liquidity sweep aligned with future MSS direction
        if df["sweep_low"].iat[i] and b >= 0:
            s["sweep"] = ("low", i, l[i])
        if df["sweep_high"].iat[i] and b <= 0:
            s["sweep"] = ("high", i, h[i])

        # 2) After sweep, look for MSS in opposite direction (= bias direction)
        if s["sweep"] is not None and s["mss_idx"] is None:
            sweep_kind = s["sweep"][0]
            if sweep_kind == "low" and df["mss_up"].iat[i]:
                s["mss_idx"] = i
                s["dir"] = "long"
            elif sweep_kind == "high" and df["mss_dn"].iat[i]:
                s["mss_idx"] = i
                s["dir"] = "short"

        # 3) After MSS, find the FVG inside the displacement and wait for retrace
        if s["mss_idx"] is not None and open_trade is None:
            # Look through FVGs formed between sweep bar and current bar
            sweep_idx = s["sweep"][1]
            candidate = None
            for j in range(sweep_idx, i + 1):
                for fvg in fvgs_by_idx.get(j, []):
                    if s["dir"] == "long" and fvg.direction == "bull":
                        candidate = fvg
                    elif s["dir"] == "short" and fvg.direction == "bear":
                        candidate = fvg
            if candidate is not None and (i - candidate.idx) <= fvg_max_age_bars:
                # Check retrace: long entry if low touches FVG top zone
                if s["dir"] == "long" and l[i] <= candidate.top and h[i] >= candidate.bottom:
                    entry = min(c[i], candidate.top)
                    swept_low = s["sweep"][2]
                    stop = swept_low - _atr_pad(df, i)
                    risk = entry - stop
                    if risk > 0:
                        target = entry + rr * risk
                        open_trade = Trade(ts, None, "long", entry, stop, target,
                                           killzone=kz)
                        open_trade_bars = 0
                        state[kz] = _empty_kz_state()
                elif s["dir"] == "short" and h[i] >= candidate.bottom and l[i] <= candidate.top:
                    entry = max(c[i], candidate.bottom)
                    swept_high = s["sweep"][2]
                    stop = swept_high + _atr_pad(df, i)
                    risk = stop - entry
                    if risk > 0:
                        target = entry - rr * risk
                        open_trade = Trade(ts, None, "short", entry, stop, target,
                                           killzone=kz)
                        open_trade_bars = 0
                        state[kz] = _empty_kz_state()

    return trades


def _empty_kz_state() -> dict:
    return {"sweep": None, "mss_idx": None, "dir": None}


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
