"""ICT primitives for the 2022 Model (self-contained, no external dependencies).

Includes:
  - Swing point detection (fractal, no look-ahead)
  - Fair Value Gap (FVG) detection
  - Market Structure Shift (MSS / CHoCH) detection
  - Liquidity sweep detection
  - Higher-timeframe bias (EMA cross)
  - Order Block (OB) detection on 1m
  - Helpers shared with the strategy module
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Swing points (fractal, no look-ahead)
# ---------------------------------------------------------------------------

def swing_points(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Mark n-bar fractal swing highs/lows.

    To avoid look-ahead bias the swing is recorded at bar i+n (the
    confirmation bar), not at the swing bar itself.  The actual price
    level is stored alongside.
    """
    h = df["high"].values
    l = df["low"].values
    sh       = np.zeros(len(df), dtype=bool)
    sl       = np.zeros(len(df), dtype=bool)
    sh_price = np.full(len(df), np.nan)
    sl_price = np.full(len(df), np.nan)
    for i in range(n, len(df) - n):
        if h[i] == max(h[i - n: i + n + 1]) and h[i] > max(h[i - n: i]):
            sh[i + n]       = True
            sh_price[i + n] = h[i]
        if l[i] == min(l[i - n: i + n + 1]) and l[i] < min(l[i - n: i]):
            sl[i + n]       = True
            sl_price[i + n] = l[i]
    out               = df.copy()
    out["swing_high"]       = sh
    out["swing_low"]        = sl
    out["swing_high_price"] = sh_price
    out["swing_low_price"]  = sl_price
    return out


# ---------------------------------------------------------------------------
# Fair Value Gap (3-candle imbalance)
# ---------------------------------------------------------------------------

@dataclass
class FVG:
    idx: int            # index of candle 3 (the bar that creates the gap with candle 1)
    time: pd.Timestamp
    direction: str      # 'bull' | 'bear'
    top: float
    bottom: float
    mitigated: bool = False
    mitigated_idx: int | None = None


def detect_fvgs(df: pd.DataFrame) -> list[FVG]:
    """A bullish FVG exists when low[i] > high[i-2] (gap between c1 and c3).
    A bearish FVG exists when high[i] < low[i-2]."""
    h = df["high"].values
    l = df["low"].values
    fvgs: list[FVG] = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            fvgs.append(FVG(i, df.index[i], "bull", l[i], h[i - 2]))
        elif h[i] < l[i - 2]:
            fvgs.append(FVG(i, df.index[i], "bear", l[i - 2], h[i]))
    return fvgs


# ---------------------------------------------------------------------------
# Market Structure Shift (CHoCH / BOS)
# ---------------------------------------------------------------------------

def detect_mss(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Mark bars where price breaks the most recent confirmed swing high/low,
    indicating a market-structure shift (CHoCH) used for confirmation after a
    liquidity sweep."""
    sw = swing_points(df, n=2)
    sh_levels: list[tuple[int, float]] = []
    sl_levels: list[tuple[int, float]] = []
    mss_up = np.zeros(len(df), dtype=bool)
    mss_dn = np.zeros(len(df), dtype=bool)
    last_break_idx_up = -1
    last_break_idx_dn = -1
    h = df["high"].values
    c = df["close"].values
    l = df["low"].values
    for i in range(len(df)):
        if sw["swing_high"].iat[i]:
            sh_levels.append((i, float(sw["swing_high_price"].iat[i])))
        if sw["swing_low"].iat[i]:
            sl_levels.append((i, float(sw["swing_low_price"].iat[i])))
        # Trim to lookback window
        sh_levels = [(j, v) for j, v in sh_levels if i - j <= lookback]
        sl_levels = [(j, v) for j, v in sl_levels if i - j <= lookback]
        # Up MSS: close above most recent swing high (use last 5 swing highs)
        if sh_levels:
            top = max(v for _, v in sh_levels[-5:])
            if c[i] > top and i > last_break_idx_up:
                mss_up[i]         = True
                last_break_idx_up = i
        # Down MSS: close below most recent swing low
        if sl_levels:
            bot = min(v for _, v in sl_levels[-5:])
            if c[i] < bot and i > last_break_idx_dn:
                mss_dn[i]         = True
                last_break_idx_dn = i
    out          = df.copy()
    out["mss_up"] = mss_up
    out["mss_dn"] = mss_dn
    return out


# ---------------------------------------------------------------------------
# Liquidity sweep
# ---------------------------------------------------------------------------

def liquidity_sweep(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Detect bars where price wicks beyond a recent swing extreme and closes
    back inside (liquidity grab / stop hunt).

    Extra columns produced:
      sweep_high / sweep_high_level — bar swept a prior-window high; level = that high
      sweep_low  / sweep_low_level  — bar swept a prior-window low;  level = that low
    """
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    swept_high       = np.zeros(n, dtype=bool)
    swept_low        = np.zeros(n, dtype=bool)
    swept_high_level = np.full(n, np.nan)
    swept_low_level  = np.full(n, np.nan)
    for i in range(window, n):
        prior_high = h[i - window: i].max()
        prior_low  = l[i - window: i].min()
        if h[i] > prior_high and c[i] < prior_high:
            swept_high[i]       = True
            swept_high_level[i] = prior_high
        if l[i] < prior_low and c[i] > prior_low:
            swept_low[i]       = True
            swept_low_level[i] = prior_low
    out                    = df.copy()
    out["sweep_high"]       = swept_high
    out["sweep_low"]        = swept_low
    out["sweep_high_level"] = swept_high_level
    out["sweep_low_level"]  = swept_low_level
    return out


# ---------------------------------------------------------------------------
# Higher-timeframe bias
# ---------------------------------------------------------------------------

def htf_bias(htf: pd.DataFrame,
             ema_fast: int = 20,
             ema_slow: int = 50) -> pd.Series:
    """Simple EMA-cross bias on HTF closes.  +1 bull, -1 bear, 0 flat."""
    f = htf["close"].ewm(span=ema_fast, adjust=False).mean()
    s = htf["close"].ewm(span=ema_slow, adjust=False).mean()
    bias = pd.Series(0, index=htf.index, dtype=int)
    bias[f > s] = 1
    bias[f < s] = -1
    return bias


def align_bias_to_ltf(bias_htf: pd.Series,
                      ltf_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill HTF bias onto a finer (LTF) index."""
    return bias_htf.reindex(ltf_index, method="ffill").fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Order Block
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    """A single Order Block identified on the entry timeframe (1m).

    The OB is the *last opposite-colour candle* immediately before the
    impulse move that created the linked FVG.  It is recorded at the same
    bar index as the FVG (candle-3 of the FVG sequence) — no look-ahead.

    For a bullish OB (last bearish candle before an up-impulse):
        entry zone  = [ob_low .. ob_high]
        stop below  = ob_low - pad
    For a bearish OB (last bullish candle before a down-impulse):
        entry zone  = [ob_low .. ob_high]
        stop above  = ob_high + pad
    """
    idx: int               # bar index where the OB becomes known (= FVG.idx)
    time: pd.Timestamp
    direction: str         # 'bull' | 'bear'
    ob_high: float
    ob_low: float
    ob_bar_idx: int        # actual index of the OB candle (before the impulse)


def detect_order_blocks(df: pd.DataFrame) -> list[OrderBlock]:
    """Detect Order Blocks by linking each FVG to its preceding opposite candle.

    Algorithm (no look-ahead — all information is available by bar i):
      For each bullish FVG at bar i (low[i] > high[i-2]):
        Search backward from i-2 to find the last candle whose close < open
        (a bearish candle).  That candle is the bullish Order Block.
      For each bearish FVG at bar i (high[i] < low[i-2]):
        Search backward from i-2 for the last bullish candle (close > open).
    """
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    fvgs = detect_fvgs(df)
    obs: list[OrderBlock] = []

    for fvg in fvgs:
        i = fvg.idx
        # Search from the first FVG candle (i-2) backward
        search_start = i - 2
        if search_start < 0:
            continue

        if fvg.direction == "bull":
            # Find last bearish candle (close < open) at or before search_start
            ob_idx = None
            for j in range(search_start, max(search_start - 20, -1), -1):
                if closes[j] < opens[j]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                obs.append(OrderBlock(
                    idx=i,
                    time=df.index[i],
                    direction="bull",
                    ob_high=highs[ob_idx],
                    ob_low=lows[ob_idx],
                    ob_bar_idx=ob_idx,
                ))
        else:  # bear FVG
            # Find last bullish candle (close > open) at or before search_start
            ob_idx = None
            for j in range(search_start, max(search_start - 20, -1), -1):
                if closes[j] > opens[j]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                obs.append(OrderBlock(
                    idx=i,
                    time=df.index[i],
                    direction="bear",
                    ob_high=highs[ob_idx],
                    ob_low=lows[ob_idx],
                    ob_bar_idx=ob_idx,
                ))

    return obs


# ---------------------------------------------------------------------------
# Event-list helpers (for efficient in-order lookups during the walk)
# ---------------------------------------------------------------------------

def build_event_list(series: pd.Series) -> list[pd.Timestamp]:
    """Return a sorted list of timestamps where *series* is True."""
    return sorted(series.index[series.astype(bool)].tolist())


def build_level_event_list(
    series_bool: pd.Series,
    series_level: pd.Series,
) -> list[tuple[pd.Timestamp, float]]:
    """Return a sorted list of (timestamp, price_level) for True entries.

    Used to associate a swept price level with each sweep/MSS event so
    that the strategy can read the exact price level without look-ahead.
    """
    idx = series_bool.index[series_bool.astype(bool)]
    events = [(ts, float(series_level.loc[ts])) for ts in idx
              if not np.isnan(series_level.loc[ts])]
    return sorted(events, key=lambda x: x[0])


def most_recent_event_before(
    events: list[pd.Timestamp],
    ts: pd.Timestamp,
    max_age: pd.Timedelta,
) -> Optional[pd.Timestamp]:
    """Return the most-recent event timestamp strictly before *ts* and within
    *max_age*, or None."""
    # bisect_left gives the insertion point for ts; idx-1 is the last event < ts
    idx = bisect.bisect_left(events, ts)
    if idx == 0:
        return None
    candidate = events[idx - 1]
    if ts - candidate <= max_age:
        return candidate
    return None


def most_recent_level_event_before(
    events: list[tuple[pd.Timestamp, float]],
    ts: pd.Timestamp,
    max_age: pd.Timedelta,
) -> Optional[tuple[pd.Timestamp, float]]:
    """Like most_recent_event_before but for (timestamp, level) tuples.

    Returns (timestamp, price_level) of the most-recent event strictly
    before *ts* and within *max_age*, or None.
    """
    timestamps = [e[0] for e in events]
    idx = bisect.bisect_left(timestamps, ts)
    if idx == 0:
        return None
    candidate_ts, candidate_level = events[idx - 1]
    if ts - candidate_ts <= max_age:
        return candidate_ts, candidate_level
    return None


def first_event_between(
    events: list[pd.Timestamp],
    after: pd.Timestamp,
    before: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """Return the first event strictly after *after* and strictly before *before*."""
    idx = bisect.bisect_right(events, after)
    if idx >= len(events):
        return None
    candidate = events[idx]
    if candidate < before:
        return candidate
    return None
