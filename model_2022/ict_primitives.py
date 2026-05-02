"""ICT primitives for the 2022 Model.

Extends the silver_bullet primitives with:
  - Order Block (OB) detection on 1m
  - HTF bias via EMA cross (same as silver_bullet, plus a structure variant)
  - Helpers shared with the strategy module
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Re-export shared primitives from silver_bullet so callers only import here.
from silver_bullet.ict_primitives import (  # noqa: F401
    swing_points,
    FVG,
    detect_fvgs,
    detect_mss,
    liquidity_sweep,
    htf_bias,
    align_bias_to_ltf,
)


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
