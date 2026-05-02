"""Core ICT primitives: swings, FVG, BOS/MSS, liquidity sweeps."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ----- Swing points (fractal) -----
def swing_points(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Mark n-bar fractal swing highs/lows. To avoid look-ahead bias the
    swing is recorded at the bar where it becomes confirmed (i.e. index
    i + n), not at the swing bar itself. The actual price level and
    timestamp of the swing are stored alongside."""
    h = df["high"].values
    l = df["low"].values
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    sh_price = np.full(len(df), np.nan)
    sl_price = np.full(len(df), np.nan)
    for i in range(n, len(df) - n):
        if h[i] == max(h[i - n : i + n + 1]) and h[i] > max(h[i - n : i]):
            sh[i + n] = True
            sh_price[i + n] = h[i]
        if l[i] == min(l[i - n : i + n + 1]) and l[i] < min(l[i - n : i]):
            sl[i + n] = True
            sl_price[i + n] = l[i]
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = sh_price
    out["swing_low_price"]  = sl_price
    return out


# ----- Fair Value Gap (3-candle imbalance) -----
@dataclass
class FVG:
    idx: int           # index of candle 3 (the one creating the gap with candle 1)
    time: pd.Timestamp
    direction: str     # 'bull' or 'bear'
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


# ----- Market Structure Shift / Break of Structure -----
def detect_mss(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Mark bars where price breaks the most recent confirmed swing high/low,
    indicating a market-structure shift (CHoCH) — used for confirmation after
    a liquidity sweep."""
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
            sh_levels.append((i, sw["swing_high_price"].iat[i]))
        if sw["swing_low"].iat[i]:
            sl_levels.append((i, sw["swing_low_price"].iat[i]))
        # Trim window
        sh_levels = [(j, v) for j, v in sh_levels if i - j <= lookback]
        sl_levels = [(j, v) for j, v in sl_levels if i - j <= lookback]
        # Up MSS: close above most recent swing high
        if sh_levels:
            top = max(v for _, v in sh_levels[-5:])  # last 5 swing highs
            if c[i] > top and i > last_break_idx_up:
                mss_up[i] = True
                last_break_idx_up = i
        if sl_levels:
            bot = min(v for _, v in sl_levels[-5:])
            if c[i] < bot and i > last_break_idx_dn:
                mss_dn[i] = True
                last_break_idx_dn = i
    out = df.copy()
    out["mss_up"] = mss_up
    out["mss_dn"] = mss_dn
    return out


# ----- Liquidity sweep (taking out a recent swing high/low) -----
def liquidity_sweep(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """A bar sweeps liquidity if its high exceeds the highest high of the
    prior `window` bars but closes back below it (sell-side sweep above),
    or its low pierces the lowest low and closes back above (sweep below)."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    swept_high = np.zeros(len(df), dtype=bool)
    swept_low = np.zeros(len(df), dtype=bool)
    for i in range(window, len(df)):
        prior_high = h[i - window : i].max()
        prior_low = l[i - window : i].min()
        if h[i] > prior_high and c[i] < prior_high:
            swept_high[i] = True
        if l[i] < prior_low and c[i] > prior_low:
            swept_low[i] = True
    out = df.copy()
    out["sweep_high"] = swept_high
    out["sweep_low"] = swept_low
    return out


# ----- Higher-timeframe bias -----
def htf_bias(htf: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50) -> pd.Series:
    """Simple bias: EMA cross on HTF closes. +1 bull, -1 bear, 0 flat."""
    f = htf["close"].ewm(span=ema_fast, adjust=False).mean()
    s = htf["close"].ewm(span=ema_slow, adjust=False).mean()
    bias = pd.Series(0, index=htf.index, dtype=int)
    bias[f > s] = 1
    bias[f < s] = -1
    return bias


def align_bias_to_ltf(bias_htf: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    return bias_htf.reindex(ltf_index, method="ffill").fillna(0).astype(int)
