"""Load 1m NAS data and resample to any higher timeframe.

The 1m CSV uses the same tab-delimited broker-time (EET/EEST) format as
the 5m data: DateTime, Open, High, Low, Close, Volume, TickVolume.
"""
from __future__ import annotations
import pandas as pd

BROKER_TZ = "Europe/Helsinki"   # EET/EEST, typical MT5 broker time
NY_TZ = "America/New_York"


def load_1m(path: str) -> pd.DataFrame:
    """Load 1m CSV, convert broker time to NY time, sort ascending."""
    df = pd.read_csv(path, sep="\t")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M:%S")
    df = df.sort_values("DateTime").drop_duplicates("DateTime").reset_index(drop=True)
    df = df.set_index("DateTime")
    df.index = (
        df.index
        .tz_localize(BROKER_TZ, ambiguous="NaT", nonexistent="NaT")
        .tz_convert(NY_TZ)
    )
    df = df[~df.index.isna()]
    df = df[["Open", "High", "Low", "Close", "Volume", "TickVolume"]]
    df.columns = ["open", "high", "low", "close", "volume", "tick_volume"]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to *rule* (e.g. '15min', '4h', '1D').

    Bars are timestamped at the *right* edge — the moment the bar actually
    closes. A bar covering [T, T+rule) is labelled T+rule. This is what
    `align_bias_to_ltf` (ffill) needs to avoid look-ahead: at any LTF
    timestamp t, the most recent HTF bar with index ≤ t is one whose close
    is already known by t.
    """
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":      "sum",
        "tick_volume": "sum",
    }
    return df.resample(rule, label="right", closed="left").agg(agg).dropna()
