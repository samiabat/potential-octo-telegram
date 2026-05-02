"""Load NAS 5m data and convert broker time (EET/EEST) to America/New_York."""
from __future__ import annotations
import pandas as pd

BROKER_TZ = "Europe/Helsinki"  # EET/EEST, matches typical MT5 broker server
NY_TZ = "America/New_York"


def load_5m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M:%S")
    df = df.sort_values("DateTime").drop_duplicates("DateTime").reset_index(drop=True)
    df = df.set_index("DateTime")
    # Localize broker time, convert to NY for ICT killzone alignment
    df.index = (
        df.index.tz_localize(BROKER_TZ, ambiguous="NaT", nonexistent="NaT")
        .tz_convert(NY_TZ)
    )
    df = df[~df.index.isna()]
    df = df[["Open", "High", "Low", "Close", "Volume", "TickVolume"]]
    df.columns = ["open", "high", "low", "close", "volume", "tick_volume"]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "tick_volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna()
    return out
