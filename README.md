# ICT 2022 Model — AM Session Backtest (NAS 1m)

Fully rule-based implementation of the ICT **2022 Model** on Nasdaq 1-minute
data (`1m_data.csv`, 2016–2025), filtered to the **AM session only**
(08:00–12:00 ET).

---

## Cascade (all gates required, strictly sequential)

```
4H EMA Bias (bull / bear)
  → Liquidity Sweep on 15m  (wick beyond recent swing H/L, close back inside)
  → MSS / CHoCH on 1m       (structural break confirming reversal)
  → FVG entry on 1m         (retrace into a Fair Value Gap after the MSS)
```

**One sweep → one trade.**  Once a sweep produces an entry the sweep is
consumed and cannot trigger additional setups.

---

## Entry · Stop · Target

| | Rule |
|---|---|
| **Entry** | FVG inner edge touched on the bar *after* FVG formation |
| **Stop**  | Below nearest 1m swing low (long) / above nearest 1m swing high (short); fallback: sweep level |
| **Target** | Entry ± 2R (fixed RR) |
| **Expiry** | Closed at market after 2h (120 bars) if neither stop nor target hit |

---

## Per-trade charts (5 files per trade)

Each subfolder under `big_test_only_am/trade_charts_YYYY/` contains:

| File | Content |
|---|---|
| `1d_bias.png`     | **Daily** candles + zigzag HH/HL/LH/LL (daily bias) |
| `4h_bias.png`     | **4H** candles + zigzag HH/HL/LH/LL (4H trend context) |
| `15m_context.png` | 15m candles — sweep + MSS context |
| `5m_context.png`  | 5m candles — intermediate structure |
| `1m_entry.png`    | 1m candles — FVG, entry, SL, TP |

HTF charts (Daily, 4H) show zigzag structure labels only — execution charts
(15m, 5m, 1m) are kept clean.

---

## Config

| Parameter | Value |
|---|---|
| 4H EMA | 20 / 50 |
| 15m sweep window | 20 bars |
| Sweep max age | 4h |
| MSS timeframe | 1m |
| Arm window | 90 min |
| FVG min size | 0.2 × ATR(14) |
| Max trades per sweep | **1** |
| Killzone | AM only (08:00–12:00 ET) |

---

## Run

```bash
pip install pandas numpy matplotlib pytz
python3 run_big_test_only_am.py
```

Results → `big_test_only_am/`

---

## Layout

```
model_2022/
  ict_primitives.py   swings, FVG, MSS, sweep, HTF bias, OB (self-contained)
  strategy.py         2022 model cascade state machine
  backtest.py         stats, plots, per-trade charts
  data_loader.py      CSV load + resample

run_big_test_only_am.py

big_test_only_am/
  trades.csv · stats.json · *.png
  trade_charts_2020/  (5 charts per trade, regenerated each run)

1m_data.csv
```
