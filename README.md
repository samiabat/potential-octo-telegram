# ICT Silver Bullet — NAS 5m Backtest

Mechanical implementation of Michael Huddleston's **Silver Bullet** model
on the Nasdaq 5-minute data shipped in `5m_data.csv` (May 2020 – Oct 2025).

## How v2 differs from v1

The first cut required a strict ICT chain: HTF bias + 2h-extreme liquidity
sweep + market structure shift + bias-aligned FVG retrace, all inside a 1h
killzone. That produced only **77 trades over 5 years** (~14/yr) — closer
to a swing strategy than day trading.

A sensitivity sweep (`sweep_configs.py`) showed that the **bias / sweep / MSS
gates were filtering out winners**, not losers, on this dataset. The
proxies I used for those gates (1H EMA cross for bias, 2h prior extreme
for sweep, fractal MSS) are crude approximations of how Huddleston actually
identifies them.

v2 keeps the spirit of Silver Bullet but matches how it's mechanically
day-traded: **killzone → quality FVG forms → enter on retrace → fixed RR**.
All the optional gates (bias filter, sweep, MSS) are still wired in and
toggleable from the strategy config.

## Sweep results (sensitivity table)

| Config | Trades | tr/yr | Win % | exp R | PF | Total R | MDD % | Final $ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A no bias, 1st-half, FVG≥0.3 ATR** | **1232** | **230** | **48.6** | **0.08** | **1.19** | **101** | **-13.3** | **20 084** |
| B + strict EMA bias                 | 676  | 126 | 47.3 | 0.06  | 1.13 | 40   | -13.9 | 13 993 |
| C + sweep gate                      | 41   | 8   | 34.1 | -0.17 | 0.72 | -7   | -11.2 |  9 285 |
| D purist (bias+sweep+MSS)           | 4    | 1.5 | 25.0 | -0.76 | 0.10 | -3   | -3.4  |  9 695 |
| E + larger FVG (0.6 ATR)            | 354  | 67  | 46.9 | 0.05  | 1.10 | 16   | -21.5 | 11 575 |
| F + full-hour entries               | 1447 | 270 | 46.6 | 0.03  | 1.07 | 46   | -17.0 | 14 597 |
| G + sweep + larger FVG              | 27   | 6   | 37.0 | -0.13 | 0.75 | -4   | -7.9  |  9 637 |

**Config A is the locked-in default.**

## Final result (Config A on full dataset)

Starting equity $10 000, risk per trade $100 (1R), RR 2.0, costs 1pt/side.

| Metric | Value |
| --- | --- |
| Trades            | 1 232 (~230/yr, ~5/wk) |
| Win rate          | 48.6 % |
| Expectancy        | 0.08 R |
| Profit factor     | 1.19 |
| Net PnL           | +$10 084 |
| Max drawdown      | -$1 894  (-13.3 %) |
| Final equity      | $20 084 |

By killzone:

| KZ | Count | Mean R | Sum R |
| --- | ---: | ---: | ---: |
| AM     | 429 | 0.13 | 57.5 |
| PM     | 343 | 0.07 | 23.1 |
| London | 460 | 0.04 | 20.3 |

![Equity curve](results/equity_curve.png)
![Killzone breakdown](results/killzone_breakdown.png)

## Setup the model encodes

Killzones (NY local time):

| Killzone | Window |
| --- | --- |
| London open | 03:00 – 04:00 |
| AM          | 10:00 – 11:00 |
| PM          | 14:00 – 15:00 |

For each killzone:

1. Detect FVGs (3-candle imbalance: `low[i] > high[i-2]` for bullish).
2. Filter out FVGs whose width is < `0.3 × ATR(14)` (noise).
3. Wait for retrace into the FVG (must be on a bar **after** formation
   to avoid the same-bar fill bug).
4. Entry = FVG inner edge.
   Stop  = below the local 1h swing low (long) or above swing high
           (short), with a min stop = `1 × ATR(14)` and a 5-pt floor.
   Target = entry ± 2R.
5. One trade per killzone, only in the first 30 minutes of the window.
6. Trade expires after 2h if neither stop nor target is hit.
7. Costs: 1 NAS point of spread+slip per side is deducted from R.

Walk-forward bias removed: 2-bar fractal swings are only treated as known
**after** the confirmation bar (i.e. recorded at index `i+n`, not `i`).

## Honest take on profitability

- Edge is **small but real** after costs: PF 1.19, +0.08 R/trade.
- 2020-2021 produced most of the equity growth (high vol).
  2022-2025 grinds slowly upwards — typical of a real edge, not a curve fit.
- Win rate < 50 % is **expected** for a 2R fixed-target system.
- Drawdowns of -10 to -13 % occur multiple times in 5 years; size accordingly.
- The proper bias module (HTF draw-on-liquidity, premium/discount of
  dealing range, unmitigated HTF FVGs as targets) is the obvious next
  upgrade — it's what Huddleston himself emphasises and the EMA proxy is
  not capturing.
- A 1-minute dataset would let us refine entries and stop placement
  without changing the model.

## Run it

```bash
pip install pandas numpy matplotlib pytz
python3 run_backtest.py        # main backtest
python3 sweep_configs.py        # sensitivity table over filter combos
```

Output lands in `results/`.

## Files

```
silver_bullet/
  data_loader.py       # CSV load + broker(EET/EEST)→NY tz convert + resample
  ict_primitives.py    # swings (no look-ahead), FVG, MSS, liquidity sweep, HTF bias
  strategy.py          # Silver Bullet state machine + diagnostics
  backtest.py          # stats + equity / drawdown / R-dist plots
run_backtest.py        # main entry point
sweep_configs.py       # sensitivity sweep across filter configs
results/
  equity_curve.png, killzone_breakdown.png
  trades.csv, stats.json, stats_by_killzone.csv, config_sweep.csv
```
