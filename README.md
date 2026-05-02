# ICT Silver Bullet — NAS 5m Backtest

Mechanical implementation of Michael Huddleston's **Silver Bullet** model
on the Nasdaq 5-minute data shipped in `5m_data.csv` (May 2020 – Oct 2025).

## Setup the model encodes

For each Silver Bullet killzone (NY local time):

| Killzone | Window |
| --- | --- |
| London open | 03:00 – 04:00 |
| AM          | 10:00 – 11:00 |
| PM          | 14:00 – 15:00 |

1. **HTF bias** — 1H EMA(20) vs EMA(50) cross used as a mechanical proxy for
   premium/discount + draw-on-liquidity.
2. **Liquidity sweep** — inside the killzone, a 5m bar must take out the
   highest-high or lowest-low of the prior 2h and close back inside.
3. **Market structure shift (MSS / CHoCH)** — close beyond the most recent
   confirmed swing high/low, in the bias direction. Swings use a 2-bar
   fractal and are only treated as known **after** the confirmation bar
   (no look-ahead).
4. **FVG entry** — find the most recent 3-candle FVG inside the displacement,
   limit-enter on retrace into the FVG.
5. **Risk** — stop beyond the swept extreme (+ 0.1×ATR pad), fixed RR = 2.
   Trade expires after 3h if neither stop nor target is hit.
6. **Costs** — 1 NAS point of spread+slippage per side is deducted from the
   R-multiple of every trade.

## Results

Starting equity: **$10 000** | Risk per trade: **$100 (1R)** | RR: **2.0**

| Metric | Value |
| --- | --- |
| Trades            | 77 |
| Win rate          | 58.4% |
| Expectancy        | 0.33 R |
| Total R           | 25.5 |
| Profit factor     | 2.10 |
| Net PnL           | $2 553 |
| Max drawdown      | $-1 015 (-7.93%) |
| Final equity      | $12 553 |

By killzone:

| Killzone | Count | Mean R | Sum R |
| --- | --- | --- | --- |
| London | 41 | 0.23 | 9.4 |
| AM     | 22 | 0.24 | 5.4 |
| PM     | 14 | 0.77 | 10.7 |

Equity curve / drawdown / R distribution: `results/equity_curve.png`
Per-killzone breakdown: `results/killzone_breakdown.png`
Per-trade ledger: `results/trades.csv`

![Equity curve](results/equity_curve.png)
![Killzone breakdown](results/killzone_breakdown.png)

## Honest caveats

- **77 trades over 5+ years is a small sample.** The 95% confidence interval
  on win rate is roughly ±11 percentage points. Treat the absolute numbers
  as an indicative edge, not a forecast.
- **PM killzone carries the model.** With only 14 trades it is the most
  fragile — could be variance, could be real (PM is quieter so MSS quality
  is higher). Walk-forward testing recommended before sizing it up.
- **2024-2025 has been flat-to-down.** Whether this is regime change or
  random variance can't be told from one sample.
- **Costs are simplified.** 1 point per side is a reasonable estimate for
  retail NAS CFD; futures (NQ) would be cheaper, retail spot CFDs from
  bucket-shop brokers can be much worse.
- **Broker timezone assumed EET/EEST** (typical MT5 server). If your data
  source is something else, edit `BROKER_TZ` in `silver_bullet/data_loader.py`.
- **Bias proxy is an EMA cross.** ICT's actual bias methodology is
  discretionary (HTF PD arrays, draw on liquidity, weekly profile). A real
  upgrade is to encode draw-on-liquidity from prior-day/week high-low and
  unmitigated HTF FVGs.

## Run it

```bash
pip install pandas numpy matplotlib pytz
python3 run_backtest.py
```

Output lands in `results/`.

## Files

```
silver_bullet/
  data_loader.py       # CSV load + broker→NY tz convert + resample
  ict_primitives.py    # swings, FVG, MSS, liquidity sweep, HTF bias
  strategy.py          # Silver Bullet state machine + trade execution
  backtest.py          # stats + equity / drawdown / R-dist plots
run_backtest.py        # entry point
```
