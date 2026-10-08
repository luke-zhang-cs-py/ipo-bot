# Tracking: the paper bot against the market, 2026-01-02 to 2026-10-07

Updated 2026-10-08 09:13 UTC by `python evaluation/tracker.py`; every run is a line in `track.json`. The bot is the paper replay (a breakout signal on 30 large caps, every guardrail on, fills from daily bars); this measures how its account moves with the market, not whether it has an edge.

| | Account | SPY |
|---|---|---|
| Return | +10.71% | +13.77% |
| Worst drawdown | -3.75% | -9.13% |

Over 192 trading days: daily correlation with SPY 0.500 (95% CI 0.373 to 0.612), beta 0.339 (95% CI 0.247 to 0.423), tracking error +11.56% a year, average daily return against SPY's -4.05% a year (95% CI -30.61% to +22.90%). Intervals resample the days 1,000 times; an interval that spans zero is no evidence either way.

## Trend accordance

Invested on days SPY was above its 200-day average: 75.6% on average (179 days); below it: 74.6% (13 days). By regime: bull: 75.8% (158 d) | ordinary: 73.9% (34 d).

## Execution

86 signals: filled 33 (38.4%), monitor: price deviation 33 (38.4%), blocked: exposure 13 (15.1%), monitor: stoploss guard 6 (7.0%), monitor: cooldown 1 (1.2%). Fill rate of planned shares 100.0%; mean slippage from the signal's price -0.023%; 18 stops hit.

Look-ahead check on the signal: 0 found in 30 sampled days.
