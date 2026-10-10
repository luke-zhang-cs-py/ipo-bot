# Tracking: the paper bot against the market, 2026-01-02 to 2026-10-09

Updated 2026-10-10 01:50 UTC by `python evaluation/tracker.py`; every run is a line in `track.json`. The bot is the paper replay (a breakout signal on 30 large caps, every guardrail on, fills from daily bars); this measures how its account moves with the market, not whether it has an edge.

| | Account | SPY |
|---|---|---|
| Return | +11.79% | +13.96% |
| Worst drawdown | -3.37% | -9.13% |

Over 194 trading days: daily correlation with SPY 0.483 (95% CI 0.358 to 0.596), beta 0.319 (95% CI 0.235 to 0.404), tracking error +11.64% a year, average daily return against SPY's -2.99% a year (95% CI -30.04% to +23.13%). Intervals resample the days 1,000 times; an interval that spans zero is no evidence either way.

## Trend accordance

Invested on days SPY was above its 200-day average: 75.8% on average (181 days); below it: 75.0% (13 days). By regime: bull: 76.1% (160 d) | ordinary: 74.3% (34 d).

## Execution

86 signals: filled 33 (38.4%), monitor: price deviation 33 (38.4%), blocked: exposure 13 (15.1%), monitor: stoploss guard 6 (7.0%), monitor: cooldown 1 (1.2%). Fill rate of planned shares 100.0%; mean slippage from the signal's price -0.023%; 18 stops hit.

Look-ahead check on the signal: 0 found in 30 sampled days.
