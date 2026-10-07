# Tracking: the paper bot against the market, 2026-01-02 to 2026-10-07

Updated 2026-10-07 22:24 UTC by `python evaluation/tracker.py`; every run is a line in `track.json`. The bot is the paper replay (a breakout signal on 30 large caps, every guardrail on, fills from daily bars); this measures how its account moves with the market, not whether it has an edge.

| | Account | SPY |
|---|---|---|
| Return | +10.7% | +13.8% |
| Worst drawdown | -3.7% | -9.1% |

Daily correlation with SPY 0.50, beta 0.34, tracking error +11.6% a year over 192 trading days.

## Trend accordance

Invested on days SPY was above its 200-day average: +76% (179 days); below it: +75% (13 days). By regime: bull: +76% (158 d) | ordinary: +74% (34 d).

## Execution

86 signals: filled 33, monitor: price deviation 33, blocked: exposure 13, monitor: stoploss guard 6, monitor: cooldown 1. Fill rate of planned shares +100%; mean slippage from the signal's price -0.02%; 18 stops hit.

Look-ahead check on the signal: 0 found in 30 sampled days.
