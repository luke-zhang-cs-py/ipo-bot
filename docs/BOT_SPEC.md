# IPO bot: trading specification and runbook

What the bot does when it trades, every limit that can stop it, and what a person does when one fires.
It describes the code in this repository; it makes no claim of profit, and nothing here is advice.

## The loop

| Step | What happens | Where |
|---|---|---|
| Read | Filings, financials and prices from the data tools, each dated; in a backtest nothing after the cutoff | `src/tools.py` |
| Decide | A rating with bull, base and bear cases; a buy only on Overweight, which needs +15% expected return and medium or high conviction | `src/verify.py` (thresholds) |
| Size | The smallest of every limit in the portfolio file (below) | `src/portfolio.py` |
| Execute | A limit order 0.5% over the planned entry; partial fills worked, each remainder re-sized; the fill confirmed, never assumed | `src/execution.py` |
| Monitor | Kill switch, API failures, stale quotes, abnormal spreads, price deviation; every step logged | `src/monitor.py`, `src/paper.py` |

```
for each signal (symbol, entry, stop, conviction):
    if monitor.gate(signal) is not None:        log "stopped by the monitor: <reason>"; continue
    plan = size_position(portfolio, signal)       # every limit computed; the smallest wins
    if plan.shares == 0:                          log "blocked: <rule>"; continue
    repeat up to 3 orders:
        fill = broker.submit(symbol, shares, limit = entry * 1.005)
        on API error:                             log it; monitor counts it; stop
        record order id, filled shares, average price, fee
        if filled in full or nothing traded:      stop
        shares = min(plan left, size_position(portfolio after the fill), risk budget left / (limit - stop))
    place a stop for the shares actually filled; log the whole result
```

## Risk limits (the portfolio file's `rules`; the guardrails are off unless set)

| Rule | Example | Effect |
|---|---|---|
| `risk_per_trade_pct` | 1 | Loss at the stop-working price at most 1% of the portfolio (x0.5 / x1 / x1.5 by conviction) |
| `max_position_pct` / `max_sector_pct` | 10 / 30 | No stock or sector above that share |
| `min_cash_pct` | 5 | Cash kept |
| `max_order_value` | 500 | No single order worth more |
| `max_gross_exposure_pct` | 90 | No new buys once that much is invested |
| `max_daily_loss_pct` | 2 | No new buys once today's loss (`day_pnl`) reaches it |
| `max_drawdown_pct` | 8 | No new buys once the portfolio is that far below `equity_high`: review before resuming |
| `earnings_blackout_hours` | 48 | No buys that close before the next earnings release |
| `max_volume_pct` | 1 | No order above that share of average daily volume |
| `"halted": true` | | The kill switch: nothing is bought until it is set back |

A guardrail that is set but can't be checked (no earnings date, no volume, no `equity_high`) is named in
the result's `warnings`, never silently passed.

## Monitor (each stop names itself in the log)

| Check | Default | Action |
|---|---|---|
| Kill file `forecasts/paper/KILL` | present | Stop all orders |
| API failures in a row | 3 | Pause until reset |
| Quote age | over 60 s | Skip the signal: stale data |
| Spread | over 0.5% of the mid | Skip: abnormal spread |
| Ask against the signal's price | over 0.5% away | Skip: price deviation |

## Backtest protocol

1. Rules written in code before any test (`evaluation/strategies.py`, `evaluation/ipo_eval.py`), not tuned on results.
2. Periods that include a rally, a drawdown and a range; results reported by regime, labelled from SPY's past only.
3. A signal read from a close fills at the next open; the IPO model's features are audited for look-ahead (dates,
   a perturbation test, and a planted leak it must catch).
4. Fees, slippage, minimum order size and partial fills simulated on every fill (0.10% + 0.05% a side by default).
5. Chronological splits: walk-forward by year; a holdout scored once and recorded (`bench_runs/*_log.jsonl`).
6. Reported: net return, CAGR, maximum drawdown, trades, win rate, average win and loss, profit factor,
   expectancy, time invested.
7. Re-run at 1.5x and 2x costs, with fills a day late, with each parameter moved 20%, and on each half of the
   history; a result that changes sign against holding the market under any of these is called fragile.

## API keys

- Paper trading only: `src/paper.py` refuses any Alpaca endpoint but `paper-api.alpaca.markets`.
- Keys live in `.env` (git-ignored), never in code. Use keys that can trade but not withdraw or transfer; restrict
  them by IP where the broker allows; rotate them after any suspected exposure.
- Review the paper log daily during the first weeks; promote nothing to real money until the live log matches
  the backtest's assumptions (fill rate, slippage, blocks).

## The log (`forecasts/paper/log.jsonl`, one line per signal)

| Field | Content |
|---|---|
| `logged` | UTC time of the line |
| `signal` | symbol, entry, stop, conviction, sector, earnings date, volume, quote |
| `result.status` | filled, partial, unfilled, blocked, stopped by the monitor, rejected by the rules, api error |
| `result.blocked_by` | the rule or monitor check that stopped it |
| `result.orders[]` | order id, shares asked, limit, shares filled, average price, fee, status |
| `result.filled`, `avg_price`, `fees`, `loss_at_stop`, `risk_budget`, `stop_order`, `warnings` | the outcome |
| `latency_s` | signal to result |

`python src/paper.py report` turns the log into the divergence report: outcomes by cause, the fill rate of
planned shares, slippage from the signal's price, fees and latency.

## Runbook

| Event | Response |
|---|---|
| Three API failures in a row | The bot pauses itself. Check the broker's status page and the keys; reset when fixed. |
| Daily loss limit reached | No new buys today. Read the day's log before the next session. |
| Drawdown limit reached | No new buys. Review every open position and the strategy's assumptions; set a new `equity_high` deliberately. |
| Stale data or abnormal spreads, repeatedly | The feed or the market is broken: halt with the kill file until it is understood. |
| Fill rate or slippage worse than the backtest assumed | The backtest's costs were too kind: re-run it at the measured costs before trading on. |
| Anything unexplained | `touch forecasts/paper/KILL` first, investigate second. |

## What was taken from other open-source trading frameworks

Each idea below is re-implemented here from the project's documented behaviour; no code is copied (freqtrade and
backtrader are GPL-3.0, which would bind this MIT project if their code were copied).

| Idea | From | Here |
|---|---|---|
| Look-ahead analysis: a rule's decisions on the full history against the history cut at each decision | freqtrade `lookahead-analysis` | `evaluation/bias_checks.py`, every bot and the replay signal |
| Recursive (warm-up) analysis: the same decision with less history loaded | freqtrade `recursive-analysis` | `evaluation/bias_checks.py` |
| Protections: a cooldown after selling, a pause after repeated stop-outs | freqtrade `CooldownPeriod`, `StoplossGuard` | `src/monitor.py` |
| Volume-share slippage: at most 2.5% of a bar's volume fills, impact 0.1 x (share of volume)^2 | zipline `VolumeShareSlippage` (its defaults) | `evaluation/strategies.py`, `execution.BarBroker` |
| A per-share commission with a minimum | zipline `PerShare`; common US broker schedules | `evaluation/strategies.py` |
| Split-adjusted prices with dividends paid as cash | QuantConnect LEAN `SplitAdjusted` / `Raw` normalisation | `evaluation/strategies.py` (`mode="split_adjusted"`) |
| A resting limit fills only when the price trades through it, not on a touch | NautilusTrader fill models; backtrader bar fills | `execution.BarBroker` |

Not taken: leverage, shorting, market making, cross-venue arbitrage, and machine-learned exits; none fits a
long-only research bot, and each adds failure modes the tests above don't cover.
