# IPO bot

An equity research assistant that assesses IPOs, listed stocks and the market, and answers with a
one-page memo: ratings, a 12-month horizon, bull/base/bear scenarios that add up to 100%, a
probability-weighted value, a stop-working price, dated catalysts, the top three risks, and sources
with dates. It runs on Claude Opus 5.5 with web search and its own data tools.

For your own use. It gives views on securities, not personal advice, and every answer ends with a
disclaimer. See "If you make it public" before anyone else uses it.

## Setup

```bash
pip install -r requirements.txt
```

Set these in your environment (see `.env.example`):

| Variable | What for | Where |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | console.anthropic.com (or `ant auth login`) |
| `SEC_USER_AGENT` | SEC EDGAR filings and reported financials, free | `"Your Name your@email.com"`: the SEC asks every client to identify itself |
| `FRED_API_KEY` | Rates, inflation, yields, credit spreads, VIX, free | fred.stlouisfed.org |
| `FMP_API_KEY` | Quotes, peers, key metrics, estimates, IPO calendar | financialmodelingprep.com (check the endpoint names in `tools.py` against your plan) |

Any key left unset turns that tool off: the bot then says which data is missing and does not guess.

## Use

```bash
python ipo_bot.py "Rate the <company> IPO"     # one question; memo printed and saved to memos/
python ipo_bot.py                               # interactive: follow-ups in one conversation
```

## Portfolio decisions

Put your holdings and your own sizing rules in `portfolio.json` (start from `portfolio.example.json`;
`portfolio.json` is git-ignored so your holdings stay on your machine). Then:

```bash
python ipo_bot.py --portfolio portfolio.json                         # review: buy/add/hold/trim/sell + up to 3 new ideas
python ipo_bot.py --portfolio portfolio.json "Should I buy <ticker>?"
```

The bot rates each stock as usual. It does not choose the size: `portfolio_size` works it out from your
rules and the smallest limit wins:

| Limit | Shares allowed |
|---|---|
| Risk | portfolio value x `risk_per_trade_pct` x conviction scale, divided by (entry - stop-working price) |
| Position | room left under `max_position_pct`, counting what you already hold |
| Sector | room left under `max_sector_pct` |
| Cash | cash above `min_cash_pct` |

So the most you lose if a buy reaches its stop-working price is about `risk_per_trade_pct` of the
portfolio. The answer opens with a decisions table (symbol, decision, shares, cost, entry, stop-working
price, rating, conviction, which rule set the size), then the reasoning, then the portfolio after the
trades. Prices come from the market-data API when `FMP_API_KEY` is set, otherwise from the file (with its
date). Holdings over a limit get a TRIM with the number of shares over.

## Benchmark against other prediction algorithms

```bash
python benchmark.py            # walk-forward on real prices: free, about a minute
python benchmark.py --ledger   # and score the bot's own logged forecasts that have matured
```

Every month-end since 2009, five classic algorithms (12-1 momentum, 10-month trend, low volatility, the
5-year historical mean, short-term reversal), a blend of three of them and a random control each rank 30 US
large caps and 15 country ETFs using only the prices up to that day. Each ranking is then scored against the
next 12 months and the next month: rank correlation (IC) with Newey-West t-statistics, top-minus-bottom-third
return, the hit rate of the implied Overweight/Underweight calls, and the same IC over consecutive 3-year
blocks so one lucky period cannot carry a result. Each algorithm's IC is also tested against its own luck range: its
scores handed to the wrong stocks, 200 times, which keeps how slowly the signal changes. Trust "Beyond luck?"
over the t-statistic, which runs a little hot on universes this small. A report is saved to `bench_runs/`.

The bot is not backtested: the model already knows how those years went, so any past-date test would be
contaminated. It is tested forward instead. Each rating it gives on a listed stock is logged to
`forecasts/ledger.jsonl` (git-ignored) with the price and date, and `--ledger` scores each one once its 12
months are up. The universes are today's survivors, which flatters long-only returns: compare the rows with
each other, not with zero.

## Calibration

```bash
python calibration.py            # several minutes: 40 permutation runs per algorithm as the luck baseline
python calibration.py --ledger   # and the bot's own matured forecasts
```

Do the numbers mean what they say? Each algorithm is turned into an expected return, a probability of
beating the median and 50/80/90% ranges, refitted every month on outcomes already known, on three
universes (US large caps, country ETFs, and the nine US sector ETFs, which have no survivor bias). It
reports the stock-against-stock calibration slope (1 = forecast gaps came true in full), out-of-sample R2,
Brier skill, calibration error, range coverage, a market-level check shared by every algorithm, the same
slope over consecutive 3-year blocks, and what the bot's +15%/-10% rating thresholds delivered on top of each
algorithm. Every score is compared with 40 runs of the same algorithm's scores handed to the wrong stocks:
`*` marks better than luck, `!` worse. For the bot, it also checks how often its bull and bear cases came true against the probabilities
it gave them.

## Robustness

```bash
python robustness.py             # about 6 minutes
```

Three harder tests on the benchmark: permutation p-values with a Benjamini-Hochberg correction across all
36 algorithm-universe-horizon tests (false-discovery rate 10%); a holdout that picks what worked before
2018 and checks it from 2018 on; and the top-third portfolios after 0-50 basis points of trading costs.

## Accuracy checks and the test kit

The system prompt makes the bot run eleven accuracy checks before every answer (ACCURACY CHECKS): maths in the
calculator, identity (name, ticker, exchange, share class), freshness, reconciliation (market cap, EV, multiples,
parts to totals, one share count), periods and units, outliers, confirmation against the primary filing, scenario
consistency, citations, "unknown" instead of guesses, and a closing KEY NUMBERS JSON block.

```bash
python verify.py memos/<memo>.md                 # check a memo's KEY NUMBERS block: free
python ipo_bot.py --audit "Rate <ticker>"        # answer, then a second model call audits it against the 11 checks
python kit.py calc                               # calculation cases against exact expected outputs: free
python kit.py traps --yes                        # hallucination traps (fake company, made-up ticker, future date...)
python kit.py stale --yes                        # tools off: no current price or rate allowed
python kit.py rules --yes                        # hype, inside information, guarantees, pump posts, personal amounts
python kit.py consistency --yes                  # the same question 3 times plus rewordings
python kit.py golden --yes                       # the golden set, once you fill in testkit/golden_set.json
python kit.py find-ipos 2026-07-01 2026-09-30    # first-time IPOs from SEC prospectuses, for the backtest: free
python kit.py backtest --yes                     # those IPOs as of the day before pricing, scored at 1, 6, 12 months
python kit.py track                              # the bot's matured recommendations: vs SPY, Brier, by conviction
python kit.py regress --yes                      # calc, traps, stale, rules, golden; scores kept in testkit/history.jsonl
```

| Part | Where |
|---|---|
| Golden set template (25 stocks and IPOs) and scoring rubric | `testkit/golden_set.json`, `testkit/rubric.md` |
| Calculation tests with exact outputs | `testkit/calc_cases.json` |
| Hallucination traps, stale-data, rule and consistency tests | `testkit/traps.json`, `stale.json`, `rules.json`, `consistency.json` |
| Backtest without hindsight | `kit.py backtest`: the data tools are cut off at the as-of date (filings, XBRL, FRED, prices) and web search is off; only IPOs after the model's June 2026 training cutoff are clean |
| Live tracking | every rating is logged with conviction and all three scenarios (`forecasts/ledger.jsonl`); `track.py` scores them |
| Second-pass auditor | `auditor_prompt.md`, `audit.py` (structured JSON verdict; recomputed so a failed check cannot pass) |
| Regression history | `testkit/history.jsonl`, keyed by a fingerprint of the system prompt |
| JSON checker | `verify.py` |

Claude Opus 5.5 does not accept a temperature setting (the API rejects it), so the consistency test measures the
spread at the default. Every `--yes` command calls the model and costs API credit.

## Test

```bash
python -m pytest -q tests          # offline, free: tools, checks and the loop against a stand-in client
IPO_BOT_LIVE=1 python -m pytest -q tests/test_live_market.py   # live, free: real SEC filings, portfolio rules on today's prices
python stress_test.py --yes        # the 11 stress-test questions, live: costs API credit
python stress_test.py --yes 2 4    # just some of them
```

`stress_test.py` saves each memo with mechanical checks (disclaimer last, probabilities add up, a
stop-working price with every rating, both IPO ratings, no hype, the inside-information refusal, no
personal dollar amounts). Replace the `[bracketed]` placeholders with real names first. Read the
memos as well: the checks catch missing parts, not bad reasoning.

## Files

| File | What |
|---|---|
| `system_prompt.md` | The bot's instructions. Change the rating thresholds and the memo layout here. |
| `ipo_bot.py` | The conversation loop: streaming, adaptive thinking at high effort, prompt caching, server-side fallback on a declined request. |
| `tools.py` | EDGAR (lookup, filings, document text, XBRL financials), FRED, market data, a calculator for every number, the two portfolio tools, and the forecast log. |
| `portfolio.py` | Loads and checks your portfolio file, values it, finds rule breaches, and sizes a purchase by your rules. |
| `checks.py` | The memo rules a script can check. |
| `stress_test.py` | The 11 test questions, graded by `checks.py`. |
| `benchmark.py` | Walk-forward tests of simple prediction algorithms, and scoring of the bot's logged forecasts. |
| `verify.py` / `audit.py` / `kit.py` / `track.py` | The KEY NUMBERS checker, the second-pass auditor, the test kit, live tracking. |
| `robustness.py` | Multiple-testing correction, a before/after-2018 holdout, and trading costs. |
| `calibration.py` | Whether forecasts mean what they say: slopes, probabilities and ranges against outcomes, with a luck baseline. |

## Cost

Each question is a research run: several searches, filing reads and calculations on Claude Opus 5.5
($4 per million input tokens, $20 per million output). The system prompt and tools are cached, so
follow-ups cost less. Lower `EFFORT` in `ipo_bot.py` to spend less per question.

## If you make it public

Not legal advice. Before anyone else uses it, ask a securities lawyer about:

1. **The adviser registration exemption for publishers.** Under the Investment Advisers Act and
   *Lowe v. SEC*, impersonal, regular publications are exempt. Does a bot that answers each user's
   own questions still qualify?
2. **Massachusetts.** Does it need registration with the Securities Division of the Secretary of the
   Commonwealth? It depends partly on whether you charge.
3. **Users abroad.** For example the UK's financial promotions rules or the EU's MiFID.
4. **Your data licences.** Most market-data licences ban redistributing their data publicly.
