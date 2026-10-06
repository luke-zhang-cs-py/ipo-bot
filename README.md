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
blocks so one lucky period cannot carry a result. A report is saved to `bench_runs/`.

The bot is not backtested: the model already knows how those years went, so any past-date test would be
contaminated. It is tested forward instead. Each rating it gives on a listed stock is logged to
`forecasts/ledger.jsonl` (git-ignored) with the price and date, and `--ledger` scores each one once its 12
months are up. The universes are today's survivors, which flatters long-only returns: compare the rows with
each other, not with zero.

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
