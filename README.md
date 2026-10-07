# IPO bot

[![tests](https://github.com/luke-zhang-cs-py/ipo-bot/actions/workflows/tests.yml/badge.svg)](https://github.com/luke-zhang-cs-py/ipo-bot/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Claude](https://img.shields.io/badge/Claude-Opus%205.5-0F766E.svg)](https://docs.anthropic.com/)

IPO and stock research on Claude that answers in one page, then lets code **check every number in it**.

### ▶ [Project a stock, size a trade, check a memo →](https://luke-zhang-cs-py.github.io/ipo-bot/app/)
Runs in your browser: every NYSE ticker the SEC lists, searchable by ticker, company or sector; a memo's bull, base
and bear cases drawn as a price projection (2,000 simulated paths that average out at the probability-weighted value);
a table of trade ideas you can sort by expected return, reward to risk, chance of hitting the stop or shares allowed;
and the bot's own sizing rules and memo checks, ported from the Python and tested against it. No API key, nothing uploaded.

![A price projection with bull, base and bear paths and 10-90% bands, then sizing a purchase as the stop-working price moves, then the memo checker catching a rating the numbers don't support](docs/demo.gif)

**[Read the write-up →](https://luke-zhang-cs-py.github.io/ipo-bot/)** — what a memo contains, why its numbers can be
trusted, and what an honest benchmark found.

## How it works

```
question -> Claude + tools (SEC EDGAR, XBRL, FRED, market data, web, calculator) -> one-page memo -> KEY NUMBERS -> verify.py
```

- **The memo:** a rating, bull, base and bear scenarios that add up to 100%, a probability-weighted value, a
  *stop-working price*, dated catalysts and sources. An IPO gets two ratings: at the offer and in the aftermarket.
- **11 accuracy checks** run before every answer: identity, freshness, reconciliation, periods and units, outliers, the
  primary filing, scenario consistency, citations, and "unknown" instead of a guess.
- **KEY NUMBERS:** every memo ends with a JSON block that `verify.py` recomputes: market cap, EV, multiples, the
  scenario maths and the rating thresholds. A second model call (`audit.py`) audits the memo against its sources.
- **Portfolio mode:** with your holdings and rules loaded, it decides buy, add, hold, trim or sell, but the size comes
  from the smallest of four limits in your own file: risk per trade, position cap, sector cap, cash floor.

## What the evaluation found

The model knows how past years turned out, so it is scored forward: each rating is logged with its price and
scenarios (`src/track.py`). To give it a bar to clear, seven simple prediction algorithms were tested every month from 2009
on 30 US large caps, 15 country ETFs and the 9 US sector ETFs, each seeing only the prices known that day.

| Test | Result |
|---|---|
| 36 algorithm-universe-horizon tests, permutation p-values, false-discovery rate 10% | 4 pass a naive 5% bar; **0 of 36** survive |
| Picked on 2009-2017, checked from 2018 | 1 of 5 held up |
| 12-month 80% ranges | held the outcome only 70% to 77% of the time |

Simple price signals give no 12-month edge that holds up, and ranges drawn from history are too narrow; both findings
went back into the prompt. Details: [`evaluation/`](evaluation/).

## Run it

```bash
pip install -r requirements.txt
python src/ipo_bot.py "Rate the <company> IPO"          # needs ANTHROPIC_API_KEY; data keys in .env.example
python src/ipo_bot.py --portfolio portfolio.json         # review your holdings by your own rules
python src/ipo_bot.py --audit "Rate <ticker>"            # plus a second-pass audit
```

## Tests

```bash
python -m pytest -q tests                  # 112 offline tests, free (9 need Node: the browser demo against the Python)
IPO_BOT_LIVE=1 python -m pytest -q tests/test_live_market.py   # 15 live checks on real SEC filings and prices
python testkit/kit.py regress --yes        # the test kit, live: calculation cases, hallucination traps, a tools-off
                                           # stale-data test, rule tests; costs API credit
```

The test kit (`testkit/`) also has a golden-set template, a consistency test, and a no-hindsight IPO
backtest that cuts the data tools off at the day before pricing.

## Layout

```
src/          the bot (ipo_bot.py), its data tools, portfolio sizing, accuracy checks, verify, audit, forecast ledger
prompts/      the system and auditor prompts
evaluation/   benchmark, calibration, robustness, stress test
testkit/      the test kit (kit.py) and its cases
tests/        the pytest suite
docs/         the write-up and the browser demo (GitHub Pages)
scripts/      rebuilds the NYSE list from SEC data, re-records the demo GIF
```

## Limits

- No live run is recorded: every live test needs an API key and costs credit.
- The stock universes are today's survivors, which flatters long-only results; the sector ETFs are the survivor-free check.
- Views on securities, not personal advice; every answer ends with that disclaimer. Before anyone else uses it, ask a
  securities lawyer about adviser registration (including Massachusetts) and your data licences.
