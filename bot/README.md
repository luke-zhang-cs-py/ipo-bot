# The keyless bot

`python -m bot update` collects market and IPO data from keyless public sources, checks it, makes
point-in-time predictions, scores them when they resolve, and writes a health report. A fresh clone runs it
with no keys and no configuration.

It predicts two things:

- **Stocks:** whether each S&P 500 member closes higher the next session (a probability, plus an expected log
  return). The prediction is made at 22:00 UTC, after the close.
- **IPOs:** whether a first-time IPO's first close is 20% or more above its offer price (a probability, plus
  an expected first-day return). The prediction is made when the registration becomes effective, before the
  first trade.

```bash
pip install -r requirements.txt
python -m bot update                 # everything a fresh clone needs (the "all" job)
python -m bot update --job daily     # or one job: daily, edgar, weekly, monthly
python -m bot backtest               # walk-forward backtest and leakage test -> reports/backtest.md
python -m bot retrain                # refit; the new model is adopted only if it beats the current one
python -m bot health                 # the latest health summary
```

The first run reads the history from 2016 onward, about 500 Yahoo requests at 2 a second. The EDGAR backfill
then fills in over the next few runs, within a request budget per run. Set `SEC_USER_AGENT="Your Name
you@example.com"` (in the environment or `.env`) to include EDGAR. This is not a key: SEC's fair-access policy
asks every caller to name itself with a contact address and refuses requests that don't. Without it, everything
else runs and the health report says EDGAR was skipped.

## Sources

Every source sits behind an adapter in `bot/adapters/`. An adapter fetches, parses and checks the shape of the
response; a changed format becomes a schema failure, not stored data. Every raw response is kept, gzipped,
under `data/raw/<date>/<host>/`. When a source fails, the next one is tried and the run records a warning.

| Data | Primary | Fallback | Terms |
|---|---|---|---|
| Daily bars, splits, dividends | Yahoo chart API | Cboe CDN | Yahoo: unofficial, no published API terms (see Limits) |
| Second price source (reconciliation) | Cboe CDN | — | public files behind cboe.com's pages |
| Treasury yields (3M, 2Y, 10Y) | US Treasury CSV | FRED CSV | public domain |
| VIX | Cboe `VIX_History.csv` | FRED CSV | published by Cboe |
| S&P 500 members | Wikipedia (MediaWiki parse API) | the last members read | CC BY-SA; automated reads allowed with a named User-Agent |
| Listed or delisted | Nasdaq Trader symbol files | the last status read | public directory files |
| IPO filings (S-1, S-1/A, F-1, F-1/A, EFFECT, 424B4, RW) | EDGAR daily and quarterly form indexes | EDGAR full-text search | SEC public data: under 10 requests a second, User-Agent with a contact |
| Company record (SIC, earlier reports) | EDGAR submissions | — | as above |
| Prospectus facts (range, offer, shares, lead bank, symbol) | the filing's document | — | as above |

Some keyless sources were tried and left out:

- **Nasdaq's quote API** refuses any client that names itself, so reading it would mean pretending to be a
  browser.
- **Stooq** sits behind a JavaScript proof-of-work page.

Keyed sources could be added as optional adapters; none is needed.

## Schedule

`.github/workflows/bot.yml` runs these jobs. For a local machine, the alternatives are `scripts/bot_crontab.txt`
(cron) and `scripts/bot_schedule.ps1` (Windows Task Scheduler, slots in UTC).

| Job | When (UTC) | What |
|---|---|---|
| daily | 22:00, Monday–Friday | universe, prices, macro, reconciliation, scoring, predictions |
| edgar | every 3 hours, Monday–Friday | filings, companies, documents, first trades; IPO scoring and predictions |
| weekly | Saturday 06:00 | walk-forward backtest and leakage test → `reports/backtest.md` |
| monthly | the 1st, 08:00 | retrain; adopted only if better out of sample |

Every job:

- holds a lock file, so two runs never overlap; a lock older than 6 hours is from a run that died and is taken
  over with a warning;
- records a run row (start, end, status, summary);
- writes the health report.

A missed slot shows up in the next health report as a stale job.

Updates are idempotent. A rerun at the same moment reads nothing new and stores nothing new.

Each step fetches only what is new:

- **Prices:** from the newest stored session, re-reading 5 sessions before it to catch revisions. After a
  split, the whole history is re-read and stored as a new version.
- **Treasury yields:** from the newest stored year.
- **EDGAR:** day by day from a cursor. Older quarters are backfilled newest first, within a budget of 1,500 SEC
  requests per run.

Requests time out after 30 seconds. Failures are retried with exponential backoff (1, 2, 4, 8 seconds), or
after the server's `Retry-After` on a 429. Logs are JSON lines in `data/logs/bot.jsonl`.

## Storage

All data is kept in one SQLite file, `data/bot.sqlite`. In GitHub Actions it lives in the Actions cache.

- **Every row has `available_at`:** the moment the bot could first have known it.
- **First releases are kept.** A source that later changes a value (a split rescales old prices, an amended
  filing) adds a new version with the time it was read.
- **Reads are point-in-time.** A read "as of" a moment sees, for each key, the newest version available then,
  so a backtest only uses what was knowable on the day.
- **Backfilled history** carries its nominal publication time (16:00 New York plus the source's delay)
  instead of the read time.
- **History can't be changed.** Triggers refuse every UPDATE and DELETE.
- **Ledgers are append-only.** Predictions, outcomes and models are written once and never revised. They are
  mirrored to `tracking/bot/*.jsonl`, which is committed, so the record survives losing the database.
- **Delisted stocks and withdrawn or postponed IPOs are kept.**

## Checks

### On the data

These run on every update; the results go to the health report.

- **Schema and types:** every adapter checks the shape of its response, and a changed format becomes a schema
  failure.
- **Prices:**
  - duplicates within a read are dropped, with a warning;
  - prices must be above zero, with the close inside the day's range;
  - a daily move beyond 50% is flagged unless a split or dividend on record explains it.
- **Missing sessions:** NYSE trading days with no bar inside a symbol's history, checked against the exchange
  calendar in `bot/markets.py`. The calendar covers rule 7.2 holidays, special closures and Eastern time with
  DST.
- **Staleness:**
  - the newest bar must not lag the last closed session by more than one session;
  - each macro series must be current;
  - each scheduled job must have run since its last slot.
- **Reconciliation:** Yahoo's and Cboe's closes must agree within 0.5%. This is checked over the last 20
  sessions for the S&P 500 index every day, a rotating twentieth of the universe (all of it every four weeks),
  and any symbol with a suspicious move. Days before a split inside the window are skipped, because the two
  sources scale them differently.
- **IPOs:**
  - the offer must be within half the range's low to twice its high;
  - the first trade must fall within two sessions of the 424B4;
  - the first trade must come after effectiveness and registration;
  - follow-ons and SPACs are excluded;
  - withdrawn and postponed deals are kept and counted.
- **Point in time:** no bar may be available before its close, and the append-only triggers must be in place.

### On the predictions

- **Outputs:** every output must be finite, every probability within [0, 1], and every requested subject
  answered by every forecaster. A forecast that fails is never recorded. The same data gives the same
  predictions, which the replay test checks.
- **Walk-forward backtest** (`weekly`):
  - **Stocks:** refitted every 21 sessions on the previous three years.
  - **IPOs:** refitted each quarter on the IPOs that had traded before it.
  - **Scores:** Brier, log loss, hit rate, calibration (ECE), MAE and RMSE, reported separately for stocks and
    IPOs.
- **Baselines** for stocks:
  - **base rate:** each stock's up-day rate so far;
  - **random walk:** a probability of 0.5 and a return of 0.
- **Baselines** for IPOs:
  - **base rate:** the pop rate so far;
  - **average recent IPO:** the last 20 IPOs' pop rate and mean return.
- **Significance:** the model is compared with each baseline on Brier, log loss and absolute error. The tests
  are Diebold-Mariano with the Harvey-Leybourne-Newbold correction and a moving-block bootstrap, Holm-corrected.
  For stocks, the test series is the daily mean loss, not 500 correlated rows a day.
- **Scrambled-future leakage test:** everything after a cutoff is replaced with scrambled values. The features
  up to the cutoff, and a model fitted on what was known then, must not change at all. A test shows the check
  catching a feature that peeks at tomorrow.
- **Live tracking:**
  - predictions are scored when they resolve;
  - a stock target with no close two sessions on, or a withdrawn IPO, resolves as void;
  - rolling metrics are kept per week for stocks and per quarter for IPOs;
  - three periods in a row worse than the best baseline is a warning, and so is a calibration error above 0.10.
- **Retraining** (`monthly`): the candidate is judged only on outcomes after the current model's training
  data, never on data the current model was fitted on, and is adopted only if its Brier is lower.

## Tests

```bash
python -m pytest -q tests/bot --cov=bot --cov-branch   # offline: no network, no keys; fails under 100% coverage
python -m ruff check bot tests/bot && python -m black --check bot tests/bot && python -m mypy
mutmut run                                             # Linux only (CI runs it weekly)
```

- **Adapters** are tested on real responses recorded from each source (trimmed, in
  `tests/bot/fixtures/recorded/`), and on malformed, empty and changed ones.
- **End-to-end tests** run whole updates against `tests/bot/world.py`. It is a synthetic, seeded market that
  answers every source in that source's own format. It includes:
  - a 2-for-1 split;
  - a special dividend;
  - a delisting;
  - a missing day;
  - a Yahoo-Cboe disagreement;
  - an IPO calendar with range revisions, withdrawals, postponements, a follow-on and a SPAC.
- **Error paths:** time-outs, connection resets, 429s, 5xx errors, schema changes, sources down, the lock held,
  a crash mid-run, and DST and holiday clocks.
- **Property tests:** Hypothesis checks the calendar, the store and the scores.
- **No network:** a socket guard fails any test that tries to use it.
- **`# pragma: no cover`:** three lines, each with its reason. They are the real network call, a guard for a
  missing optional package, and an unreachable loop exit.

On the synthetic world, the backtest finds the signal that was planted in the IPOs (the range revision): the
model beats both baselines on Brier, with Holm-adjusted DM p = 0.018. It finds no edge in the stocks, which are
random walks there. That is the null result it should give.

## Limits

- **Yahoo's chart API is unofficial.** Yahoo publishes no terms for reading it programmatically. The bot reads
  it slowly under its own name and reconciles it against Cboe. If that's not acceptable, a keyed price adapter
  (Tiingo, Alpaca) would replace it.
- **Membership history starts with the bot's first snapshot.** Wikipedia's table lists today's members only,
  so backtests before that date use the members seen then: survivorship bias, which flatters stock results.
  Each health report states the date this history starts.
- **Yahoo drops delisted companies' prices.** IPOs that later delisted can lack a first trade; the health
  report lists them as missing.
- **EDGAR needs a contact address.** Without `SEC_USER_AGENT`, there are no IPO predictions.
- **FRED is blocked on some networks.** It is only a fallback.
- **These are research predictions, not investment advice.**
