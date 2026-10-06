# Golden set: how to fill it in and how it is scored

## Filling in an entry (testkit/golden_set.json)

For each company, open the filing named in `filing` and copy:

| Field | Listed stocks | IPOs |
|---|---|---|
| `as_of` | the date you take the figures (and the price) | the day before listing (`YYYY-MM-DD`): the bot runs as of that date |
| `expected` | price (close on `as_of`), diluted_shares, revenue, net_income, debt, cash | offer_price_low, offer_price_high, shares_offered, revenue, net_income |
| `expected_source` | a word the bot's citation must contain: the form (`10-Q`, `20-F`), or the accession number for a strict test | `S-1/A`, `424B4`, or an accession number |
| `subject.exchange`, `subject.share_class` | from the filing's cover page | from the prospectus cover |

Rules for the figures:
- Full units, in the filing's currency converted to USD only if the bot is asked for USD: 2150000000, not 2,150 (millions).
- Revenue and net income: the period the filing reports most recently (say which in a note); the bot's period must match.
- Diluted shares: the cover-page share count or the diluted weighted average, but choose one and note it.
- Leave a field null if you cannot find it in the filing: it is skipped, not scored.

## How an entry is scored (kit.py golden)

Each entry scores four parts, equally weighted, each 0 to 100%:

| Part | Full marks when |
|---|---|
| Extraction | every filled-in figure in the bot's KEY NUMBERS is within `tolerance_pct` (default 1%) of yours |
| Maths | every verify.py check on the block passes (market cap, EV, multiples, scenarios, rating) |
| Citation | each figure's `source` contains the `expected_source` word you gave |
| Identity | the block's ticker matches the entry's ticker (share class and ADRs are the trap here) |

An entry passes at 100%. The suite reports the pass count and the average score, and `kit.py regress` keeps both in
`testkit/history.jsonl` so a prompt change that drops a score shows up.

## Reading a failure

Each answer is saved under `testkit/runs/<time>/golden/` with its checks and the tool results it used. Common causes:
- Extraction off by 1,000x: a unit slip (thousands vs millions as filed).
- Extraction off by a few %: a different period or share count; check the bot's `period` and source.
- Citation miss: the figure came from a data API or the web rather than the filing.
- Identity miss: GOOG for GOOGL, BRK.A for BRK.B, or the home listing for an ADR.

## IPOs and hindsight

The ten IPOs in the template listed between 2023 and 2025, inside the model's training data. The bot is run as of the
day before listing, with its data tools cut off at that date and web search off, but it may still remember the
outcome. They test extraction, maths and citation. The no-hindsight test is `kit.py backtest`, on IPOs after June 2026.
