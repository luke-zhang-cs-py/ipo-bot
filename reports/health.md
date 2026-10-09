# Bot health

Run `20261009T005117Z-edgar-763afb` at 2026-10-09T00:51:17Z, data through the 2026-10-08 session. **Status: failing.**

## Coverage

- Universe: 0 symbols; 0 with prices; 0 current.
- Missing tickers (no prices at all): none
- Stale: SPX, UST10Y, UST2Y, UST3M, VIX
- Missing trading days inside histories: 0 days over 0 symbols
- Missing fields in the last year of bars: none
- Delisted symbols kept: 0; membership history recorded from not yet (earlier backtests use the membership seen then).

## Checks

- staleness: 5

Errors (first 20):

- staleness SPX: no prices at all
- staleness UST3M: newest value None, behind 2026-10-08
- staleness UST2Y: newest value None, behind 2026-10-08
- staleness UST10Y: newest value None, behind 2026-10-08
- staleness VIX: newest value None, behind 2026-10-08

## Sources this run

- edgar_skipped: reason=SEC_USER_AGENT is not set; EDGAR asks every caller for "Name email"

## IPO pipeline

- 0 first-time IPO deals of 0 registrants; by status: none
- 0 with an offer price read, 0 with a first trade.
- EDGAR daily index read through not yet; backfill reached not started.

## Live tracking

- stock: no resolved predictions yet
- ipo: no resolved predictions yet
