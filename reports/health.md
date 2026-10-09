# Bot health

Run `20261009T220158Z-edgar-179d63` at 2026-10-09T22:01:58Z, data through the 2026-10-09 session. **Status: failing.**

## Coverage

- Universe: 503 symbols; 503 with prices; 502 current.
- Missing tickers (no prices at all): none
- Stale: HUBB
- Missing trading days inside histories: 1 days over 1 symbols
- Missing fields in the last year of bars: none
- Delisted symbols kept: 0; membership history recorded from 2026-10-09T01:51:19Z (earlier backtests use the membership seen then).

## Checks

- missing_days: 1
- moves: 12
- staleness: 1

Errors (first 20):

- staleness HUBB: newest bar 2026-10-07, 2 trading days behind 2026-10-09

## Sources this run

- edgar_skipped: reason=SEC_USER_AGENT is not set; EDGAR asks every caller for "Name email"

## IPO pipeline

- 0 first-time IPO deals of 0 registrants; by status: none
- 0 with an offer price read, 0 with a first trade.
- EDGAR daily index read through not yet; backfill reached not started.

## Live tracking

- stock: no resolved predictions yet
- ipo: no resolved predictions yet
- stock model stock-2026-10-09-bc9b4b29c3, shrinkage 1
