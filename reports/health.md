# Bot health

Run `20261010T013510Z-daily-a9b7f9` at 2026-10-10T01:35:10Z, data through the 2026-10-09 session. **Status: degraded.**

## Coverage

- Universe: 503 symbols; 503 with prices; 503 current.
- Missing tickers (no prices at all): none
- Stale: none
- Missing trading days inside histories: 1 days over 1 symbols
- Missing fields in the last year of bars: volume 1
- Delisted symbols kept: 0; membership history recorded from 2026-10-09T01:51:19Z (earlier backtests use the membership seen then).

## Checks

- missing_days: 1
- moves: 12
- staleness: 1

## Sources this run

- prices_fallback: symbol=VRSK, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VRSN, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VRT, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VRTX, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VST, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VTR, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VTRS, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VYLR, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=VZ, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WAB, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WAT, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WDAY, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WDC, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WEC, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WELL, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WFC, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WM, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WMB, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WMT, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WRB, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WSM, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WST, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WTW, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WY, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=WYNN, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=XEL, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=XOM, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=XYL, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=XYZ, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]
- prices_fallback: symbol=YUM, source=cboe, errors=[('yahoo', 'rate_limited: 429 from query1.finance.yahoo.com')]

## IPO pipeline

- 0 first-time IPO deals of 0 registrants; by status: none
- 0 with an offer price read, 0 with a first trade.
- EDGAR daily index read through not yet; backfill reached not started.

## Live tracking

- stock, 2026-W41: base_rate Brier 0.2438 (n=470); model Brier 0.2459 (n=470); random_walk Brier 0.2500 (n=470)
- ipo: no resolved predictions yet
- stock model stock-2026-10-09-bc9b4b29c3, shrinkage 1
- **stock: calibration drift: expected calibration error 0.119 over the last 1 periods (limit 0.10)**
