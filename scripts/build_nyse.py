"""Rebuild docs/app/nyse.json: every NYSE ticker the SEC lists, with its company, CIK and a sector from its SIC code.

    python scripts/build_nyse.py

Sources are the SEC's own (public domain): company_tickers_exchange.json for the tickers and exchange, and each
company's EDGAR header for its SIC code. Needs SEC_USER_AGENT in .env. About 2,600 small requests, six at a time but
under the SEC's fair-access limit of 10 a second; finished companies are cached in bench_data/, so a rerun picks up
where it stopped.
The SEC lists no prices: the demo asks for one.
"""
import concurrent.futures
import datetime as dt
import html
import json
import pathlib
import re
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "docs" / "app" / "nyse.json"
CACHE = ROOT / "bench_data" / "nyse_sic.json"
SEC_INTERVAL = 0.125       # seconds between request starts: 8 a second, under the SEC's fair-access limit of 10
SECTORS = ["Technology", "Health Care", "Financials", "Real Estate", "Energy", "Materials", "Industrials",
           "Consumer Discretionary", "Consumer Staples", "Communication", "Utilities", "Other"]

# SIC ranges to a sector, most specific first. An approximation: SIC predates today's sector schemes.
SIC_RANGES = [
    (2830, 2836, "Health Care"), (3841, 3851, "Health Care"), (8000, 8099, "Health Care"), (5120, 5129, "Health Care"),
    (2840, 2844, "Consumer Staples"), (5140, 5149, "Consumer Staples"), (5400, 5499, "Consumer Staples"), (5910, 5912, "Consumer Staples"),
    (3570, 3579, "Technology"), (3600, 3629, "Technology"), (3640, 3699, "Technology"), (3820, 3829, "Technology"), (7370, 7379, "Technology"),
    (3630, 3639, "Consumer Discretionary"), (3710, 3719, "Consumer Discretionary"), (3750, 3799, "Consumer Discretionary"),
    (7310, 7319, "Communication"), (2700, 2799, "Communication"), (4800, 4899, "Communication"), (7800, 7899, "Communication"),
    (4950, 4959, "Industrials"), (6798, 6798, "Real Estate"), (6500, 6553, "Real Estate"),
    (1200, 1399, "Energy"), (2900, 2999, "Energy"),
    (100, 999, "Consumer Staples"), (1000, 1099, "Materials"), (1400, 1499, "Materials"), (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"),
    (2500, 2599, "Consumer Discretionary"), (2600, 2699, "Materials"), (2800, 2899, "Materials"),
    (3000, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"), (3400, 3899, "Industrials"),
    (3900, 3999, "Consumer Discretionary"), (4000, 4799, "Industrials"), (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"), (5200, 5999, "Consumer Discretionary"), (6000, 6799, "Financials"),
    (7000, 7299, "Consumer Discretionary"), (7300, 7399, "Industrials"), (7500, 7599, "Consumer Discretionary"),
    (7900, 7999, "Consumer Discretionary"), (8200, 8299, "Consumer Discretionary"), (8100, 8999, "Industrials"),
]


def sic_sector(sic):
    """The sector for a SIC code (a string or number); Other when there is none or it is 9995/9999 (non-operating)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Other"
    for lo, hi, sector in SIC_RANGES:
        if lo <= code <= hi:
            return sector
    return "Other"


def industry(desc):
    """The SIC description as a readable name: the SEC escapes its ampersands twice ("&amp;amp;") and writes in capitals."""
    text = html.unescape(html.unescape(desc or "")).strip()
    return re.sub(r"\b([A-Za-z])([A-Za-z']*)", lambda m: m.group(1).upper() + m.group(2).lower(), text)


def sic_of(cik, get, headers):
    url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%010d&type=&dateb=&owner=include&count=1&output=atom" % cik)
    text = get(url, headers)
    code = re.search(r"<assigned-sic>(\d+)</assigned-sic>", text)
    desc = re.search(r"<assigned-sic-desc>(.*?)</assigned-sic-desc>", text)
    return [code.group(1) if code else None, desc.group(1).strip() if desc else None]


def look_up_all(ciks, lookup, cache, save=lambda: None, interval=SEC_INTERVAL, workers=6, retries=3, backoff=2.0):
    """Fill cache[str(cik)] with lookup(cik) for each CIK, a few at a time, starting requests `interval` apart.

    A CIK whose every try fails is returned in the failed list and left out of the cache, so the next run retries it.
    save() is called every 100 lookups and at the end, so an interrupted run keeps its progress.
    """
    lock, gate = threading.Lock(), {"next": 0.0}

    def one(cik):
        for attempt in range(retries):
            with lock:                   # requests start at least `interval` apart, across all threads
                wait = gate["next"] - time.monotonic()
                gate["next"] = max(gate["next"], time.monotonic()) + interval
            if wait > 0:
                time.sleep(wait)
            try:
                return cik, lookup(cik)
            except Exception as e:       # a throttled or failed request: wait and try again
                print(f"  CIK {cik}: {e}; retrying", flush=True)
                time.sleep(backoff * (1 + attempt))
        return cik, None

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (cik, sic) in enumerate(pool.map(one, ciks), 1):
            if sic is None:
                failed.append(cik)
            else:
                cache[str(cik)] = sic
            if n % 100 == 0 or n == len(ciks):
                save()
                print(f"  {n}/{len(ciks)}", flush=True)
    return failed


def rows_for(nyse, fields, cache):
    """The published rows, sorted by ticker: [ticker, name, cik, sector, industry]."""
    rows = []
    for r in nyse:
        sic, desc = cache.get(str(r[fields["cik"]]), [None, None])
        rows.append([r[fields["ticker"]], r[fields["name"]], r[fields["cik"]], sic_sector(sic), industry(desc)])
    return sorted(rows, key=lambda x: x[0])


def main():
    import ipo_bot
    import tools
    ipo_bot.load_env()
    headers = tools._sec_headers()
    raw = json.loads(tools._get("https://www.sec.gov/files/company_tickers_exchange.json", headers))
    fields = {f: raw["fields"].index(f) for f in ("cik", "name", "ticker", "exchange")}
    nyse = [r for r in raw["data"] if r[fields["exchange"]] == "NYSE" and r[fields["ticker"]]]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = sorted({r[fields["cik"]] for r in nyse} - {int(k) for k in cache})
    print(f"{len(nyse)} NYSE tickers, {len({r[fields['cik']] for r in nyse})} companies; {len(todo)} to look up")
    CACHE.parent.mkdir(exist_ok=True)
    failed = look_up_all(todo, lambda cik: sic_of(cik, tools._get, headers), cache,
                         save=lambda: CACHE.write_text(json.dumps(cache), encoding="utf-8"))
    if failed:
        print(f"  {len(failed)} companies could not be looked up and have no sector this time; run again to retry them")
    rows = rows_for(nyse, fields, cache)
    OUT.write_text(json.dumps({
        "source": "SEC company_tickers_exchange.json and EDGAR company headers (SIC); sector mapped from SIC",
        "as_of": dt.date.today().isoformat(), "fields": ["ticker", "name", "cik", "sector", "industry"], "rows": rows,
    }, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size // 1024} KB, {len(rows)} tickers)")


if __name__ == "__main__":
    main()
