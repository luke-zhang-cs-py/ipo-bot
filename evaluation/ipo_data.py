"""The IPO dataset behind ipo_eval.py: every US IPO with a final prospectus from 2015 on, what was public before
it listed, and how it traded after.

    python evaluation/ipo_data.py                 build or extend bench_data/ipo/ipos.jsonl (resumes; about an hour)
    python evaluation/ipo_data.py --summary       counts by year, and how many have prices
    python evaluation/ipo_data.py --refresh       re-read every IPO's cover and prices (after a parser fix)
    python evaluation/ipo_data.py --initial-ranges   add each IPO's first filed price range

Before the listing (features): the offer price, shares and lead bookrunner from the final prospectus (424B4),
the price range from the last amended registration (S-1/A or F-1/A) filed before it, the SIC code, and
whether it is a foreign filer. After (outcomes): the first day's open and close and the closes 21 and 252
trading days on, from Yahoo's daily chart with later splits undone so they compare with the offer price.

Every row keeps the date of each source it used, so ipo_eval.py can check that nothing a feature uses
is dated on or after the listing day. Blank-check companies (SPACs), and companies that filed annual or
quarterly reports before the prospectus (follow-ons, uplistings), are left out.

A listing Yahoo no longer has (a delisted company) keeps its prospectus data and has no prices: the
evaluation counts those per year, because leaving them out flatters every later return.
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
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:                                   # the OS certificate store, as the bot uses (needed behind inspecting proxies)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

OUT = ROOT / "bench_data" / "ipo" / "ipos.jsonl"
SEC_INTERVAL = 0.125          # seconds between SEC requests: 8 a second, under the fair-access limit of 10
HEAD_BYTES = 300_000          # a prospectus's cover page and summary sit in its first few hundred kilobytes
FIRST_YEAR = 2015
WORKERS = 6                   # filings read at once

# Bookrunners, most-named first; the first one found on the cover is taken as the lead.
BANKS = [
    "Goldman Sachs", "Morgan Stanley", "J.P. Morgan", "BofA Securities", "Merrill Lynch", "Citigroup", "Credit Suisse",
    "Barclays", "Deutsche Bank", "UBS", "Jefferies", "Wells Fargo", "RBC Capital", "Evercore", "Cowen", "TD Cowen",
    "Piper Sandler", "Piper Jaffray", "Stifel", "Raymond James", "William Blair", "Leerink", "SVB Leerink", "Needham",
    "Oppenheimer", "BMO Capital", "KeyBanc", "Baird", "Cantor", "Guggenheim", "Mizuho", "Nomura", "HSBC", "Truist",
    "SunTrust", "Canaccord", "Roth Capital", "Lake Street", "Craig-Hallum", "B. Riley", "Ladenburg", "Maxim Group",
    "Aegis Capital", "EF Hutton", "Boustead", "ThinkEquity", "Univest", "Network 1", "Benchmark", "Kingswood",
    "Joseph Gunnar", "Spartan Capital", "Dawson James", "Prime Number", "US Tiger", "Tiger Brokers", "AMTD",
    "Alexander Capital", "WestPark", "Network 1 Financial", "D. Boral", "Revere Securities", "R.F. Lafferty",
]


# ----------------------------------------------------------------------------- parsing (pure, tested)

def text_of(html_bytes):
    """Plain text from a filing's HTML: tags dropped, entities for spaces and dollars read, whitespace collapsed."""
    t = html_bytes.decode("latin-1") if isinstance(html_bytes, bytes) else html_bytes
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t)


def _num(s):
    return float(s.replace(",", ""))


def offer_price(text, rng=None):
    """The final offer price per share or ADS, or None. Candidates come strongest pattern first; with the
    marketed range known, the first within half its low to twice its high wins, so a per-share fee or
    discount on the cover ("$1.50 per share") is not taken for the price."""
    candidates = []
    for pat in (r"initial public offering price (?:is|of|per (?:share|ADS)[^$]{0,40}?)\s*\$\s?([\d,]+(?:\.\d+)?)",
                r"public offering price[^$]{0,60}\$\s?([\d,]+\.\d\d) per (?:share|ADS|American)",
                r"\$\s?([\d,]+\.\d\d) per (?:share|ADS)"):
        candidates += [_num(m.group(1)) for m in re.finditer(pat, text[:120000], re.I)]
    candidates = [c for c in candidates if 0.5 <= c <= 500]
    if rng:
        fits = [c for c in candidates if rng[0] * 0.5 <= c <= rng[1] * 2]
        return fits[0] if fits else None
    return candidates[0] if candidates else None


def price_range(text):
    """The marketed price range (low, high) from a preliminary prospectus, or None. A single assumed price
    (common for small fixed-price deals) is returned as (p, p)."""
    m = (re.search(r"between \$\s?([\d,]+(?:\.\d+)?) and \$\s?([\d,]+(?:\.\d+)?) per (?:share|ADS|American)", text, re.I)
         or re.search(r"offering price[^.$]{0,80}between \$\s?([\d,]+(?:\.\d+)?) and \$\s?([\d,]+(?:\.\d+)?)", text, re.I))
    if m and 0.5 <= _num(m.group(1)) <= _num(m.group(2)) <= 500:
        return _num(m.group(1)), _num(m.group(2))
    m = re.search(r"assumed (?:initial public )?offering price of \$\s?([\d,]+(?:\.\d+)?)", text, re.I)
    if m and 0.5 <= _num(m.group(1)) <= 500:
        return _num(m.group(1)), _num(m.group(1))
    return None


def shares_offered(text):
    """The number of shares or ADSs offered on the cover, or None."""
    m = re.search(r"([\d,]{5,}) (?:Shares|shares of (?:Class [A-Z] )?(?:common|ordinary)|American Depositary Shares|ADSs|Ordinary Shares)",
                  text[:20000])
    if m:
        n = _num(m.group(1))
        if 50_000 <= n <= 5_000_000_000:
            return n
    return None


def listing_symbol(text):
    """The ticker the shares listed under, from the prospectus ("under the symbol “SNAP”"), or None."""
    m = re.search(r"under the (?:trading )?symbols?\s*[“\"'‘]\s*([A-Z][A-Z0-9.\-]{0,6})\s*[”\"'’]", text[:80000])
    return m.group(1).rstrip(".-") if m else None             # "SNAP." at a sentence end is SNAP


def lead_bank(text):
    """The first bookrunner named on the cover (the lead, by convention), or None."""
    head = text[:60000]
    found = [(head.find(b), b) for b in BANKS if head.find(b) >= 0]
    if not found:
        return None
    name = min(found)[1]
    return {"Merrill Lynch": "BofA Securities", "Piper Jaffray": "Piper Sandler", "SunTrust": "Truist",
            "Leerink": "SVB Leerink", "Cowen": "TD Cowen", "Network 1 Financial": "Network 1"}.get(name, name)


def first_days(chart, listing_hint, offer):
    """From a Yahoo chart result: the listing day's open and close and the closes 21 and 252 trading days on,
    in the offer's terms (later splits undone). None when the chart's history doesn't start at this IPO."""
    ts = chart.get("timestamp") or []
    q = (chart.get("indicators") or {}).get("quote", [{}])[0]
    if not ts or not q.get("close"):
        return None
    days = [dt.datetime.fromtimestamp(t, dt.timezone.utc).date() for t in ts]
    hint = dt.date.fromisoformat(listing_hint)
    if abs((days[0] - hint).days) > 7:
        return None                    # history starts elsewhere: an uplisting or a reused ticker
    factor = 1.0                       # Yahoo divides earlier prices by every later split; multiply them back
    for s in ((chart.get("events") or {}).get("splits") or {}).values():
        if dt.datetime.fromtimestamp(s["date"], dt.timezone.utc).date() > days[0] and s.get("numerator") and s.get("denominator"):
            factor *= s["numerator"] / s["denominator"]
    def at(i, field):
        v = q[field][i] if i < len(ts) else None
        return round(v * factor, 4) if v else None
    out = {"listing_date": days[0].isoformat(), "open": at(0, "open"), "close": at(0, "close"),
           "close_21": at(21, "close"), "close_252": at(252, "close"), "split_factor": factor}
    # Yahoo's old histories are sometimes rescaled with no split on record (Carvana's 2017 opens at $2.70 on a
    # $15 offer). An IPO almost never opens below half its offer or above four times it, so outside that the
    # prices are kept but marked suspect, and the evaluation leaves them out of every return.
    out["suspect"] = bool(offer and out["open"] and not (0.5 <= out["open"] / offer <= 4))
    return out


# ----------------------------------------------------------------------------- fetching

class Sec:
    """SEC requests at the fair-access pace, with the bot's User-Agent."""

    def __init__(self):
        import ipo_bot
        import tools
        ipo_bot.load_env()
        self.tools, self.headers, self.next = tools, tools._sec_headers(), 0.0
        self.lock = threading.Lock()

    def _wait(self):
        """Requests start at least SEC_INTERVAL apart, across every thread."""
        with self.lock:
            start = max(self.next, time.monotonic())
            self.next = start + SEC_INTERVAL
        time.sleep(max(0.0, start - time.monotonic()))

    def json(self, url):
        for attempt in range(3):
            self._wait()
            try:
                return json.loads(self.tools._get(url, self.headers))
            except Exception:
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"SEC request failed three times: {url}")

    def head(self, url, limit=HEAD_BYTES):
        """The first `limit` bytes of a filing: the server ignores Range, so the read stops early instead.
        A busy server (503, 429) is retried after a pause."""
        for attempt in range(4):
            self._wait()
            req = urllib.request.Request(url, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read(limit)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))


def prospectus_hits(sec, year):
    """Every 424B4 filed in the year: {adsh, cik, name, ticker, file, file_date, sic}."""
    out, start = [], 0
    for half in ((f"{year}-01-01", f"{year}-06-30"), (f"{year}-07-01", f"{year}-12-31")):
        start = 0
        while True:
            d = sec.json("https://efts.sec.gov/LATEST/search-index?forms=424B4&dateRange=custom"
                         f"&startdt={half[0]}&enddt={half[1]}&from={start}")
            batch = d.get("hits", {}).get("hits", [])
            for h in batch:
                s = h["_source"]
                name = (s.get("display_names") or [""])[0]
                m = re.match(r"(.*?)\s+\(([A-Z0-9., -]+)\)\s+\(CIK", name)
                out.append({"adsh": s.get("adsh"), "cik": (s.get("ciks") or [""])[0].lstrip("0"),
                            "name": (m.group(1) if m else name.split("(CIK")[0]).strip(),
                            "ticker": (m.group(2).split(",")[0].strip() if m else ""), "file": h["_id"].split(":", 1)[1],
                            "file_date": s.get("file_date"), "sic": (s.get("sics") or [None])[0]})
            if len(batch) < 100:
                break
            start += 100
    return out


REPORTS = ("10-K", "10-Q", "20-F", "40-F")


def filings_of(sec, cik, before=None):
    """A company's filings as (form, date, accession, primary document), oldest first, and its SIC code.

    With `before`, the older pages are fetched only when the recent ones reach back past it and hold no
    annual or quarterly report from before it: most 424B4s are follow-ons, settled by the first page.
    """
    sub = sec.json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    recent = sub.get("filings", {}).get("recent", {})
    rows = list(zip(recent.get("form", []), recent.get("filingDate", []), recent.get("accessionNumber", []), recent.get("primaryDocument", [])))
    settled = before and (any(f in REPORTS and d < before for f, d, _, _ in rows) or (rows and min(d for _, d, _, _ in rows) <= before))
    if not settled:
        for f in sub.get("filings", {}).get("files", []):
            p = sec.json(f"https://data.sec.gov/submissions/{f['name']}")
            rows += list(zip(p.get("form", []), p.get("filingDate", []), p.get("accessionNumber", []), p.get("primaryDocument", [])))
    return sorted(rows, key=lambda r: r[1]), str(sub.get("sic") or "")


def yahoo_chart(symbol, around):
    t0 = int(dt.datetime.fromisoformat(around).replace(tzinfo=dt.timezone.utc).timestamp()) - 10 * 86400
    # to today, not just the first year: Yahoo adjusts old prices for every later split, so every split since
    # the listing has to be on record to undo it (Shopify's 2022 10-for-1, years of biotech reverse splits)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('.', '-')}"
           f"?period1={t0}&period2={int(time.time())}&interval=1d&events=split")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["chart"]["result"][0]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # not on Yahoo: delisted, or a ticker it never carried
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError):
            time.sleep(5 * (attempt + 1))
    return None


def build_row(sec, hit):
    """One IPO's row, or None when it is not a first-time operating-company IPO."""
    if hit["sic"] == "6770" or re.search(r"(?i)acquisition corp|\bspac\b|capital corp\.? [ivx]+\b|blank check", hit["name"]):
        return None
    filings, sic = filings_of(sec, hit["cik"], before=hit["file_date"])
    if sic == "6770":
        return None
    if any(f in REPORTS and d < hit["file_date"] for f, d, _, _ in filings):
        return None                    # already reporting: a follow-on or an uplisting, not an IPO
    # Only the company's first 424B4 is its IPO.
    first = min((d for f, d, _, _ in filings if f == "424B4"), default=hit["file_date"])
    if first < hit["file_date"]:
        return None
    base = f"https://www.sec.gov/Archives/edgar/data/{int(hit['cik'])}"
    text = text_of(sec.head(f"{base}/{hit['adsh'].replace('-', '')}/{hit['file']}"))
    row = {"cik": hit["cik"], "company": hit["name"], "ticker": hit["ticker"], "sic": sic or hit["sic"],
           "prospectus_date": hit["file_date"], "file": hit["file"], "shares": shares_offered(text), "lead_bank": lead_bank(text),
           "foreign": any(f.startswith("F-1") for f, d, _, _ in filings if d <= hit["file_date"]),
           "range": None, "range_date": None, "sources": {"prospectus": hit["file_date"]}}
    amended = [r for r in filings if r[0] in ("S-1/A", "F-1/A", "S-1", "F-1") and r[1] <= hit["file_date"]]
    for form, date, acc, doc in reversed(amended[-3:]):      # the latest registration that states a range
        rng = price_range(text_of(sec.head(f"{base}/{acc.replace('-', '')}/{doc}")))
        if rng:
            row["range"], row["range_date"] = list(rng), date
            row["sources"]["range"] = date
            break
    offer = offer_price(text, row["range"])
    row["offer_price"] = offer
    row["listing_symbol"] = listing_symbol(text)
    row["prices"] = None
    for symbol in dict.fromkeys(t for t in (row["listing_symbol"], hit["ticker"]) if t):
        chart = yahoo_chart(symbol, hit["file_date"])
        prices = first_days(chart, hit["file_date"], offer) if chart else None
        if prices:
            row["prices"], row["price_symbol"] = prices, symbol
            break
    return row


def build(years=None):
    sec = Sec()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        latest = {}
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["adsh"]] = r
        # settled: not an IPO, or an IPO already priced, or one tried under the prospectus's own symbol
        done = {a for a, r in latest.items() if (r.get("skip") and not r["skip"].startswith("error"))
                or r.get("prices") or "listing_symbol" in r}
    years = years or range(FIRST_YEAR, dt.date.today().year + 1)
    def one(hit):
        try:
            row = build_row(sec, hit)
        except Exception as e:                   # one bad filing must not stop the build
            row = {"skip": f"error: {e}"}
        return {"adsh": hit["adsh"], **(row or {"skip": "not an IPO"})}

    with OUT.open("a", encoding="utf-8") as f:
        for year in years:
            hits = [h for h in prospectus_hits(sec, year) if h["adsh"] not in done]
            print(f"{year}: {len(hits)} prospectuses to read", flush=True)
            # several filings at once: each waits on slow downloads, while the SEC pace is shared (Sec._wait)
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for n, line in enumerate(pool.map(one, hits), 1):
                    f.write(json.dumps(line) + "\n")
                    f.flush()
                    if n % 50 == 0:
                        print(f"  {n}/{len(hits)}", flush=True)


def refresh():
    """Re-read every IPO row's prospectus cover (offer, shares, bank, symbol) and its prices, keeping the range
    already found: for after a fix to those parsers, without walking every filing again."""
    sec = Sec()
    rows = load(dedupe=False)
    files = {}
    for year in sorted({r["prospectus_date"][:4] for r in rows}):
        files.update({h["adsh"]: h for h in prospectus_hits(sec, int(year))})
    print(f"refreshing {len(rows)} IPO rows", flush=True)

    def one(r):
        hit = files.get(r["adsh"])
        if not hit:
            return None
        try:
            base = f"https://www.sec.gov/Archives/edgar/data/{int(r['cik'])}"
            text = text_of(sec.head(f"{base}/{hit['adsh'].replace('-', '')}/{hit['file']}"))
            offer = offer_price(text, r.get("range"))
            new = {**r, "file": hit["file"], "offer_price": offer, "shares": shares_offered(text), "lead_bank": lead_bank(text),
                   "listing_symbol": listing_symbol(text), "prices": None}
            new.pop("price_symbol", None)
            for symbol in dict.fromkeys(t for t in (new["listing_symbol"], hit["ticker"]) if t):
                chart = yahoo_chart(symbol, r["prospectus_date"])
                prices = first_days(chart, r["prospectus_date"], offer) if chart else None
                if prices:
                    new["prices"], new["price_symbol"] = prices, symbol
                    break
            return new
        except Exception as e:
            return {**r, "refresh_error": str(e)}
    with OUT.open("a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, new in enumerate(pool.map(one, rows), 1):
            if new:
                f.write(json.dumps(new) + "\n")
                f.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(rows)}", flush=True)


INITIAL = ROOT / "bench_data" / "ipo" / "initial_ranges.json"


def add_initial_ranges():
    """For every IPO, the first price range it filed (the earliest S-1/F-1 or amendment that states one), dated:
    the price range at the start of marketing, against which the final offer's revision is usually measured.
    Cached in bench_data/ipo/initial_ranges.json; load() merges it in."""
    sec = Sec()
    done = json.loads(INITIAL.read_text(encoding="utf-8")) if INITIAL.exists() else {}
    todo = [r for r in load() if r["adsh"] not in done and r.get("range")]
    print(f"{len(todo)} IPOs to look up", flush=True)

    def one(r):
        try:
            filings, _ = filings_of(sec, r["cik"], before=r["prospectus_date"])
            regs = [f for f in filings if f[0] in ("S-1", "S-1/A", "F-1", "F-1/A") and f[1] <= r["prospectus_date"]]
            base = f"https://www.sec.gov/Archives/edgar/data/{int(r['cik'])}"
            for form, date, acc, doc in regs[:6]:               # the earliest filings, oldest first
                rng = price_range(text_of(sec.head(f"{base}/{acc.replace('-', '')}/{doc}")))
                if rng:
                    return r["adsh"], {"range": list(rng), "date": date}
            return r["adsh"], {"range": None, "date": None}
        except Exception as e:
            return r["adsh"], {"error": str(e)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, (adsh, v) in enumerate(pool.map(one, todo), 1):
            if "error" not in v:
                done[adsh] = v
            if n % 100 == 0 or n == len(todo):
                INITIAL.write_text(json.dumps(done), encoding="utf-8")
                print(f"  {n}/{len(todo)}", flush=True)


def load(path=OUT, dedupe=True):
    """The IPO rows (skipped filings left out), oldest prospectus first."""
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["adsh"]] = r              # a filing redone later replaces its earlier row
    rows = sorted([r for r in latest.values() if "skip" not in r], key=lambda r: r["prospectus_date"])
    initial = json.loads(INITIAL.read_text(encoding="utf-8")) if INITIAL.exists() else {}
    for r in rows:
        if r["adsh"] in initial:
            r["initial_range"], r["initial_range_date"] = initial[r["adsh"]]["range"], initial[r["adsh"]]["date"]
    if not dedupe:
        return rows
    first = {}                                 # a company's IPO is its first final prospectus; a second one the same
    for r in rows:                             # day (another share class, a corrected filing) is not another IPO
        first.setdefault(r["cik"], r)
    return sorted(first.values(), key=lambda r: r["prospectus_date"])


def summary():
    rows = load()
    by = {}
    for r in rows:
        y = r["prospectus_date"][:4]
        b = by.setdefault(y, [0, 0, 0])
        b[0] += 1
        b[1] += bool(r.get("prices"))
        b[2] += bool(r.get("range"))
    print("year  IPOs  with prices  with a range")
    for y, (n, p, g) in sorted(by.items()):
        print(f"{y}  {n:>4}  {p:>4} ({100 * p / n:.0f}%)  {g:>4} ({100 * g / n:.0f}%)")


if __name__ == "__main__":
    (summary() if "--summary" in sys.argv else refresh() if "--refresh" in sys.argv
     else add_initial_ranges() if "--initial-ranges" in sys.argv else build())
