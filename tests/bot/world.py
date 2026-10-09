"""A synthetic market for offline tests: every source the bot reads, answered from one seeded, made-up world in
the sources' own formats (Yahoo's chart JSON, Cboe's JSON and CSV, the Treasury's CSV, FRED's CSV, Nasdaq
Trader's pipe files, Wikipedia's parse API and page history (monthly revisions, the older ones in the old
table layout), EDGAR's form indexes, full-text search, submissions and filings).

Nothing here is real data. It exists so the whole update can run with no network: prices only up to the
world's clock, a 2-for-1 split, a special dividend, a delisting, a missing day, a source disagreement, and an
IPO calendar with range revisions, withdrawals, postponements, a follow-on, a SPAC and a foreign filer.

    w = World(now=...)              # then Http(cfg, opener=w.opener, sleep=lambda s: None)
    w.fail["query1.finance.yahoo.com"] = 500      # a source goes down
    w.schema["yahoo"] = True                      # a source changes its format
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.message import Message
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from bot import markets

START = dt.date(2023, 1, 3)
HORIZON = dt.date(2027, 6, 30)
SPLIT_DAY = dt.date(2025, 6, 2)  # SPLT splits 2-for-1 before this session
DIVIDEND_DAY = dt.date(2025, 3, 3)  # DIVD goes ex a special dividend of 60% of its price
DELIST_DAY = dt.date(2025, 9, 30)  # GONE's last session
HOLE_DAY = dt.date(2025, 2, 11)  # Yahoo has no bar for HOLE this day (Cboe does)
MISMATCH_DAYS = 3  # Cboe's AAA close is 1% off this many sessions before the newest
SYMBOLS = ("AAA", "BBB", "SPLT", "DIVD", "GONE", "HOLE")
OLD_LAYOUT = dt.date(2024, 1, 1)  # revisions before this use the page's old column names and no table id
REV_BASE = 5000  # the revision id of the first month's revision


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _days(a: dt.date, b: dt.date) -> List[dt.date]:
    return markets.trading_days(a, b)


@dataclass
class Deal:
    k: int
    cik: int
    company: str
    symbol: str
    form: str  # S-1 or F-1
    filed: dt.date
    amend: dt.date
    revise: Optional[dt.date]
    effect: Optional[dt.date]
    priced: Optional[dt.date]
    trade: Optional[dt.date]
    withdrawn: Optional[dt.date]
    low: float
    high: float
    rev: float
    offer: Optional[float]
    ret: Optional[float]
    shares: int
    bank: str
    follow_on: bool = False
    filings: List[Tuple[str, dt.date, str]] = field(default_factory=list)  # (form, date, accession)


class World:
    def __init__(self, now: dt.datetime, seed: int = 1, symbols: Tuple[str, ...] = SYMBOLS, deals: int = 140):
        self.now = now
        self.seed = seed
        self.symbols = symbols
        self.fail: Dict[str, Union[int, str]] = {}  # host (or "host/path-fragment") -> HTTP code or "timeout"
        self.schema: Dict[str, bool] = {}  # source name -> answer in a changed format
        self.broken_revs: set = set()  # Wikipedia revision ids whose table cannot be read
        self.calls: List[str] = []
        self.days = _days(START, HORIZON)
        self.raw = {s: self._path(s) for s in (*symbols, "SPX")}
        self.vix = self._series("VIX", 18.0, 0.05)
        self.ust = {
            "UST3M": self._series("UST3M", 4.5, 0.01),
            "UST2Y": self._series("UST2Y", 4.2, 0.01),
            "UST10Y": self._series("UST10Y", 4.0, 0.008),
        }
        self.deals = self._deals(deals)

    # ------------------------------------------------------------------ the world's facts

    def _path(self, sym: str) -> Dict[dt.date, Tuple[float, float, float, float, float]]:
        rng = np.random.default_rng(_seed(self.seed, sym))
        p = 50.0 + rng.uniform(0, 100)
        out = {}
        for d in self.days:
            if sym == "GONE" and d > DELIST_DAY:
                break
            p *= math.exp(rng.normal(0.0003, 0.015))
            if sym == "SPLT" and d == SPLIT_DAY:
                p /= 2  # the raw price halves at the split
            if sym == "DIVD" and d == DIVIDEND_DAY:
                p *= 0.4
            o = p * math.exp(rng.normal(0, 0.004))
            hi, lo = max(o, p) * 1.006, min(o, p) * 0.994
            out[d] = (round(o, 4), round(hi, 4), round(lo, 4), round(p, 4), float(rng.integers(1_000_000, 9_000_000)))
        return out

    def _series(self, name: str, level: float, vol: float) -> Dict[dt.date, float]:
        rng = np.random.default_rng(_seed(self.seed, name))
        x, out = level, {}
        for d in self.days:
            x = max(0.05, x * math.exp(rng.normal(0, vol)))
            out[d] = round(x, 2)
        return out

    def _deals(self, n: int) -> List[Deal]:
        rng = np.random.default_rng(_seed(self.seed, "ipo"))
        banks = ("Goldman Sachs", "Morgan Stanley", "Jefferies", "Aegis Capital", "Cantor")
        out = []
        for k in range(n):
            t0 = self.days[5 + 6 * k]
            idx = self.days.index(t0)
            day = lambda j, idx=idx: self.days[idx + j]  # noqa: E731
            mid = float(rng.uniform(6, 22))
            low, high = round(mid * 0.9, 0), round(mid * 0.9, 0) + 2
            rev = float(rng.choice([-0.2, 0.0, 0.0, 0.15, 0.3]))
            withdrawn = k % 11 == 5
            postponed = k % 13 == 7
            spac = k % 19 == 4
            foreign = k % 23 == 0
            name = f"Synth {'Acquisition Corp' if spac else 'Holdings Inc.'} {k}"
            heat = 0.1 * math.sin(k / 9)
            ret = 0.12 + 0.9 * rev + heat + float(rng.normal(0, 0.12))
            mid_final = (low + high) / 2 * (1 + rev)
            offer = round(mid_final * (1 + rev / 3), 2)
            d = Deal(
                k=k,
                cik=9_100_000 + k,
                company=name,
                symbol=f"Q{k:03d}",
                form="F-1" if foreign else "S-1",
                filed=t0,
                amend=day(15),
                revise=day(25) if rev else None,
                effect=None if withdrawn else day(30),
                priced=None if (withdrawn or postponed) else day(31),
                trade=None if (withdrawn or postponed) else day(31),
                withdrawn=day(28) if withdrawn else None,
                low=low,
                high=high,
                rev=rev,
                offer=None if (withdrawn or postponed) else offer,
                ret=None if (withdrawn or postponed) else ret,
                shares=int(rng.integers(2, 30)) * 1_000_000,
                bank=banks[k % len(banks)],
                follow_on=k % 17 == 3,
            )
            seq = 0

            def acc(seq: int, k: int = k, t0: dt.date = t0) -> str:
                return f"{9_100_000_000 + k:010d}-{t0.year % 100:02d}-{seq:06d}"

            d.filings.append((d.form, d.filed, acc(seq := seq + 1)))
            d.filings.append((d.form + "/A", d.amend, acc(seq := seq + 1)))
            if d.revise:
                d.filings.append((d.form + "/A", d.revise, acc(seq := seq + 1)))
            if d.withdrawn:
                d.filings.append(("RW", d.withdrawn, acc(seq := seq + 1)))
            if d.effect:
                d.filings.append(("EFFECT", d.effect, f"9999999995-{d.effect.year % 100:02d}-{100000 + k:06d}"))
            if d.priced:
                d.filings.append(("424B4", d.priced, acc(seq := seq + 1)))
            out.append(d)
        return out

    def through(self) -> dt.date:
        return markets.last_closed_session(self.now)

    def published(self, d: dt.date, delay_h: float = 0.25) -> bool:
        return markets.close_utc(d) + dt.timedelta(hours=delay_h) <= self.now

    def bars(self, sym: str, adjusted: bool) -> List[Tuple[dt.date, Tuple[float, ...]]]:
        """Bars known now: split-adjusted (as Yahoo serves them) or raw (as Cboe does)."""
        out = []
        split_known = self.now >= markets.open_utc(SPLIT_DAY)
        for d, bar in self.raw[sym].items():
            if not self.published(d):
                break
            if adjusted and sym == "SPLT" and split_known and d < SPLIT_DAY:
                bar = (*(round(v / 2, 4) for v in bar[:4]), bar[4] * 2)
            out.append((d, bar))
        return out

    def ipo_bars(self, deal: Deal) -> List[Tuple[dt.date, Tuple[float, ...]]]:
        if not deal.trade or not deal.offer or deal.ret is None:
            return []
        rng = np.random.default_rng(_seed(self.seed, deal.symbol))
        p = deal.offer * (1 + deal.ret)
        out = []
        for d in self.days[self.days.index(deal.trade) :]:
            if not self.published(d):
                break
            o = deal.offer * (1 + deal.ret / 2) if d == deal.trade else p
            out.append((d, (round(o, 4), round(max(o, p) * 1.01, 4), round(min(o, p) * 0.99, 4), round(p, 4), 1e6)))
            p *= math.exp(rng.normal(0, 0.03))
        return out

    # ------------------------------------------------------------------ the opener

    def opener(self, req: urllib.request.Request, timeout: float) -> bytes:
        url = req.full_url
        self.calls.append(url)
        parts = urllib.parse.urlsplit(url)
        for key, what in self.fail.items():
            if key in url:
                if what == "timeout":
                    raise TimeoutError("timed out")
                if what == "reset":
                    raise ConnectionResetError("connection reset")
                raise self._http(url, int(what))
        q = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        host, path = parts.netloc, parts.path
        if host == "query1.finance.yahoo.com":
            return self._yahoo(urllib.parse.unquote(path.rsplit("/", 1)[1]), int(q["period1"]), int(q["period2"]))
        if host == "cdn.cboe.com" and path.endswith("VIX_History.csv"):
            return self._vix_csv()
        if host == "cdn.cboe.com":
            return self._cboe(path.rsplit("/", 1)[1].removesuffix(".json"))
        if host == "home.treasury.gov":
            return self._treasury(int(path.split("/")[-2]))
        if host == "fred.stlouisfed.org":
            return self._fred(q["id"])
        if host == "www.nasdaqtrader.com":
            return self._listing(path.rsplit("/", 1)[1])
        if host == "en.wikipedia.org" and q.get("action") == "query":
            return self._revisions(q["rvstart"])
        if host == "en.wikipedia.org" and "oldid" in q:
            return self._old_wiki(int(q["oldid"]))
        if host == "en.wikipedia.org":
            return self._wiki()
        if host == "efts.sec.gov":
            return self._efts(
                dt.date.fromisoformat(q["startdt"]), dt.date.fromisoformat(q["enddt"]), int(q.get("from", 0))
            )
        if host == "data.sec.gov":
            return self._submissions(int(path.rsplit("CIK", 1)[1].removesuffix(".json")))
        if host == "www.sec.gov" and "/daily-index/" in path:
            return self._daily_index(dt.datetime.strptime(path.rsplit(".", 2)[-2], "%Y%m%d").date(), url)
        if host == "www.sec.gov" and "/full-index/" in path:
            y, qtr = int(path.split("/")[-3]), int(path.split("/")[-2][3:])
            return self._full_index(y, qtr)
        if host == "www.sec.gov" and "/edgar/data/" in path:
            return self._document(path.rsplit("/", 1)[1].removesuffix(".txt"), url)
        raise self._http(url, 404)

    @staticmethod
    def _http(url: str, code: int) -> urllib.error.HTTPError:
        hdrs = Message()
        if code == 429:
            hdrs["Retry-After"] = "2"
        return urllib.error.HTTPError(url, code, "synthetic", hdrs, io.BytesIO(b""))

    # ------------------------------------------------------------------ sources

    def _yahoo(self, ticker: str, p1: int, p2: int) -> bytes:
        if self.schema.get("yahoo"):
            return json.dumps(
                {"chart": {"result": [{"meta": {}, "timestamp": [1], "indicators": {"quote": [{"c": [1]}]}}]}}
            ).encode()
        sym = {"^GSPC": "SPX"}.get(ticker, ticker.replace("-", "."))
        deal = next((d for d in self.deals if d.symbol == sym), None)
        if sym in self.raw:
            series = self.bars(sym, adjusted=True)
        elif deal:
            series = self.ipo_bars(deal)
        else:
            series = []
        if not series or (sym == "GONE" and self.through() > DELIST_DAY + dt.timedelta(days=30)):
            return json.dumps(
                {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}
            ).encode()
        lo, hi = dt.datetime.fromtimestamp(p1, dt.UTC).date(), dt.datetime.fromtimestamp(p2, dt.UTC).date()
        rows = [(d, b) for d, b in series if lo <= d < hi and not (sym == "HOLE" and d == HOLE_DAY)]
        if self.schema.get("duplicate") and rows:
            rows.append(rows[-1])  # the same session twice, as Yahoo sometimes sends during the day
        stamps = [int((markets.open_utc(d)).timestamp()) for d, _ in rows]
        offset = markets.eastern_offset_hours(self.now) * 3600
        quote = {k: [b[i] for _, b in rows] for i, k in enumerate(("open", "high", "low", "close", "volume"))}
        events: Dict[str, Dict[str, object]] = {}
        if sym == "SPLT" and lo <= SPLIT_DAY < hi and self.now >= markets.open_utc(SPLIT_DAY):
            t = int(markets.open_utc(SPLIT_DAY).timestamp())
            events["splits"] = {str(t): {"date": t, "numerator": 2.0, "denominator": 1.0, "splitRatio": "2:1"}}
        if sym == "DIVD" and lo <= DIVIDEND_DAY < hi and self.now >= markets.open_utc(DIVIDEND_DAY):
            t = int(markets.open_utc(DIVIDEND_DAY).timestamp())
            prev = self.raw["DIVD"][markets.previous_trading_day(DIVIDEND_DAY)][3]
            events["dividends"] = {str(t): {"amount": round(prev * 0.6, 4), "date": t}}
        res = {
            "meta": {"symbol": sym, "gmtoffset": offset},
            "timestamp": stamps,
            "events": events,
            "indicators": {"quote": [quote]},
        }
        if not rows:
            res.pop("timestamp")
        return json.dumps({"chart": {"result": [res], "error": None}}).encode()

    def _cboe(self, name: str) -> bytes:
        sym = {"_SPX": "SPX"}.get(name, name)
        if sym not in self.raw:
            raise self._http(name, 403)
        newest = self.through()
        bad = newest
        for _ in range(MISMATCH_DAYS):
            bad = markets.previous_trading_day(bad)
        data = []
        for d, b in self.bars(sym, adjusted=False):
            close = b[3] * (1.01 if (sym == "AAA" and d == bad) else 1.0)
            data.append(
                {
                    "date": d.isoformat(),
                    "open": b[0],
                    "high": b[1],
                    "low": b[2],
                    "close": round(close, 4),
                    "volume": b[4],
                }
            )
        return json.dumps({"timestamp": str(self.now), "data": data, "symbol": name}).encode()

    def _vix_csv(self) -> bytes:
        if self.schema.get("cboe_vix"):
            return b"Date,Value\n01/02/2024,13.2\n"
        lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
        lines += [
            f"{d:%m/%d/%Y},{v:.6f},{v:.6f},{v:.6f},{v:.6f}" for d, v in self.vix.items() if self.published(d, 0.5)
        ]
        if self.schema.get("duplicate"):
            lines.append(lines[-1])
        return ("\n".join(lines) + "\n").encode()

    def _treasury(self, year: int) -> bytes:
        cols = ["Date", "1 Mo", "3 Mo", "2 Yr", "10 Yr", "30 Yr"]
        rows = [",".join(cols)]
        for d in sorted(self.ust["UST10Y"], reverse=True):
            if d.year == year and self.published(d, 2):
                rows.append(
                    f"{d:%m/%d/%Y},{self.ust['UST3M'][d]},{self.ust['UST3M'][d]},{self.ust['UST2Y'][d]},"
                    f"{self.ust['UST10Y'][d]},{self.ust['UST10Y'][d] + 0.3:.2f}"
                )
        return ("\n".join(rows) + "\n").encode()

    def _fred(self, sid: str) -> bytes:
        name = {"DGS3MO": "UST3M", "DGS2": "UST2Y", "DGS10": "UST10Y", "VIXCLS": "VIX"}[sid]
        series = self.vix if name == "VIX" else self.ust[name]
        lines = [f"observation_date,{sid}"]
        for d, v in series.items():
            if self.published(d, 2):
                lines.append(f"{d.isoformat()},{v if d.day != 15 else '.'}")
        return ("\n".join(lines) + "\n").encode()

    def listed(self) -> List[str]:
        return [s for s in self.symbols if not (s == "GONE" and self.through() > DELIST_DAY)]

    def _listing(self, name: str) -> bytes:
        stamp = f"File Creation Time: {self.now:%m%d%Y%H:%M}|||||||"
        if name == "nasdaqlisted.txt":
            head = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
            rows = [f"{s}|{s} Corp. - Common Stock|Q|N|N|100|N|N" for s in self.listed() if s < "C"]
            rows.append("ZTEST|Test Issue|Q|Y|N|100|N|N")
        else:
            head = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"
            rows = [f"{s}|{s} Inc. Common Stock|N|{s}|N|100|N|{s}" for s in self.listed() if s >= "C"]
            rows.append("SPYX|Synthetic ETF|P|SPYX|Y|100|N|SPYX")
        return ("\r\n".join([head, *rows, stamp]) + "\r\n").encode()

    def members(self, on: Optional[dt.date] = None) -> List[str]:
        """The index on a day (default: the last finished session): GONE until its delisting, HOLE after
        HOLE_DAY (a joiner, so earlier sessions must not count it)."""
        d = on or self.through()
        listed = [s for s in self.symbols if not (s == "GONE" and d > DELIST_DAY)]
        return [s for s in listed if s != "HOLE" or d > HOLE_DAY]

    def revisions(self) -> List[Tuple[int, dt.datetime]]:
        """The page's history: a revision on the 20th of each month at noon UTC from START's month (none in
        July, so August's 1st still sees June's), up to now."""
        out = []
        y, m = START.year, START.month
        while True:
            t = dt.datetime(y, m, 20, 12, tzinfo=dt.UTC)
            if t > self.now:
                return out
            if m != 7:
                out.append((REV_BASE + (y - START.year) * 12 + m - START.month, t))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    def _revisions(self, start: str) -> bytes:
        t = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        older = [r for r in self.revisions() if r[1] <= t]
        page: Dict[str, object] = {"pageid": 1, "ns": 0, "title": "List of S&P 500 companies"}
        if older:
            revid, when = older[-1]
            page["revisions"] = [{"revid": revid, "parentid": revid - 1, "timestamp": f"{when:%Y-%m-%dT%H:%M:%SZ}"}]
        return json.dumps({"batchcomplete": True, "query": {"pages": [page]}}).encode()

    def _old_wiki(self, revid: int) -> bytes:
        when = dict(self.revisions())[revid]
        if revid in self.broken_revs:
            return json.dumps({"parse": {"text": "<p>Vandalised.</p>"}}).encode()
        return self._page(self.members(when.date()), old=when.date() < OLD_LAYOUT)

    def _wiki(self) -> bytes:
        if self.schema.get("wikipedia"):
            return json.dumps({"parse": {"text": "<p>The table moved.</p>"}}).encode()
        return self._page(self.members(), old=False)

    @staticmethod
    def _page(members: List[str], old: bool) -> bytes:
        if old:  # the 2016-2019 layout: no id, "Ticker symbol", "Date first added" with a footnote, no dates
            head = '<tr><th>Ticker symbol</th><th>Security</th><th>SEC filings</th><th>GICS Sector</th><th>Date first added<sup class="reference"><a href="#n">&#91;3&#93;</a></sup></th><th>CIK</th></tr>'
            rows = [
                f'<tr>\n<td><a href="x">{s}</a></td>\n<td>{s} &amp; Co</td>\n<td>reports</td>\n<td>Industrials</td>\n<td></td>\n<td>{1000 + i:010d}</td></tr>'
                for i, s in enumerate(members)
            ]
            table = '<table class="wikitable sortable">\n<tbody>'
        else:
            head = "<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>"
            rows = [
                f'<tr>\n<td><a href="x">{s}</a>\n</td>\n<td>{s} &amp; Co</td>\n<td>Industrials</td>\n<td>Machinery</td>\n<td>Here</td>\n<td>2010-01-0{i + 1}</td>\n<td>{1000 + i:010d}</td>\n<td>1900</td></tr>'
                for i, s in enumerate(members)
            ]
            table = '<table class="wikitable" id="constituents">\n<tbody>'
        html = table + head + "\n" + "\n".join(rows) + "</tbody></table>"
        return json.dumps({"parse": {"title": "List of S&P 500 companies", "text": html}}).encode()

    # ------------------------------------------------------------------ EDGAR

    def filings_on(self, d: dt.date) -> List[Tuple[Deal, str, str]]:
        return [(deal, form, acc) for deal in self.deals for form, day, acc in deal.filings if day == d]

    def _index_published(self, d: dt.date) -> bool:
        return markets.is_trading_day(d) and markets.close_utc(d) + dt.timedelta(hours=7) <= self.now

    def _line(self, deal: Deal, form: str, acc: str, day: dt.date, sep: str) -> str:
        date = day.isoformat() if sep else f"{day:%Y%m%d}"
        return f"{form:<17}{deal.company:<62}{deal.cik:<12}{date:<12}edgar/data/{deal.cik}/{acc}.txt"

    def _index(self, days: List[dt.date], sep: bool) -> str:
        head = (
            "Description:           Daily Index of EDGAR Dissemination Feed by Form Type\n"
            "Last Data Received:    synthetic\nComments:              webmaster@sec.gov\n\n\n\n"
            "Form Type   Company Name                                                  CIK         Date Filed  File Name\n"
            + "-" * 140
            + "\n"
        )
        lines = [
            "10-K             Some Reporting Co                                             1234        "
            f"{days[0]:%Y%m%d}    edgar/data/1234/0000001234-{days[0].year % 100}-000001.txt"
        ]
        for d in days:
            lines += [self._line(deal, form, acc, d, "-" if sep else "") for deal, form, acc in self.filings_on(d)]
        return head + "\n".join(sorted(lines)) + "\n"

    def _daily_index(self, d: dt.date, url: str) -> bytes:
        if not self._index_published(d):
            raise self._http(url, 403)  # what EDGAR answers for an index that does not exist
        if self.schema.get("sec_index"):
            return b"<html>maintenance</html>"
        return self._index([d], sep=False).encode("latin-1")

    def _full_index(self, year: int, qtr: int) -> bytes:
        first = dt.date(year, 3 * qtr - 2, 1)
        last = dt.date(year + (qtr == 4), (3 * qtr) % 12 + 1, 1) - dt.timedelta(days=1)
        days = [d for d in _days(first, min(last, self.through())) if self._index_published(d)]
        if not days:
            raise self._http("full-index", 403)
        return gzip.compress(self._index(days, sep=True).encode("latin-1"), mtime=0)

    def _efts(self, a: dt.date, b: dt.date, offset: int) -> bytes:
        hits = []
        for d in _days(a, b):
            if markets.close_utc(d) - dt.timedelta(hours=6) > self.now:
                continue
            for deal, form, acc in self.filings_on(d):
                if form == "EFFECT":
                    continue  # full-text search misses EFFECT notices, as the real one does
                hits.append(
                    {
                        "_id": f"{acc}:{deal.symbol.lower()}.htm",
                        "_source": {
                            "ciks": [f"{deal.cik:010d}"],
                            "display_names": [f"{deal.company}  (CIK {deal.cik:010d})"],
                            "form": form,
                            "file_date": d.isoformat(),
                            "adsh": acc,
                            "sics": ["2834"],
                        },
                    }
                )
        page = hits[offset : offset + 100]
        return json.dumps({"hits": {"total": {"value": len(hits)}, "hits": page}}).encode()

    def _deal(self, cik: int) -> Optional[Deal]:
        return next((d for d in self.deals if d.cik == cik), None)

    def _submissions(self, cik: int) -> bytes:
        deal = self._deal(cik)
        if deal is None:
            raise self._http(str(cik), 404)
        rows = [(f, d.isoformat(), a, f"{a}.txt") for f, d, a in deal.filings if markets.close_utc(d) <= self.now]
        if deal.follow_on:
            rows.append(("10-K", (deal.filed - dt.timedelta(days=200)).isoformat(), "0000000001-20-000001", "k.htm"))
        rows.sort(key=lambda r: r[1], reverse=True)
        doc = {
            "cik": str(cik),
            "name": deal.company,
            "sic": "6770" if "Acquisition" in deal.company else "2834",
            "tickers": [deal.symbol] if deal.trade and markets.close_utc(deal.trade) <= self.now else [],
            "filings": {
                "recent": {
                    "form": [r[0] for r in rows],
                    "filingDate": [r[1] for r in rows],
                    "accessionNumber": [r[2] for r in rows],
                    "primaryDocument": [r[3] for r in rows],
                },
                "files": [],
            },
        }
        return json.dumps(doc).encode()

    def _document(self, acc: str, url: str) -> bytes:
        for deal in self.deals:
            for form, d, a in deal.filings:
                if a == acc:
                    return self._doc_text(deal, form, d).encode("latin-1")
        raise self._http(url, 404)

    def _doc_text(self, deal: Deal, form: str, d: dt.date) -> str:
        cover = (
            f"<html><body><p>PROSPECTUS</p><p>{deal.shares:,} Shares</p><p>{deal.company}</p>"
            f"<p>Common Stock</p><p>{deal.bank} &amp; Co.</p>"
        )
        if form in ("S-1", "F-1"):
            return (
                cover + "<p>This is our initial public offering. No public market currently exists.</p></body></html>"
            )
        if form.endswith("/A"):
            low, high = deal.low, deal.high
            if deal.revise and d >= deal.revise:
                low, high = round(low * (1 + deal.rev), 2), round(high * (1 + deal.rev), 2)
            return (
                cover + f"<p>We expect the initial public offering price to be between ${low:.2f} and ${high:.2f} "
                f"per share. We intend to list on Nasdaq under the symbol &ldquo;{deal.symbol}&rdquo;.</p></body></html>"
            )
        if form == "424B4":
            return (
                cover + f"<p>The initial public offering price is ${deal.offer:.2f} per share. Our shares will trade "
                f"under the symbol &ldquo;{deal.symbol}&rdquo;.</p></body></html>"
            )
        return "<html><body>notice</body></html>"


def at(day: dt.date, hour: int = 22, minute: int = 0) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hour, minute), tzinfo=dt.UTC)
