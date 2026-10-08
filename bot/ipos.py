"""IPO deals, assembled from stored filings as of a moment, and the checks on them.

A deal is one company's path to listing: registration (S-1 or F-1), amendments that set and revise the price
range, the SEC's notice that the registration is effective (EFFECT), the final prospectus with the offer price
(424B4), and the first trade. A deal can stop anywhere: withdrawn (RW), or postponed (effective or priced but
not trading weeks later). Every deal is kept whatever its outcome, so nothing about the IPO market's history
is lost to survivorship.

Only a company's first-time IPO counts: one that filed annual or quarterly reports before registering is
doing a follow-on, and blank-check companies (SPACs, SIC 6770) are not operating businesses.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bot import markets

REGISTRATIONS = ("S-1", "S-1/A", "F-1", "F-1/A")
POSTPONED_AFTER = 15  # trading days from effectiveness with no first trade
SPAC = re.compile(r"(?i)acquisition corp|\bspac\b|blank check|capital corp\.? [ivx]+\b")
# exchange-traded funds and commodity trusts register on S-1 too, but are not companies going public
FUND = re.compile(r"(?i)\betf\b|\bfund\b|\btrust\b")
FUND_SIC = ("6221",)
TOP_BANKS = ("Goldman Sachs", "Morgan Stanley", "J.P. Morgan", "BofA Securities", "Citigroup")


@dataclass
class Deal:
    cik: str
    company: str
    foreign: bool
    registered: str  # first registration's filing date
    ranges: List[Tuple[str, float, float]] = field(default_factory=list)  # (filed, low, high), oldest first
    shares: Optional[float] = None
    lead: Optional[str] = None
    symbol: Optional[str] = None
    effective: Optional[str] = None
    priced: Optional[str] = None  # the 424B4's filing date
    offer: Optional[float] = None
    withdrawn: Optional[str] = None
    first_trade: Optional[Dict[str, Any]] = None  # {date, open, close, symbol, source}
    ipo: bool = True  # False: a follow-on or a SPAC
    why_not: Optional[str] = None

    @property
    def initial_range(self) -> Optional[Tuple[float, float]]:
        return (self.ranges[0][1], self.ranges[0][2]) if self.ranges else None

    @property
    def latest_range(self) -> Optional[Tuple[float, float]]:
        return (self.ranges[-1][1], self.ranges[-1][2]) if self.ranges else None

    def status(self, today: dt.date) -> str:
        """withdrawn, trading, postponed, priced, effective or on_file."""
        if self.withdrawn:
            return "withdrawn"
        if self.first_trade:
            return "trading"
        since = self.effective or self.priced
        if since and len(markets.trading_days(dt.date.fromisoformat(since), today)) > POSTPONED_AFTER:
            return "postponed"
        return "priced" if self.priced else "effective" if self.effective else "on_file"

    @property
    def first_day_return(self) -> Optional[float]:
        if self.first_trade and self.offer:
            return float(self.first_trade["close"]) / self.offer - 1
        return None


def build(
    filings: Sequence[Mapping[str, Any]],
    documents: Mapping[str, str],
    companies: Mapping[str, Mapping[str, Any]],
    first_trades: Mapping[str, Mapping[str, Any]],
) -> List[Deal]:
    """Deals from rows already filtered to a moment (store.asof). documents: accession -> parsed JSON."""
    by_cik: Dict[str, List[Mapping[str, Any]]] = {}
    for f in sorted(filings, key=lambda f: (f["filed"], f["accession"])):
        by_cik.setdefault(str(f["cik"]), []).append(f)
    deals = []
    for cik, fs in by_cik.items():
        regs = [f for f in fs if f["form"] in REGISTRATIONS]
        if not regs:
            continue  # an EFFECT or 424B4 for some other kind of offering
        d = Deal(
            cik=cik,
            company=regs[-1]["company"],
            foreign=any(f["form"].startswith("F-") for f in regs),
            registered=regs[0]["filed"],
        )
        for f in fs:
            doc = json.loads(documents[f["accession"]]) if f["accession"] in documents else {}
            if f["form"] in REGISTRATIONS:
                if doc.get("range"):
                    d.ranges.append((f["filed"], float(doc["range"][0]), float(doc["range"][1])))
                d.shares = doc.get("shares") or d.shares
                d.lead = doc.get("lead") or d.lead
                d.symbol = doc.get("symbol") or d.symbol
            elif f["form"] == "EFFECT" and f["filed"] >= d.registered and not d.effective:
                d.effective = f["filed"]
            elif f["form"] == "424B4" and f["filed"] >= d.registered and not d.priced:
                d.priced = f["filed"]
                d.offer = doc.get("offer")
                d.shares = doc.get("shares") or d.shares
                d.lead = doc.get("lead") or d.lead
                d.symbol = doc.get("symbol") or d.symbol
            elif f["form"] == "RW" and f["filed"] >= d.registered:
                d.withdrawn = f["filed"]
        co = companies.get(cik)
        sic = str((co or {}).get("sic") or regs[-1].get("sic") or "")
        if sic == "6770" or SPAC.search(d.company):
            d.ipo, d.why_not = False, "blank-check company"
        elif sic in FUND_SIC or FUND.search(d.company):
            d.ipo, d.why_not = False, "a fund or trust, not an operating company"
        elif co and co.get("first_report") and co["first_report"] < d.registered:
            d.ipo, d.why_not = False, "already reporting (a follow-on or an uplisting)"
        if not d.symbol and co and co.get("tickers"):
            tickers = json.loads(co["tickers"]) if isinstance(co["tickers"], str) else co["tickers"]
            d.symbol = tickers[0] if tickers else None
        if cik in first_trades:
            d.first_trade = dict(first_trades[cik])
            if d.withdrawn and d.withdrawn > d.first_trade["date"]:
                d.withdrawn = None  # an RW after listing withdraws some other registration
        deals.append(d)
    return deals


# ---------------------------------------------------------------------------- checks


def check(deals: Sequence[Deal]) -> List[Dict[str, Any]]:
    """The IPO checks. Each problem is {check, cik, company, detail}:

    offer_vs_range       the offer is far outside the marketed range (below half its low or above twice its high):
                         a parsing error far more often than a real repricing
    pricing_vs_424b4     the first trade is not within two trading days of the final prospectus
    trade_before_pricing the first trade is before the registration became effective or before the prospectus
    missing_offer        a deal trades but its offer price was not read (its first-day return is unknown)
    """
    out = []
    for d in deals:
        if not d.ipo:
            continue
        rng = d.latest_range
        if d.offer is not None and rng and not (0.5 * rng[0] <= d.offer <= 2 * rng[1]):
            out.append(_issue("offer_vs_range", d, f"offer {d.offer} against a {rng[0]}-{rng[1]} range"))
        ft = d.first_trade
        if not ft:
            continue
        day = dt.date.fromisoformat(ft["date"])
        if d.priced:
            filed = dt.date.fromisoformat(d.priced)
            lo, hi = sorted((day, filed))
            gap = len(markets.trading_days(lo, hi)) - 1
            if gap > 2:
                out.append(_issue("pricing_vs_424b4", d, f"first trade {ft['date']}, 424B4 filed {d.priced}"))
        if (d.effective and ft["date"] < d.effective) or ft["date"] < d.registered:
            out.append(_issue("trade_before_pricing", d, f"first trade {ft['date']}, effective {d.effective}"))
        if d.offer is None:
            out.append(_issue("missing_offer", d, "trading, but no offer price was read"))
    return out


def _issue(name: str, d: Deal, detail: str) -> Dict[str, Any]:
    return {"check": name, "cik": d.cik, "company": d.company, "detail": detail}


def counts(deals: Sequence[Deal], today: dt.date) -> Dict[str, int]:
    """How many first-time IPO deals are in each status (withdrawn and postponed ones included)."""
    out: Dict[str, int] = {}
    for d in deals:
        if d.ipo:
            s = d.status(today)
            out[s] = out.get(s, 0) + 1
    return out
