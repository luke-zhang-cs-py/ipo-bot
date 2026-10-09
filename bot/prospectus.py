"""Parsers for SEC registration statements and prospectuses: pure functions on the filing's text.

Shared by the keyless bot (bot.adapters.sec) and the research dataset builder (evaluation/ipo_data.py), so the
two read filings the same way.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import Optional, Sequence, Tuple, Union

# Bookrunners, most-named first; the first one found on the cover is taken as the lead.
BANKS: Tuple[str, ...] = (
    "Goldman Sachs",
    "Morgan Stanley",
    "J.P. Morgan",
    "BofA Securities",
    "Merrill Lynch",
    "Citigroup",
    "Credit Suisse",
    "Barclays",
    "Deutsche Bank",
    "UBS",
    "Jefferies",
    "Wells Fargo",
    "RBC Capital",
    "Evercore",
    "Cowen",
    "TD Cowen",
    "Piper Sandler",
    "Piper Jaffray",
    "Stifel",
    "Raymond James",
    "William Blair",
    "Leerink",
    "SVB Leerink",
    "Needham",
    "Oppenheimer",
    "BMO Capital",
    "KeyBanc",
    "Baird",
    "Cantor",
    "Guggenheim",
    "Mizuho",
    "Nomura",
    "HSBC",
    "Truist",
    "SunTrust",
    "Canaccord",
    "Roth Capital",
    "Lake Street",
    "Craig-Hallum",
    "B. Riley",
    "Ladenburg",
    "Maxim Group",
    "Aegis Capital",
    "EF Hutton",
    "Boustead",
    "ThinkEquity",
    "Univest",
    "Network 1",
    "Benchmark",
    "Kingswood",
    "Joseph Gunnar",
    "Spartan Capital",
    "Dawson James",
    "Prime Number",
    "US Tiger",
    "Tiger Brokers",
    "AMTD",
    "Alexander Capital",
    "WestPark",
    "Network 1 Financial",
    "D. Boral",
    "Revere Securities",
    "R.F. Lafferty",
)
RENAMED = {
    "Merrill Lynch": "BofA Securities",
    "Piper Jaffray": "Piper Sandler",
    "SunTrust": "Truist",
    "Leerink": "SVB Leerink",
    "Cowen": "TD Cowen",
    "Network 1 Financial": "Network 1",
}
PRICE_MIN, PRICE_MAX = 0.5, 500.0  # a per-share price outside this is a parsing mistake, not an IPO
SHARES_MIN, SHARES_MAX = 50_000, 5_000_000_000

Range = Tuple[float, float]

# A dollar amount: it starts with a digit, so a blank template's "$ ," (an S-1 filed before its range) is no amount.
_AMOUNT = r"(\d[\d,]*(?:\.\d+)?)"
_MONEY = r"\$\s?" + _AMOUNT
# "offering", also when a filing's "ff" ligature came through as one character or was lost ("o_ering" in Lyft's).
_OFFERING = "o(?:ff|_|\ufb00)ering"
_STATED = (
    rf"between {_MONEY} and {_MONEY} per (?:share|ADS|American)",
    rf"{_OFFERING} price[^.$]{{0,80}}between {_MONEY} and {_MONEY}",
)
_ASSUMED = rf"assum(?:ed|ing an?) (?:initial public )?{_OFFERING} price (?:per (?:share|ADS) )?of {_MONEY}"
_MIDPOINT = (
    rf"{_MONEY}(?: per (?:share|ADS))?\s?,\s?which is the mid-?point of the (?:estimated )?(?:{_OFFERING} )?price range"
)
_FEE_MAX = rf"Maximum (?:Aggregate )?{_OFFERING} Price Per (?:Share|ADS)"


def text_of(raw: Union[bytes, str]) -> str:
    """Plain text from a filing's HTML: scripts and styles dropped, tags removed, entities read, whitespace collapsed."""
    t = raw.decode("latin-1") if isinstance(raw, bytes) else raw
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _plausible(p: float) -> bool:
    return PRICE_MIN <= p <= PRICE_MAX


def offer_price(text: str, rng: Optional[Sequence[float]] = None) -> Optional[float]:
    """The final offer price per share or ADS, or None. Candidates come strongest pattern first; with the marketed
    range known, the first within half its low to twice its high wins, so a per-share fee or discount on the cover
    ("$1.50 per share") is not taken for the price."""
    candidates = []
    for pat in (
        r"initial public offering price (?:is|of|per (?:share|ADS)[^$]{0,40}?)\s*" + _MONEY,
        r"public offering price[^$]{0,60}\$\s?([\d,]+\.\d\d) per (?:share|ADS|American)",
        r"\$\s?([\d,]+\.\d\d) per (?:share|ADS)",
    ):
        candidates += [_num(m.group(1)) for m in re.finditer(pat, text[:120000], re.I)]
    candidates = [c for c in candidates if _plausible(c)]
    if rng:
        fits = [c for c in candidates if rng[0] * 0.5 <= c <= rng[1] * 2]
        return fits[0] if fits else None
    return candidates[0] if candidates else None


def range_reading(text: str) -> Optional[Tuple[Range, str]]:
    """The marketed price range (low, high) from a preliminary prospectus and where it came from, or None:

    - "stated": the range the filing states ("between $14.00 and $16.00 per share");
    - "midpoint": the stated range disagreed with the midpoint the body assumes ("an assumed ... price of $105.00
      per share, which is the midpoint of the price range"), as when an amendment moves the range but its cover
      still shows the old one (Snowflake's second S-1/A: $75-85 on the cover, $105 assumed, $110 in the fee
      table), so the range is rebuilt around that midpoint: up to the fee table's maximum when it has one above
      the midpoint, else with the stated range's width;
    - "assumed": no range, a single assumed price (common for small fixed-price deals), returned as (p, p).
      It is a reference price, not a range the offer can land above or below."""
    for pat in _STATED:
        m = re.search(pat, text, re.I)
        if m:
            break
    if m and PRICE_MIN <= _num(m.group(1)) <= _num(m.group(2)) <= PRICE_MAX:
        lo, hi = _num(m.group(1)), _num(m.group(2))
        mid = _stated_midpoint(text)
        if mid is None or abs((lo + hi) / 2 - mid) <= max(0.011, 0.005 * mid):
            return (lo, hi), "stated"
        top = _fee_table_max(text, mid)
        half = top - mid if top else (hi - lo) / 2
        if _plausible(mid - half) and _plausible(mid + half):
            return (round(mid - half, 4), round(mid + half, 4)), "midpoint"
        return (mid, mid), "assumed"
    m = re.search(_ASSUMED, text, re.I)
    if m and _plausible(_num(m.group(1))):
        return (_num(m.group(1)), _num(m.group(1))), "assumed"
    return None


def _stated_midpoint(text: str) -> Optional[float]:
    """The price the body assumes as "the midpoint of the price range", the most often stated one, or None."""
    if not re.search("mid-?point", text, re.I):  # a quick look first: most filings never say it
        return None
    mids = [v for v in (_num(m.group(1)) for m in re.finditer(_MIDPOINT, text, re.I)) if _plausible(v)]
    return Counter(mids).most_common(1)[0][0] if mids else None


def _fee_table_max(text: str, mid: float) -> Optional[float]:
    """The registration fee table's maximum price per share, when it is above the stated midpoint (by at most half
    again): the top of the range the midpoint belongs to. None when there is no such table or amount."""
    m = re.search(_FEE_MAX, text, re.I)
    if not m:
        return None
    for v in re.finditer(_MONEY, text[m.end() : m.end() + 600]):
        if mid < _num(v.group(1)) <= 1.5 * mid:
            return _num(v.group(1))
    return None


def price_range(text: str) -> Optional[Range]:
    """The marketed price range (low, high) from a preliminary prospectus, or None (see range_reading). A single
    assumed price is returned as (p, p)."""
    reading = range_reading(text)
    return reading[0] if reading else None


def shares_offered(text: str) -> Optional[float]:
    """The number of shares or ADSs offered on the cover, or None."""
    m = re.search(
        r"(\d[\d,]{4,}) (?:Shares|shares of (?:Class [A-Z] )?(?:common|ordinary)|American Depositary Shares|ADSs|Ordinary Shares)",
        text[:20000],
    )
    if m:
        n = _num(m.group(1))
        if SHARES_MIN <= n <= SHARES_MAX:
            return n
    return None


def listing_symbol(text: str) -> Optional[str]:
    """The ticker the shares listed under, from the prospectus ("under the symbol “SNAP”"), or None."""
    m = re.search(r"under the (?:trading )?symbols?\s*[“\"'‘]\s*([A-Z][A-Z0-9.\-]{0,6})\s*[”\"'’]", text[:80000])
    return m.group(1).rstrip(".-") if m else None  # "SNAP." at a sentence end is SNAP


def lead_bank(text: str) -> Optional[str]:
    """The first bookrunner named on the cover (the lead, by convention), or None."""
    head = text[:60000]
    found = [(head.find(b), b) for b in BANKS if head.find(b) >= 0]
    if not found:
        return None
    name = min(found)[1]
    return RENAMED.get(name, name)
