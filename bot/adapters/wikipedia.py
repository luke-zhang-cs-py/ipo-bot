"""The S&P 500's members, from the constituents table of Wikipedia's "List of S&P 500 companies", read through
the MediaWiki parse API (which allows automated reads that name themselves).

Today's table comes from the page itself; an old revision (`oldid`, found with wikipedia_revisions) gives the
table as it stood then, which is how point-in-time membership is backfilled. Older revisions name the columns
differently ("Ticker symbol", "Date first added", footnote marks) and have no table id, so the column names are
matched through ALIASES and the first sortable wikitable stands in for the constituents table.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List, Optional

from bot.adapters.base import SchemaError, conform

NAME = "wikipedia"
TERMS = "MediaWiki API, CC BY-SA content; automated reads allowed with a descriptive User-Agent"
MIN_MEMBERS = 400  # fewer parsed means the table changed shape, not that the index shrank
# field -> the header texts it has gone by over the page's history
ALIASES = {
    "symbol": ("Symbol", "Ticker symbol", "Ticker"),
    "name": ("Security", "Company"),
    "sector": ("GICS Sector", "GICS sector"),
    "added": ("Date added", "Date first added"),
    "cik": ("CIK",),
}
REQUIRED = ("symbol", "name")


def url(oldid: Optional[int] = None, **_: Any) -> str:
    page = f"oldid={oldid}" if oldid is not None else "page=List_of_S%26P_500_companies"
    return f"https://en.wikipedia.org/w/api.php?action=parse&{page}&prop=text&format=json&formatversion=2"


def _cells(row: str) -> List[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
    return [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip() for c in cells]


def _table(text: str) -> str:
    """The constituents table: by its id, else (in older revisions) the first sortable wikitable."""
    m = re.search(r'<table[^>]*id="constituents"[^>]*>(.*?)</table>', text, re.S) or re.search(
        r'<table[^>]*class="wikitable sortable"[^>]*>(.*?)</table>', text, re.S
    )
    if not m:
        raise SchemaError("wikipedia: no constituents table")
    return m.group(1)


def parse(body: bytes, min_members: int = MIN_MEMBERS, **_: Any) -> List[Dict[str, Any]]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", _table(json.loads(body)["parse"]["text"]), re.S)
    head = [re.sub(r"\[\d+\]", "", h).strip() for h in _cells(rows[0])] if rows else []
    ix = {f: next((head.index(n) for n in names if n in head), None) for f, names in ALIASES.items()}
    if any(ix[f] is None for f in REQUIRED):
        raise SchemaError(f"wikipedia: columns {head}")
    out = []
    for r in rows[1:]:
        c = _cells(r)
        if len(c) < len(head):
            continue
        cell = {f: (c[i] if i is not None else "") for f, i in ix.items()}
        added = cell["added"][:10]
        out.append(
            {
                "symbol": cell["symbol"],
                "name": cell["name"],
                "cik": cell["cik"].lstrip("0") or None,
                "sector": cell["sector"] or None,
                "added": added if re.fullmatch(r"\d{4}-\d{2}-\d{2}", added) else None,
            }
        )
    if len(out) < min_members:
        raise SchemaError(f"wikipedia: only {len(out)} members parsed")
    return conform(out, "member")
