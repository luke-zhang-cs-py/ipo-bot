"""The S&P 500's current members, from the constituents table of Wikipedia's "List of S&P 500 companies",
read through the MediaWiki parse API (which allows automated reads that name themselves).

The table lists today's members only. Membership history is built from the bot's own dated snapshots, and
companies that leave are kept with member = 0.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List

from bot.adapters.base import SchemaError, conform

NAME = "wikipedia"
TERMS = "MediaWiki API, CC BY-SA content; automated reads allowed with a descriptive User-Agent"
MIN_MEMBERS = 400  # fewer parsed means the table changed shape, not that the index shrank


def url(**_: Any) -> str:
    return (
        "https://en.wikipedia.org/w/api.php?action=parse&page=List_of_S%26P_500_companies&prop=text"
        "&format=json&formatversion=2"
    )


def _cells(row: str) -> List[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
    return [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip() for c in cells]


def parse(body: bytes, min_members: int = MIN_MEMBERS, **_: Any) -> List[Dict[str, Any]]:
    text = json.loads(body)["parse"]["text"]
    m = re.search(r'<table[^>]*id="constituents"[^>]*>(.*?)</table>', text, re.S)
    if not m:
        raise SchemaError("wikipedia: no constituents table")
    rows = re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)
    head = _cells(rows[0]) if rows else []
    need = ("Symbol", "Security", "GICS Sector", "Date added", "CIK")
    if not all(n in head for n in need):
        raise SchemaError(f"wikipedia: columns {head}")
    ix = {n: head.index(n) for n in need}
    out = []
    for r in rows[1:]:
        c = _cells(r)
        if len(c) < len(head):
            continue
        out.append(
            {
                "symbol": c[ix["Symbol"]],
                "name": c[ix["Security"]],
                "cik": c[ix["CIK"]].lstrip("0") or None,
                "sector": c[ix["GICS Sector"]] or None,
                "added": c[ix["Date added"]][:10] or None,
            }
        )
    if len(out) < min_members:
        raise SchemaError(f"wikipedia: only {len(out)} members parsed")
    return conform(out, "member")
