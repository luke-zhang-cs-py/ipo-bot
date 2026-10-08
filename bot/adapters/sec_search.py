"""EDGAR full-text search (efts.sec.gov): the IPO pipeline's filings by form and date, a page of 100 at a time.

SEC data is public; its fair-access policy asks for at most 10 requests a second and a User-Agent naming the
caller with a contact address (the bot keeps to 8, and sends SEC_USER_AGENT).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from bot.adapters.base import SchemaError, conform

NAME = "sec_search"
TERMS = "SEC public data; fair access: under 10 requests a second, a User-Agent with a contact"
FORMS: Tuple[str, ...] = ("S-1", "S-1/A", "F-1", "F-1/A", "424B4", "EFFECT", "RW")
PAGE = 100


def url(start: dt.date, end: dt.date, offset: int = 0, forms: Sequence[str] = FORMS, **_: Any) -> str:
    return (
        f"https://efts.sec.gov/LATEST/search-index?forms={','.join(forms)}&dateRange=custom"
        f"&startdt={start.isoformat()}&enddt={end.isoformat()}&from={offset}"
    )


def _company(display: str) -> Tuple[str, str]:
    """("Snap Inc.", "SNAP") from "Snap Inc.  (SNAP)  (CIK 0001564408)"."""
    m = re.match(r"(.*?)\s+\(([A-Z0-9., -]+)\)\s+\(CIK", display)
    if m:
        return m.group(1).strip(), m.group(2).split(",")[0].strip()
    return display.split("(CIK")[0].strip(), ""


def parse(body: bytes, forms: Sequence[str] = FORMS, **_: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(filings, {"total": hits in the whole search}). Only the asked-for forms are kept: a search for S-1 also
    finds S-1MEF and the like."""
    doc = json.loads(body)
    hits = doc.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        raise SchemaError("sec_search: no hits list")
    rows = []
    for h in hits["hits"]:
        s = h["_source"]
        if s["form"] not in forms:
            continue
        company, ticker = _company((s.get("display_names") or [""])[0])
        rows.append(
            {
                "accession": s["adsh"],
                "cik": (s.get("ciks") or ["0"])[0].lstrip("0"),
                "form": s["form"],
                "filed": s["file_date"],
                "company": company,
                "ticker": ticker,
                "sic": (s.get("sics") or [None])[0],
                "file": h["_id"].split(":", 1)[1],
            }
        )
    total = hits.get("total", {})
    return conform(rows, "filing"), {"total": int(total.get("value", len(rows)) if isinstance(total, dict) else total)}
