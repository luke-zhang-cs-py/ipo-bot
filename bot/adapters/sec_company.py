"""A company's EDGAR submissions (data.sec.gov): its SIC code, tickers, and the dates it filed annual or
quarterly reports, which tell a first-time IPO from a follow-on by a company that already reports."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bot.adapters.base import SchemaError

NAME = "sec_company"
TERMS = "SEC public data; fair access: under 10 requests a second, a User-Agent with a contact"
REPORTS = ("10-K", "10-Q", "20-F", "40-F")


def url(cik: str, page: Optional[str] = None, **_: Any) -> str:
    if page:
        return f"https://data.sec.gov/submissions/{page}"
    return f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"


def parse(body: bytes, cik: str, page: Optional[str] = None, **_: Any) -> Dict[str, Any]:
    """{"cik", "name", "sic", "tickers", "first_report", "oldest", "pages"}: first_report is the earliest annual or
    quarterly report among these filings (None if none), oldest the earliest filing listed, pages the older
    pages not read yet."""
    doc = json.loads(body)
    recent = doc if page else doc.get("filings", {}).get("recent")
    if not isinstance(recent, dict) or "form" not in recent or "filingDate" not in recent:
        raise SchemaError("sec_company: no filings")
    forms: List[str] = recent["form"]
    dates: List[str] = recent["filingDate"]
    reports = [d for f, d in zip(forms, dates) if f in REPORTS]
    out: Dict[str, Any] = {
        "cik": str(int(cik)),
        "first_report": min(reports) if reports else None,
        "oldest": min(dates) if dates else None,
    }
    if not page:
        out.update(
            name=doc.get("name", ""),
            sic=str(doc.get("sic") or ""),
            tickers=list(doc.get("tickers") or []),
            pages=[f["name"] for f in doc.get("filings", {}).get("files", [])],
        )
    return out
