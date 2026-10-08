"""EDGAR's form indexes: the complete list of what was disseminated, by form type. The primary source for the
IPO pipeline's filings (full-text search, the fallback, misses EFFECT notices and some prospectuses).

Daily files (daily-index/YYYY/QTRn/form.YYYYMMDD.idx) for recent days, where a day with no file (a weekend, a
federal holiday) answers 404; quarterly files (full-index/YYYY/QTRn/form.gz, about 5 MB) for the backfill.
"""

from __future__ import annotations

import datetime as dt
import gzip
import re
from typing import Any, Dict, List, Optional, Sequence

from bot import markets
from bot.adapters.base import SchemaError, conform
from bot.adapters.sec_search import FORMS

NAME = "sec_index"
TERMS = "SEC public data; fair access: under 10 requests a second, a User-Agent with a contact"
LINE = re.compile(r"^(\S+(?: \S+)*?)\s{2,}(.+?)\s{2,}(\d+)\s+(\d{4}-?\d\d-?\d\d)\s+(edgar/data/\S+)\s*$")


def quarter(day: dt.date) -> int:
    return int(markets.quarter_label(day)[-1])


def url(day: Optional[dt.date] = None, year: Optional[int] = None, qtr: Optional[int] = None, **_: Any) -> str:
    """The daily index of `day`, or (with year and qtr instead) the quarter's full index."""
    if day is not None:
        return f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{quarter(day)}/form.{day:%Y%m%d}.idx"
    return f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.gz"


def parse(body: bytes, forms: Sequence[str] = FORMS, **_: Any) -> List[Dict[str, Any]]:
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    text = body.decode("latin-1")
    if "Form Type" not in text[:3000]:
        raise SchemaError("sec_index: no header")
    rows = []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m or m.group(1) not in forms:
            continue
        path = m.group(5)
        file = path.rsplit("/", 1)[-1]
        filed = m.group(4).replace("-", "")
        rows.append(
            {
                "accession": file.removesuffix(".txt"),
                "cik": m.group(3).lstrip("0"),
                "form": m.group(1),
                "filed": f"{filed[:4]}-{filed[4:6]}-{filed[6:]}",
                "company": m.group(2).strip(),
                "ticker": "",
                "sic": None,
                "file": file,
            }
        )
    return conform(rows, "filing")
