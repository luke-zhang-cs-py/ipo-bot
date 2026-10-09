"""Which revision of Wikipedia's "List of S&P 500 companies" was current at a moment, from the MediaWiki query
API (keyless; automated reads allowed with a descriptive User-Agent). Its id is then read with wikipedia's
parser (`oldid`) to get the member table as it stood then.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from bot.adapters.base import SchemaError

NAME = "wikipedia"
TERMS = "MediaWiki API, CC BY-SA content; automated reads allowed with a descriptive User-Agent"


def url(at: str, **_: Any) -> str:
    """The newest revision at or before `at` (an ISO moment such as 2016-01-01T00:00:00Z)."""
    return (
        "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=List_of_S%26P_500_companies"
        f"&rvlimit=1&rvdir=older&rvstart={at}&rvprop=ids%7Ctimestamp&format=json&formatversion=2"
    )


def parse(body: bytes, **_: Any) -> List[Dict[str, Any]]:
    """[{revid, timestamp}], or [] when the page had no revision yet at that moment."""
    pages = json.loads(body)["query"]["pages"]
    if not pages:
        raise SchemaError("wikipedia: no page in the revisions answer")
    revs = pages[0].get("revisions") or []
    return [{"revid": int(r["revid"]), "timestamp": str(r["timestamp"])} for r in revs[:1]]
