"""Cboe's public market-data CDN: daily history for US stocks and the S&P 500 index.

The second price source: independent of Yahoo, used to reconcile closes and to stand in when Yahoo fails.
Cboe's history is not always split-adjusted the way Yahoo's is, so reconciliation compares only bars after
the newest split.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from bot.adapters.base import SchemaError, conform

NAME = "cboe"
TERMS = "public CDN files behind cboe.com's own pages; no key; read slowly under a named User-Agent"
INDICES = {"SPX": "_SPX"}


def url(symbol: str, **_: Any) -> str:
    return f"https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{INDICES.get(symbol, symbol)}.json"


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    f = float(v)
    return f if f > 0 else None  # Cboe writes 0 for a value it does not have (an index's open)


def parse(
    body: bytes, symbol: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None, **_: Any
) -> List[Dict[str, Any]]:
    doc = json.loads(body)
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        raise SchemaError("cboe: no data list")
    lo, hi = (start.isoformat() if start else ""), (end.isoformat() if end else "9999")
    bars = []
    for d in data:
        if not lo <= d["date"] <= hi:
            continue
        close = _num(d["close"])
        if close is None:
            continue
        bars.append(
            {
                "symbol": symbol,
                "date": d["date"],
                "open": _num(d.get("open")),
                "high": _num(d.get("high")),
                "low": _num(d.get("low")),
                "close": close,
                "volume": _num(d.get("volume")),
            }
        )
    return conform(bars, "bar")
