"""The VIX close, from Cboe's VIX_History.csv (Cboe publishes the index)."""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Any, Dict, List

from bot.adapters.base import SchemaError, conform

NAME = "cboe_vix"
TERMS = "published by Cboe as a public download"


def url(**_: Any) -> str:
    return "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def parse(body: bytes, **_: Any) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    if not reader.fieldnames or "DATE" not in reader.fieldnames or "CLOSE" not in reader.fieldnames:
        raise SchemaError(f"cboe_vix: columns {reader.fieldnames}")
    rows = [
        {
            "series": "VIX",
            "date": dt.datetime.strptime(r["DATE"], "%m/%d/%Y").date().isoformat(),
            "value": float(r["CLOSE"]),
        }
        for r in reader
        if r["CLOSE"]
    ]
    return conform(rows, "macro")
