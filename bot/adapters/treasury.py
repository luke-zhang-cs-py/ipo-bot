"""US Treasury daily par yield curve rates, from the Treasury's own CSV download (public, no key)."""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Any, Dict, List

from bot.adapters.base import SchemaError, conform

NAME = "treasury"
TERMS = "US government data, public domain"
COLUMNS = {"3 Mo": "UST3M", "2 Yr": "UST2Y", "10 Yr": "UST10Y"}


def url(year: int, **_: Any) -> str:
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
        f"{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    )


def parse(body: bytes, **_: Any) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    names = reader.fieldnames or []
    if "Date" not in names or not all(c in names for c in COLUMNS):
        raise SchemaError(f"treasury: columns {names}")
    rows = []
    for r in reader:
        day = dt.datetime.strptime(r["Date"], "%m/%d/%Y").date().isoformat()
        rows += [{"series": s, "date": day, "value": float(r[c])} for c, s in COLUMNS.items() if r.get(c)]
    rows.sort(key=lambda x: (x["date"], x["series"]))
    return conform(rows, "macro")
