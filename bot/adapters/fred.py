"""FRED's keyless CSV download (fredgraph.csv): the fallback for Treasury yields and the VIX.

Some networks block fred.stlouisfed.org; the bot then keeps the primary source's data and says so.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

from bot.adapters.base import SchemaError, conform

NAME = "fred"
TERMS = "St. Louis Fed public download; each series' copyright stays with its source (Treasury, Cboe)"
SERIES = {"UST3M": "DGS3MO", "UST2Y": "DGS2", "UST10Y": "DGS10", "VIX": "VIXCLS"}


def url(series: str, **_: Any) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES[series]}"


def parse(body: bytes, series: str, **_: Any) -> List[Dict[str, Any]]:
    reader = csv.reader(io.StringIO(body.decode("utf-8-sig")))
    head = next(reader, None)
    if not head or len(head) != 2 or head[1] != SERIES[series]:
        raise SchemaError(f"fred: header {head}")
    rows = [{"series": series, "date": d, "value": float(v)} for d, v in reader if v not in (".", "")]
    return conform(rows, "macro")
