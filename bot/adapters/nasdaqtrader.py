"""Nasdaq Trader's symbol directory: every security listed on a US exchange today (nasdaqlisted.txt for
Nasdaq, otherlisted.txt for NYSE and the rest). A symbol that leaves the files has been delisted; its history
is kept.
"""

from __future__ import annotations

from typing import Any, Dict, List

from bot.adapters.base import SchemaError, conform

NAME = "nasdaqtrader"
TERMS = "public symbol directory files published for market participants"
FILES = {"nasdaq": "nasdaqlisted.txt", "other": "otherlisted.txt"}
EXCHANGES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX", "Q": "Nasdaq"}


def url(which: str, **_: Any) -> str:
    return f"https://www.nasdaqtrader.com/dynamic/SymDir/{FILES[which]}"


def parse(body: bytes, which: str, **_: Any) -> List[Dict[str, Any]]:
    lines = body.decode("utf-8", "replace").splitlines()
    if not lines:
        raise SchemaError("nasdaqtrader: empty file")
    head = lines[0].split("|")
    sym_col = "Symbol" if which == "nasdaq" else "ACT Symbol"
    if sym_col not in head or "Security Name" not in head or "Test Issue" not in head:
        raise SchemaError(f"nasdaqtrader: header {head}")
    if not lines[-1].startswith("File Creation Time"):
        raise SchemaError("nasdaqtrader: file is cut short (no creation-time footer)")
    rows = []
    for line in lines[1:-1]:
        f = dict(zip(head, line.split("|")))
        if f.get("Test Issue") == "Y" or f.get("ETF") == "Y":
            continue
        exch = "Q" if which == "nasdaq" else f.get("Exchange", "")
        rows.append({"symbol": f[sym_col], "name": f["Security Name"], "exchange": EXCHANGES.get(exch, exch)})
    return conform(rows, "listing")
