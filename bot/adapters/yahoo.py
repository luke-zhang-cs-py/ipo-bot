"""Yahoo Finance's chart endpoint: daily bars (split-adjusted as of the read), splits and dividends.

Unofficial and undocumented: no key and no published terms for programmatic use, so it is read slowly (2 a
second), under the bot's own User-Agent, and checked against Cboe. Missing prices are left missing.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Tuple

from bot.adapters.base import SchemaError, conform
from bot.http import SourceError

NAME = "yahoo"
TERMS = "unofficial endpoint, no published API terms; read at 2 requests a second under a named User-Agent"


def _epoch(d: dt.date) -> int:
    return int(dt.datetime.combine(d, dt.time(), tzinfo=dt.UTC).timestamp())


def ticker(symbol: str) -> str:
    """Yahoo writes share classes with a dash (BRK-B) and the S&P 500 index as ^GSPC."""
    return {"SPX": "%5EGSPC"}.get(symbol, symbol.replace(".", "-"))


def url(symbol: str, start: dt.date, end: dt.date, **_: Any) -> str:
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker(symbol)}?period1={_epoch(start)}"
        f"&period2={_epoch(end + dt.timedelta(days=1))}&interval=1d&events=div%2Csplit"
    )


def parse(body: bytes, symbol: str, **_: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(bars, {"actions": splits and dividends}). A symbol Yahoo does not know raises not_found."""
    doc = json.loads(body)
    chart = doc.get("chart")
    if not isinstance(chart, dict):
        raise SchemaError("yahoo: no chart object")
    err = chart.get("error")
    if err:
        code = err.get("code") if isinstance(err, dict) else str(err)
        raise SourceError("not_found" if code == "Not Found" else "http", f"yahoo {symbol}: {code}")
    res = (chart.get("result") or [None])[0]
    if not isinstance(res, dict) or "meta" not in res:
        raise SchemaError("yahoo: no result")
    offset = int(res["meta"].get("gmtoffset", -14400))
    stamps = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if stamps and not all(k in quote for k in ("open", "high", "low", "close", "volume")):
        raise SchemaError("yahoo: quote fields missing")

    def day(t: int) -> str:
        return dt.datetime.fromtimestamp(t + offset, dt.UTC).date().isoformat()

    bars = []
    for i, t in enumerate(stamps):
        if quote["close"][i] is None:
            continue  # a day Yahoo has no close for: missing, not invented
        bars.append(
            {
                "symbol": symbol,
                "date": day(t),
                "open": quote["open"][i],
                "high": quote["high"][i],
                "low": quote["low"][i],
                "close": quote["close"][i],
                "volume": quote["volume"][i],
            }
        )
    events = res.get("events") or {}
    actions = [
        {"symbol": symbol, "date": day(int(s["date"])), "kind": "split", "value": s["numerator"] / s["denominator"]}
        for s in (events.get("splits") or {}).values()
        if s.get("denominator")
    ]
    actions += [
        {"symbol": symbol, "date": day(int(d["date"])), "kind": "dividend", "value": d["amount"]}
        for d in (events.get("dividends") or {}).values()
    ]
    actions.sort(key=lambda a: (a["date"], a["kind"]))
    return conform(bars, "bar"), {"actions": conform(actions, "action")}
