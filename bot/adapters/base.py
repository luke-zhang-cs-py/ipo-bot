"""What every adapter shares: fetch, parse, check the result's shape, fall back to the next source."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bot.http import Http, SourceError
from bot.log import event

Row = Dict[str, Any]

# The shape of each kind of row an adapter returns: column -> type (float columns may be None only if listed).
SCHEMAS: Dict[str, Dict[str, type]] = {
    "bar": {"symbol": str, "date": str, "open": float, "high": float, "low": float, "close": float, "volume": float},
    "action": {"symbol": str, "date": str, "kind": str, "value": float},
    "macro": {"series": str, "date": str, "value": float},
    "member": {"symbol": str, "name": str, "cik": str, "sector": str, "added": str},
    "listing": {"symbol": str, "name": str, "exchange": str},
    "filing": {"accession": str, "cik": str, "form": str, "filed": str, "company": str},
}
NULLABLE = {"bar": {"open", "high", "low", "volume"}, "member": {"cik", "sector", "added"}, "filing": set()}


class SchemaError(SourceError):
    """A response that does not have the shape the parser expects: the source changed its format."""

    def __init__(self, message: str):
        super().__init__("schema", message)


def conform(rows: Sequence[Row], kind: str) -> List[Row]:
    """Rows checked against SCHEMAS[kind] (ints taken as floats). Raises SchemaError on a missing column, a
    wrong type or a non-finite number."""
    schema, nullable = SCHEMAS[kind], NULLABLE.get(kind, set())
    out = []
    for r in rows:
        fixed = dict(r)
        for col, typ in schema.items():
            v = r.get(col)
            if v is None:
                if col in nullable:
                    continue
                raise SchemaError(f"{kind}: missing {col} in {r}")
            if typ is float and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            if not isinstance(v, typ):
                raise SchemaError(f"{kind}: {col} is {type(v).__name__}, not {typ.__name__}")
            if isinstance(v, float) and not math.isfinite(v):
                raise SchemaError(f"{kind}: {col} is not finite")
            fixed[col] = v
        out.append(fixed)
    return out


@dataclass
class Result:
    """What a read got: rows (empty when every source failed), the source that answered, and what went wrong."""

    rows: List[Row] = field(default_factory=list)
    source: Optional[str] = None
    failures: List[Tuple[str, str]] = field(default_factory=list)  # (source, error)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.source is not None

    @property
    def degraded(self) -> bool:
        """Answered, but not by the first choice."""
        return self.ok and bool(self.failures)


def read(http: Http, adapter: ModuleType, **kw: Any) -> Result:
    """Fetch and parse one source. Never raises for a source problem: the failure is in the result."""
    try:
        url = adapter.url(**kw)
        body = http.get(url, headers=getattr(adapter, "HEADERS", None), limit=getattr(adapter, "LIMIT", None))
        parsed = adapter.parse(body, **kw)
    except SourceError as e:
        return Result(failures=[(adapter.NAME, str(e))])
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        return Result(failures=[(adapter.NAME, f"schema: {type(e).__name__}: {e}")])
    if isinstance(parsed, dict):  # a parser that returns one record puts it in extra
        rows, extra = [], parsed
    else:
        rows, extra = parsed if isinstance(parsed, tuple) else (parsed, {})
    return Result(rows=rows, source=adapter.NAME, extra=extra)


def first_ok(
    http: Http,
    attempts: Sequence[Tuple[ModuleType, Mapping[str, Any]]],
    log: Optional[logging.Logger] = None,
    what: str = "",
) -> Result:
    """The first of the attempts that answers; earlier failures are kept in the result (and logged)."""
    failures: List[Tuple[str, str]] = []
    for adapter, kw in attempts:
        res = read(http, adapter, **kw)
        if res.ok:
            res.failures = failures
            if failures and log:
                event(log, "fallback_used", logging.WARNING, what=what, source=res.source, failures=failures)
            return res
        failures += res.failures
        if log:
            event(log, "source_failed", logging.WARNING, what=what, source=adapter.NAME, error=res.failures[-1][1])
    return Result(failures=failures)
