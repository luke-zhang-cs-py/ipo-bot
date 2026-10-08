"""Collection: the universe, daily prices and macro series, read incrementally into the store.

Each step reads only what is new (from the newest stored day, re-reading a few days before it to catch
revisions), falls back to a second source when the first fails, and records a warning instead of stopping
the run when a source is down.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from bot import checks, markets
from bot.adapters import cboe, cboe_vix, fred, nasdaqtrader, treasury, wikipedia, yahoo
from bot.adapters.base import first_ok, read
from bot.context import Ctx
from bot.store import available, iso

LEFT_WINDOW = dt.timedelta(days=30)  # how long a symbol that left is still collected
INDEX = "SPX"  # the S&P 500 index itself: read from both sources every day
PRICE_DELAY = dt.timedelta(minutes=15)  # a close is published within minutes of 16:00 New York
VIX_DELAY = dt.timedelta(minutes=30)  # the VIX settles at 16:15
TREASURY_DELAY = dt.timedelta(hours=2)  # the Treasury posts the day's curve by 18:00 New York


def _nominal(day: str, delay: dt.timedelta) -> dt.datetime:
    return markets.close_utc(dt.date.fromisoformat(day)) + delay


# ---------------------------------------------------------------------------- the universe


def update_universe(ctx: Ctx) -> List[str]:
    """Record who is in the universe today and which tracked symbols are still listed; returns the symbols to
    collect: every member past or present that is not known to be delisted (BOT_SYMBOLS replaces the index)."""
    st = ctx.store
    if ctx.cfg.symbols:
        st.put(
            "universe",
            [
                {
                    "symbol": s,
                    "source": "config",
                    "name": s,
                    "cik": None,
                    "sector": None,
                    "exchange": None,
                    "added": None,
                    "member": 1,
                    "listed": 1,
                }
                for s in ctx.cfg.symbols
            ],
            ctx.run_id,
            ctx.at,
        )
        return list(ctx.cfg.symbols)
    res = read(ctx.http, wikipedia, min_members=ctx.cfg.min_members)
    before = {r["symbol"]: r for r in st.asof("universe", where={"source": "wikipedia"})}
    if res.ok:
        now = {r["symbol"] for r in res.rows}
        rows = [{**r, "source": "wikipedia", "exchange": None, "member": 1, "listed": None} for r in res.rows]
        rows += [{**r, "member": 0} for s, r in before.items() if s not in now and r["member"]]  # left the index: kept
        ctx.count("universe_rows", st.put("universe", rows, ctx.run_id, ctx.at))
    else:
        ctx.warn("universe_not_refreshed", source="wikipedia", error=res.failures[-1][1], kept=len(before))
    tracked = {r["symbol"] for r in st.asof("universe", where={"source": "wikipedia"})}
    _update_listings(ctx, tracked)
    # who to collect: current members, plus anyone who left the index or was delisted in the last month (their
    # final sessions still come in after the news); the rest keep their history and are not read again
    cutoff = iso(ctx.now - LEFT_WINDOW)
    rows = st.asof("universe")
    current = {r["symbol"] for r in rows if r["source"] == "wikipedia" and r["member"] == 1}
    gone = {r["symbol"] for r in rows if r["source"] == "nasdaqtrader" and r["listed"] == 0}
    recent = {r["symbol"] for r in rows if (r["member"] == 0 or r["listed"] == 0) and r["available_at"] >= cutoff}
    return sorted((current - gone) | recent)


def _update_listings(ctx: Ctx, tracked: Set[str]) -> None:
    """Mark tracked symbols listed or not, from both of Nasdaq Trader's files (only when both were read: one
    file alone would make every symbol in the other look delisted)."""
    reads = [read(ctx.http, nasdaqtrader, which=w) for w in ("nasdaq", "other")]
    if not all(r.ok for r in reads):
        ctx.warn("listings_not_refreshed", source="nasdaqtrader", errors=[f for r in reads for f in r.failures])
        return
    listed = {r["symbol"]: r for res in reads for r in res.rows}
    rows = [
        {
            "symbol": s,
            "source": "nasdaqtrader",
            "name": listed[s]["name"] if s in listed else None,
            "cik": None,
            "sector": None,
            "exchange": listed[s]["exchange"] if s in listed else None,
            "added": None,
            "member": None,
            "listed": int(s in listed),
        }
        for s in sorted(tracked)
    ]
    ctx.count("listing_rows", ctx.store.put("universe", rows, ctx.run_id, ctx.at))


# ---------------------------------------------------------------------------- prices


def update_prices(ctx: Ctx, symbols: Sequence[str]) -> Dict[str, str]:
    """Bring every symbol's daily bars up to the last finished session. Returns symbol -> the source that
    answered (or "failed")."""
    end = markets.last_closed_session(ctx.now)
    out = {}
    for sym in [*symbols, INDEX]:
        out[sym] = update_symbol(ctx, sym, end)
    return out


def _start(ctx: Ctx, sym: str, end: dt.date) -> Optional[dt.date]:
    last = ctx.store.last_date("prices", {"symbol": sym})
    if last is None:
        return dt.date.fromisoformat(ctx.cfg.history_start)
    last_d = dt.date.fromisoformat(last)
    if last_d >= end:
        return None  # already current: nothing to read
    d = last_d
    for _ in range(ctx.cfg.refetch_days):
        d = markets.previous_trading_day(d)
    return d


def update_symbol(ctx: Ctx, sym: str, end: dt.date, full: bool = False) -> str:
    """One symbol's new bars (all of them since history_start when full, as after a split)."""
    start = dt.date.fromisoformat(ctx.cfg.history_start) if full else _start(ctx, sym, end)
    if start is None:
        return "current"
    kw = {"symbol": sym, "start": start, "end": end}
    res = first_ok(ctx.http, [(yahoo, kw), (cboe, kw)], ctx.log, what=f"prices {sym}")
    if not res.ok:
        ctx.warn("prices_failed", symbol=sym, errors=res.failures)
        return "failed"
    if res.degraded:
        ctx.warn("prices_fallback", symbol=sym, source=res.source, errors=res.failures)
    lo, hi = start.isoformat(), end.isoformat()
    bars, dupes = checks.dedupe([b for b in res.rows if lo <= b["date"] <= hi], ("symbol", "date"))
    for d in dupes:
        ctx.warn("duplicate_rows", source=res.source, key=d["subject"])
    rows = [
        {**b, "source": res.source, "available_at": available(_nominal(b["date"], PRICE_DELAY), ctx.now)} for b in bars
    ]
    ctx.count("price_rows", ctx.store.put("prices", rows, ctx.run_id, ctx.at))
    actions = [a for a in res.extra.get("actions", []) if a["date"] <= hi]
    known = {(a["date"], a["kind"]) for a in ctx.store.asof("actions", where={"symbol": sym})}
    new_splits = [a for a in actions if a["kind"] == "split" and (a["date"], "split") not in known]
    ctx.store.put(
        "actions",
        [
            {**a, "source": res.source, "available_at": available(_nominal(a["date"], PRICE_DELAY), ctx.now)}
            for a in actions
        ],
        ctx.run_id,
        ctx.at,
    )
    if new_splits and not full and lo > ctx.cfg.history_start:
        # the source has rescaled every earlier price: read the whole history again, as a new version
        ctx.count("split_refetches")
        return update_symbol(ctx, sym, end, full=True)
    return str(res.source)


def rotation(symbols: Sequence[str], day: dt.date, share: int = 20) -> List[str]:
    """The slice of the universe reconciled today: 1/share of it, so all of it every `share` trading days."""
    ordered = sorted(symbols)
    n = max(1, math.ceil(len(ordered) / share))
    k = (day.toordinal() % share) * n
    return ordered[k : k + n]


def reconcile(ctx: Ctx, symbols: Iterable[str]) -> List[Dict[str, Any]]:
    """Compare Yahoo's recent closes with Cboe's for these symbols (and the index). Returns one row per
    compared day: {symbol, date, yahoo, cboe, diff, ok}. Days before a split inside the window are skipped,
    because the two sources may scale them differently."""
    end = markets.last_closed_session(ctx.now)
    start = end
    for _ in range(ctx.cfg.reconcile_days):
        start = markets.previous_trading_day(start)
    out: List[Dict[str, Any]] = []
    for sym in sorted({*symbols, INDEX}):
        res = read(ctx.http, cboe, symbol=sym, start=start, end=end)
        if not res.ok:
            ctx.warn("reconcile_unavailable", symbol=sym, error=res.failures[-1][1])
            continue
        ctx.store.put(
            "prices",
            [
                {**b, "source": "cboe", "available_at": available(_nominal(b["date"], PRICE_DELAY), ctx.now)}
                for b in res.rows
            ],
            ctx.run_id,
            ctx.at,
        )
        splits = [a["date"] for a in ctx.store.asof("actions", where={"symbol": sym, "kind": "split"})]
        after = max([s for s in splits if s > start.isoformat()], default=start.isoformat())
        ours = {
            r["date"]: r["close"]
            for r in ctx.store.asof("prices", where={"symbol": sym, "source": "yahoo"})
            if r["date"] >= after
        }
        for b in res.rows:
            if b["date"] in ours and b["date"] >= after:
                diff = b["close"] / ours[b["date"]] - 1
                out.append(
                    {
                        "symbol": sym,
                        "date": b["date"],
                        "yahoo": ours[b["date"]],
                        "cboe": b["close"],
                        "diff": diff,
                        "ok": abs(diff) <= ctx.cfg.reconcile_tolerance,
                    }
                )
    return out


# ---------------------------------------------------------------------------- macro

MACRO_SERIES = ("UST3M", "UST2Y", "UST10Y", "VIX")


def update_macro(ctx: Ctx) -> Dict[str, str]:
    """Treasury yields (Treasury, else FRED) and the VIX (Cboe, else FRED). Returns series -> source."""
    out: Dict[str, str] = {}
    first_year = int(ctx.cfg.history_start[:4])
    last = ctx.store.last_date("macro", {"series": "UST10Y"})
    years = range(max(first_year, int(last[:4])) if last else first_year, ctx.now.year + 1)
    got: List[Dict[str, Any]] = []
    failed = False
    for y in years:
        res = read(ctx.http, treasury, year=y)
        if res.ok:
            got += res.rows
        else:
            failed = True
            ctx.warn("treasury_failed", year=y, error=res.failures[-1][1])
    src = "treasury"
    if failed:
        got, src = [], "fred"
        for s in ("UST3M", "UST2Y", "UST10Y"):
            res = read(ctx.http, fred, series=s)
            if res.ok:
                got += [r for r in res.rows if r["date"] >= ctx.cfg.history_start]
            else:
                ctx.warn("macro_failed", series=s, errors=["treasury failed", res.failures[-1][1]])
                out[s] = "failed"
    _put_macro(ctx, got, src, TREASURY_DELAY)
    for s in ("UST3M", "UST2Y", "UST10Y"):
        out.setdefault(s, src)
    res = first_ok(ctx.http, [(cboe_vix, {}), (fred, {"series": "VIX"})], ctx.log, what="VIX")
    if res.ok:
        _put_macro(ctx, [r for r in res.rows if r["date"] >= ctx.cfg.history_start], str(res.source), VIX_DELAY)
        if res.degraded:
            ctx.warn("macro_fallback", series="VIX", source=res.source, errors=res.failures)
    else:
        ctx.warn("macro_failed", series="VIX", errors=res.failures)
    out["VIX"] = str(res.source) if res.ok else "failed"
    return out


def _put_macro(ctx: Ctx, rows: List[Dict[str, Any]], source: str, delay: dt.timedelta) -> None:
    rows, dupes = checks.dedupe(rows, ("series", "date"))  # type: ignore[assignment]
    for d in dupes:
        ctx.warn("duplicate_rows", source=source, key=d["subject"])
    rows = [{**r, "source": source, "available_at": available(_nominal(r["date"], delay), ctx.now)} for r in rows]
    ctx.count("macro_rows", ctx.store.put("macro", rows, ctx.run_id, ctx.at))
