"""Collection: the universe, daily prices and macro series, read incrementally into the store.

The universe has a past as well as a present: one membership snapshot per month since history_start, read from
the page history of Wikipedia's list (the revision current on the 1st of each month) and stamped with that
revision's time, so a backtest can include a stock only on days it was in the index. Past members that are no
longer in the index get their price history read once, a few each run.

Each step reads only what is new (from the newest stored day, re-reading a few days before it to catch
revisions), falls back to a second source when the first fails, and records a warning instead of stopping
the run when a source is down.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from bot import checks, data, markets
from bot.adapters import cboe, cboe_vix, fred, nasdaqtrader, treasury, wikipedia, wikipedia_revisions, yahoo
from bot.adapters.base import first_ok, read
from bot.context import Ctx
from bot.log import event
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
    update_membership_history(ctx)
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


def _months(start: str, now: dt.datetime) -> List[str]:
    """Every month (YYYY-MM) from start's to now's."""
    y, m = int(start[:4]), int(start[5:7])
    out = []
    while (y, m) <= (now.year, now.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def update_membership_history(ctx: Ctx) -> int:
    """Backfill one membership snapshot per month, oldest first, from where the last run stopped: the page's
    revision current at 00:00 UTC on the 1st, its table read and stored with available_at = the revision's
    timestamp (members flagged 1, earlier members missing from it flagged 0). At most wiki_history_budget
    requests a run; a month whose revision is the previous month's costs one. A revision whose table cannot be
    read is skipped with a warning; a source failure stops here for this run. Returns the rows stored."""
    st = ctx.store
    cur = st.cursor(data.HISTORY)
    done, last_rev = cur.split("|", 1) if cur else ("", "")
    spent = stored = 0
    for month in _months(ctx.cfg.history_start, ctx.now):
        if month <= done:
            continue
        if spent + 2 > ctx.cfg.wiki_history_budget:
            event(ctx.log, "membership_history_paused", logging.INFO, next_month=month, requests=spent)
            break
        found = read(ctx.http, wikipedia_revisions, at=f"{month}-01T00:00:00Z")
        spent += 1
        if not found.ok:
            ctx.warn("membership_history_failed", month=month, error=found.failures[-1][1])
            break
        if found.rows and str(found.rows[0]["revid"]) != last_rev:
            rev = found.rows[0]
            snap = read(ctx.http, wikipedia, oldid=rev["revid"], min_members=ctx.cfg.min_members)
            spent += 1
            if snap.ok:
                stored += _put_snapshot(ctx, snap.rows, rev["timestamp"])
            elif snap.failures[-1][1].startswith("schema"):
                ctx.warn("membership_snapshot_skipped", month=month, revid=rev["revid"], error=snap.failures[-1][1])
            else:
                ctx.warn("membership_history_failed", month=month, error=snap.failures[-1][1])
                break
            last_rev = str(rev["revid"])
        done = month
        st.set_cursor(data.HISTORY, f"{done}|{last_rev}", ctx.run_id, ctx.at)
    ctx.count("membership_rows", stored)
    return stored


def _put_snapshot(ctx: Ctx, rows: Sequence[Dict[str, Any]], stamp: str) -> int:
    """Store one snapshot as the newest version of each symbol's history row, stamped `stamp`."""
    keep = ("symbol", "name", "cik", "sector", "exchange", "added")
    before = {r["symbol"]: r for r in ctx.store.asof("universe", where={"source": data.HISTORY})}
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            out.append({**r, "source": data.HISTORY, "exchange": None, "member": 1, "listed": None})
    out += [
        {**{k: r[k] for k in keep}, "source": data.HISTORY, "member": 0, "listed": None}
        for s, r in before.items()
        if s not in seen and r["member"]
    ]
    return ctx.store.put("universe", out, ctx.run_id, stamp)


def backfill_leavers(ctx: Ctx, collecting: Sequence[str]) -> int:
    """Read the whole price history of past members that left the index (so the backtest has them on the days
    they were members): those not collected daily, with no prices yet and not tried before, at most
    leaver_budget a run. Each is tried once; a ticker the sources no longer know is noted, not warned about.
    Returns how many were read."""
    if ctx.cfg.symbols:
        return 0
    st = ctx.store
    have = {r["symbol"] for r in st.query("SELECT DISTINCT symbol FROM prices")}
    tried = {r["name"].split(":", 1)[1] for r in st.asof("cursors") if r["name"].startswith("leaver:")}
    skip = have | tried | set(collecting)
    todo = [s for s in data.universe(st) if s not in skip][: ctx.cfg.leaver_budget]
    end = markets.last_closed_session(ctx.now)
    got = 0
    for sym in todo:
        res = update_symbol(ctx, sym, end, full=True, quiet=True)
        st.set_cursor(f"leaver:{sym}", res, ctx.run_id, ctx.at)
        got += res != "failed"
    ctx.count("leavers_read", got)
    return got


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


def update_symbol(ctx: Ctx, sym: str, end: dt.date, full: bool = False, quiet: bool = False) -> str:
    """One symbol's new bars (all of them since history_start when full, as after a split). quiet: a failure
    or a fallback is logged, not warned about (a past member's ticker the sources may no longer know).

    A backfilled bar dated before a split on record is stamped with the read time, not its nominal close time:
    the source has rescaled it (see store.available)."""
    start = dt.date.fromisoformat(ctx.cfg.history_start) if full else _start(ctx, sym, end)
    if start is None:
        return "current"
    kw = {"symbol": sym, "start": start, "end": end}
    res = first_ok(ctx.http, [(yahoo, kw), (cboe, kw)], ctx.log, what=f"prices {sym}")
    if not res.ok:
        if quiet:
            event(ctx.log, "prices_unavailable", logging.INFO, symbol=sym, errors=res.failures)
        else:
            ctx.warn("prices_failed", symbol=sym, errors=res.failures)
        return "failed"
    if res.degraded and not quiet:
        ctx.warn("prices_fallback", symbol=sym, source=res.source, errors=res.failures)
    lo, hi = start.isoformat(), end.isoformat()
    bars, dupes = checks.dedupe([b for b in res.rows if lo <= b["date"] <= hi], ("symbol", "date"))
    for d in dupes:
        ctx.warn("duplicate_rows", source=res.source, key=d["subject"])
    actions = [a for a in res.extra.get("actions", []) if a["date"] <= hi]
    stored = ctx.store.asof("actions", where={"symbol": sym})
    last_split = _last_split(stored, actions)
    rows = [{**b, "source": res.source, "available_at": _stamp(b["date"], last_split, ctx.now)} for b in bars]
    ctx.count("price_rows", ctx.store.put("prices", rows, ctx.run_id, ctx.at))
    known = {(a["date"], a["kind"]) for a in stored}
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


def _last_split(*actions: Iterable[Dict[str, Any]]) -> str:
    """The newest split date among the actions ("" if none)."""
    return max((a["date"] for group in actions for a in group if a["kind"] == "split"), default="")


def _stamp(day: str, last_split: str, now: dt.datetime) -> str:
    """A bar's available_at: rescaled (read time, if backfilled) when a split on record falls after its date."""
    return available(_nominal(day, PRICE_DELAY), now, rescaled=day < last_split)


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
        split_rows = ctx.store.asof("actions", where={"symbol": sym, "kind": "split"})
        ctx.store.put(
            "prices",
            [
                {**b, "source": "cboe", "available_at": _stamp(b["date"], _last_split(split_rows), ctx.now)}
                for b in res.rows
            ],
            ctx.run_id,
            ctx.at,
        )
        splits = [a["date"] for a in split_rows]
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
