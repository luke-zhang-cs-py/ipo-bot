"""The EDGAR side of an update: the IPO pipeline's filings, the companies behind them, what their documents
say, and each IPO's first trade.

Recent days come from the daily form index (full-text search when the index is down); older quarters are
backfilled from the quarterly index, newest first, with whatever is left of the run's SEC request budget, so a
fresh database fills up over a few runs without any one run hammering EDGAR.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bot import ipos, markets
from bot.adapters import cboe, sec_company, sec_doc, sec_index, sec_search, yahoo
from bot.adapters.base import first_ok, read
from bot.context import Ctx
from bot.store import available

DAILY_LOOKBACK = 10  # days the daily index is read back on a fresh database
FILING_DELAY = dt.timedelta(hours=6)  # EDGAR disseminates until 22:00 New York: a filing is public by then
RETRY_DAYS = 7  # how often a priced deal with no first trade found is looked for again
FIRST_TRADE_WINDOW = 10  # trading days after pricing in which the first trade must fall
DOC_FORMS = (*ipos.REGISTRATIONS, "424B4")


class Budget:
    """SEC requests left in this run."""

    def __init__(self, ctx: Ctx):
        self.ctx, self.start = ctx, ctx.http.requests

    @property
    def left(self) -> int:
        return self.ctx.cfg.sec_budget - (self.ctx.http.requests - self.start)


def _filed_at(day: str, now: dt.datetime) -> str:
    return available(markets.close_utc(dt.date.fromisoformat(day)) + FILING_DELAY, now)


def _put_filings(ctx: Ctx, rows: Sequence[Dict[str, Any]]) -> int:
    rows = [{**r, "available_at": _filed_at(r["filed"], ctx.now)} for r in rows]
    n = ctx.store.put("filings", rows, ctx.run_id, ctx.at)
    ctx.count("filing_rows", n)
    return n


def update(ctx: Ctx) -> Dict[str, Any]:
    """The whole EDGAR step; returns a summary. Without SEC_USER_AGENT it is skipped with a warning."""
    if not ctx.cfg.sec_user_agent:
        ctx.warn("edgar_skipped", reason='SEC_USER_AGENT is not set; EDGAR asks every caller for "Name email"')
        return {"skipped": True}
    budget = Budget(ctx)
    out: Dict[str, Any] = {"days": daily(ctx, budget)}
    out["companies"] = companies(ctx, budget)
    out["documents"] = documents(ctx, budget)
    out["quarters"] = backfill(ctx, budget)
    out["first_trades"] = first_trades(ctx)
    out["sec_requests"] = ctx.cfg.sec_budget - budget.left
    return out


def eastern_today(now: dt.datetime) -> dt.date:
    return (now + dt.timedelta(hours=markets.eastern_offset_hours(now))).date()


def daily(ctx: Ctx, budget: Budget) -> List[str]:
    """Read each weekday's index from the day after the cursor through today (New York time).

    A day is settled once its index has been read, or once it is two days past with no index (EDGAR was closed:
    a federal holiday). The cursor moves only over settled days, so a day that failed is read again next run.
    Today's and yesterday's indexes may not be published yet; full-text search fills in for them meanwhile, and
    for any day whose index failed (with a warning: search misses some forms, so the day stays unsettled).
    """
    today = eastern_today(ctx.now)
    cur = ctx.store.cursor("edgar_daily")
    day = dt.date.fromisoformat(cur) + dt.timedelta(days=1) if cur else today - dt.timedelta(days=DAILY_LOOKBACK)
    if not ctx.store.cursor("edgar_backfill"):
        # the backfill starts with the quarter holding the daily window's first day (the overlap is a no-op)
        nxt = dt.date(day.year + (day.month > 9), (((day.month - 1) // 3 + 1) % 4) * 3 + 1, 1)
        ctx.store.set_cursor("edgar_backfill", nxt.isoformat(), ctx.run_id, ctx.at)
    reads: List[Tuple[dt.date, Optional[Any]]] = []
    while day <= today and budget.left > 0:
        reads.append((day, read(ctx.http, sec_index, day=day) if day.weekday() < 5 else None))
        day += dt.timedelta(days=1)
    # EDGAR answers 403, not 404, for an index that does not exist (a holiday, today before it is published).
    # A 403 means "no such index" only when other indexes were read this run; otherwise it may be a block.
    answering = any(res is not None and res.ok for _, res in reads)
    done, settled = [], True
    for day, res in reads:
        ok = True
        if res is not None:
            kind = res.failures[-1][1].split(":")[0] if res.failures else ""
            missing = kind == "not_found" or (kind == "blocked" and answering)
            if res.ok:
                _put_filings(ctx, res.rows)
            elif missing and (today - day).days >= 2:
                pass  # no index two days on: EDGAR was closed that day
            else:
                ok = False
                found = read(ctx.http, sec_search, start=day, end=day)
                if found.ok:
                    _put_filings(ctx, _search_all(ctx, day, found.rows, found.extra.get("total", 0)))
                if not missing:
                    ctx.warn(
                        "filings_fallback" if found.ok else "filings_failed",
                        day=day.isoformat(),
                        errors=res.failures + found.failures,
                    )
        settled = settled and ok and day < today
        if settled:
            ctx.store.set_cursor("edgar_daily", day.isoformat(), ctx.run_id, ctx.at)
        done.append(day.isoformat())
    return done


def _search_all(ctx: Ctx, day: dt.date, rows: List[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
    """Every page of a full-text search for one day."""
    offset = sec_search.PAGE
    while offset < total:
        res = read(ctx.http, sec_search, start=day, end=day, offset=offset)
        if not res.ok:
            break
        rows += res.rows
        offset += sec_search.PAGE
    return rows


def backfill(ctx: Ctx, budget: Budget, reserve: int = 50) -> List[str]:
    """Quarters before the daily window, newest first, down to history_start's quarter."""
    done = []
    first = dt.date.fromisoformat(ctx.cfg.history_start)
    while budget.left > reserve:
        cur = ctx.store.cursor("edgar_backfill")
        if not cur:
            break
        start = dt.date.fromisoformat(cur)  # the first day of the earliest quarter already covered
        prev = (start - dt.timedelta(days=1)).replace(day=1)
        prev = prev.replace(month=((prev.month - 1) // 3) * 3 + 1)
        if prev < first.replace(day=1, month=((first.month - 1) // 3) * 3 + 1):
            break
        res = read(ctx.http, sec_index, year=prev.year, qtr=sec_index.quarter(prev))
        if not res.ok:
            ctx.warn("backfill_failed", quarter=f"{prev.year}Q{sec_index.quarter(prev)}", error=res.failures[-1][1])
            break
        _put_filings(ctx, [r for r in res.rows if r["filed"] >= ctx.cfg.history_start])
        ctx.store.set_cursor("edgar_backfill", prev.isoformat(), ctx.run_id, ctx.at)
        done.append(f"{prev.year}Q{sec_index.quarter(prev)}")
        companies(ctx, budget)
        documents(ctx, budget)
    return done


def _registrants(ctx: Ctx) -> List[str]:
    """CIKs that filed an IPO registration, newest first."""
    rows = ctx.store.query(
        "SELECT cik, max(filed) AS last FROM filings WHERE form IN (?, ?, ?, ?) GROUP BY cik " "ORDER BY last DESC",
        ipos.REGISTRATIONS,
    )
    return [r["cik"] for r in rows]


def companies(ctx: Ctx, budget: Budget) -> int:
    """Look up each new registrant's EDGAR record (is it already reporting? a SPAC?)."""
    known = {r["cik"] for r in ctx.store.asof("companies")}
    n = 0
    for cik in _registrants(ctx):
        if cik in known:
            continue
        if budget.left <= 0:
            break
        res = read(ctx.http, sec_company, cik=cik)
        if not res.ok:
            ctx.warn("company_failed", cik=cik, error=res.failures[-1][1])
            continue
        info = res.extra
        # The recent page lists the newest thousand filings. A company with more (older pages exist) and no report
        # among the recent ones is checked page by page: a report anywhere before registering makes it a follow-on.
        pages = [] if info["first_report"] else list(info.get("pages", []))
        first_report = info["first_report"]
        for page in pages:
            more = read(ctx.http, sec_company, cik=cik, page=page)
            if more.ok and more.extra["first_report"]:
                first_report = min(r for r in (first_report, more.extra["first_report"]) if r)
        ctx.store.put(
            "companies",
            [
                {
                    "cik": cik,
                    "name": info["name"],
                    "sic": info["sic"],
                    "tickers": json.dumps(info["tickers"]),
                    "first_report": first_report,
                }
            ],
            ctx.run_id,
            ctx.at,
        )
        n += 1
    return n


def documents(ctx: Ctx, budget: Budget) -> int:
    """Read the cover of each IPO registration and final prospectus not read yet: live deals first, then the
    newest historical ones. Follow-ons and SPACs are not read."""
    deals = {d.cik: d for d in _deals(ctx)}
    have = {r["accession"] for r in ctx.store.asof("documents")}
    todo = ctx.store.query(
        f"SELECT accession, cik, form, filed, file FROM filings WHERE form IN ({', '.join('?' * len(DOC_FORMS))}) "
        "GROUP BY accession ORDER BY filed DESC",
        DOC_FORMS,
    )
    n = 0
    for f in todo:
        d = deals.get(f["cik"])
        if f["accession"] in have or not d or not d.ipo:
            continue
        if budget.left <= 0:
            break
        res = read(
            ctx.http,
            sec_doc,
            cik=f["cik"],
            accession=f["accession"],
            file=f["file"],
            form=f["form"],
            marketed=d.latest_range,
        )
        if not res.ok:
            ctx.warn("document_failed", accession=f["accession"], error=res.failures[-1][1])
            continue
        ctx.store.put(
            "documents",
            [
                {
                    "accession": f["accession"],
                    "data": json.dumps(res.extra, sort_keys=True),
                    "available_at": _filed_at(f["filed"], ctx.now),
                }
            ],
            ctx.run_id,
            ctx.at,
        )
        n += 1
    ctx.count("documents_read", n)
    return n


def _deals(ctx: Ctx, at: Optional[str] = None) -> List[ipos.Deal]:
    return deals_asof(ctx.store, at)


def deals_asof(store: Any, at: Optional[str] = None) -> List[ipos.Deal]:
    """Every deal as it was knowable at `at` (None: everything stored)."""
    docs = {r["accession"]: r["data"] for r in store.asof("documents", at)}
    cos = {r["cik"]: r for r in store.asof("companies", at)}
    trades = {r["cik"]: r for r in store.asof("first_trades", at)}
    return ipos.build(store.asof("filings", at), docs, cos, trades)


def first_trades(ctx: Ctx) -> int:
    """Find the first trading day of each priced or effective IPO that has a symbol and no first trade yet."""
    end = markets.last_closed_session(ctx.now)
    n = 0
    for d in _deals(ctx):
        since = d.priced or d.effective
        if not d.ipo or d.first_trade or d.withdrawn or not since or not d.symbol:
            continue
        start = dt.date.fromisoformat(since) - dt.timedelta(days=7)
        tried = ctx.store.cursor(f"first_trade_tried:{d.cik}")
        if tried and (ctx.now.date() - dt.date.fromisoformat(tried)).days < RETRY_DAYS and (end - start).days > 30:
            continue  # looked for recently and not found: an old deal is looked for again once a week
        found = _first_trade(ctx, d, start, end)
        if not found:
            ctx.store.set_cursor(f"first_trade_tried:{d.cik}", ctx.now.date().isoformat(), ctx.run_id, ctx.at)
        if found:
            day, row = found
            ctx.store.put(
                "first_trades",
                [
                    {
                        **row,
                        "cik": d.cik,
                        "available_at": available(markets.close_utc(day) + dt.timedelta(minutes=15), ctx.now),
                    }
                ],
                ctx.run_id,
                ctx.at,
            )
            n += 1
    ctx.count("first_trades", n)
    return n


def _first_trade(ctx: Ctx, d: ipos.Deal, start: dt.date, end: dt.date) -> Optional[Tuple[dt.date, Dict[str, Any]]]:
    sym = str(d.symbol)
    kw = {"symbol": sym, "start": start, "end": end}
    res = first_ok(ctx.http, [(yahoo, kw), (cboe, kw)], ctx.log, what=f"first trade {sym}")
    bars = sorted((b for b in res.rows if start.isoformat() <= b["date"] <= end.isoformat()), key=lambda b: b["date"])
    if not bars:
        return None
    first = dt.date.fromisoformat(bars[0]["date"])
    since = dt.date.fromisoformat(d.priced or d.effective or d.registered)
    if first < markets.previous_trading_day(since) or len(markets.trading_days(since, first)) > FIRST_TRADE_WINDOW + 1:
        return None  # trading before it priced (an uplisting, a reused ticker), or not until long after
    factor = 1.0  # undo later splits so the first day compares with the offer
    for a in res.extra.get("actions", []):
        if a["kind"] == "split" and a["date"] > bars[0]["date"]:
            factor *= a["value"]
    b = bars[0]
    return first, {
        "symbol": sym,
        "date": b["date"],
        "open": b["open"] * factor if b["open"] else None,
        "close": b["close"] * factor,
        "source": res.source,
    }
