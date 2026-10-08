"""Data checks, run on every update. Each returns issues: {check, severity, subject, detail}. Severity "error"
means the data is wrong or missing where it should not be; "warning" means it needs a look.

  prices        prices above zero, high >= low, the close inside the day's range
  duplicates    a batch from a source with two rows for one key
  moves         a daily move beyond max_daily_move that no split or dividend on record explains
  missing_days  trading days on the exchange calendar with no bar, inside a symbol's history
  staleness     the newest bar or macro value older than the last finished session allows; a scheduled job
                that has not succeeded since its last slot
  reconcile     Yahoo and Cboe closes more than reconcile_tolerance apart
  point_in_time a row available before the close it describes; append-only triggers missing
  ipo           offer against range, pricing date against the 424B4, first trade after pricing
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from bot import ipos, markets
from bot.config import Settings
from bot.store import APPEND_ONLY, TABLES, Store, parse_iso

Issue = Dict[str, Any]


def issue(check: str, subject: str, detail: str, severity: str = "error") -> Issue:
    return {"check": check, "severity": severity, "subject": subject, "detail": detail}


def dedupe(rows: Sequence[Mapping[str, Any]], key: Sequence[str]) -> Tuple[List[Mapping[str, Any]], List[Issue]]:
    """Rows with one per key (the last one wins), and an issue per duplicated key."""
    seen: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    issues = []
    for r in rows:
        k = tuple(r[c] for c in key)
        if k in seen:
            issues.append(issue("duplicates", "/".join(map(str, k)), "two rows for one key in one read", "warning"))
        seen[k] = r
    return list(seen.values()), issues


def prices(bars: pd.DataFrame) -> List[Issue]:
    out: List[Issue] = []
    if bars.empty:
        return out
    bad = bars[(bars["close"] <= 0) | (bars["open"].fillna(1) <= 0) | (bars["low"].fillna(1) <= 0)]
    out += [issue("prices", f"{r.symbol} {r.date}", f"non-positive price (close {r.close})") for r in bad.itertuples()]
    h, lo, c = bars["high"], bars["low"], bars["close"]
    tol = 1e-6 * c
    inv = bars[(h.notna() & lo.notna()) & ((h + tol < lo) | (c > h + tol + 0.005 * c) | (c < lo - tol - 0.005 * c))]
    out += [
        issue("prices", f"{r.symbol} {r.date}", f"close {r.close} outside low {r.low} - high {r.high}", "warning")
        for r in inv.itertuples()
    ]
    return out


def moves(
    bars: pd.DataFrame,
    splits: Mapping[str, Mapping[str, float]],
    dividends: Mapping[str, Mapping[str, float]],
    limit: float,
) -> List[Issue]:
    """Close-to-close moves beyond `limit` that no corporate action explains. A split on the day, if the bars
    were not adjusted for it, explains a move near 1/ratio - 1; a dividend explains a drop near its size."""
    out = []
    for sym, g in bars.sort_values("date").groupby("symbol"):
        c = g["close"].to_numpy()
        d = g["date"].to_numpy()
        for i in range(1, len(c)):
            r = c[i] / c[i - 1] - 1
            if abs(r) <= limit:
                continue
            ratio = splits.get(sym, {}).get(d[i])
            if ratio and abs((1 + r) * ratio - 1) < 0.10:
                continue  # an unadjusted split: the move is the split's ratio
            div = dividends.get(sym, {}).get(d[i])
            if div and abs(-div / c[i - 1] - r) < 0.05:
                continue
            # a warning, not an error: crashes and takeovers move prices this much too; it asks for a look
            detail = f"close moved {r:+.0%} from {c[i - 1]:.4g} to {c[i]:.4g}"
            out.append(issue("moves", f"{sym} {d[i]}", detail, "warning"))
    return out


def missing_days(bars: pd.DataFrame, through: dt.date) -> Dict[str, List[str]]:
    """symbol -> trading days with no bar, between the symbol's first bar and `through` (its last bar, if the
    symbol has stopped trading: staleness covers the days after it)."""
    out: Dict[str, List[str]] = {}
    for sym, g in bars.groupby("symbol"):
        have = set(g["date"])
        first, last = dt.date.fromisoformat(min(have)), dt.date.fromisoformat(max(have))
        gap = [d.isoformat() for d in markets.trading_days(first, min(last, through)) if d.isoformat() not in have]
        if gap:
            out[str(sym)] = gap
    return out


def gap_issues(gaps: Mapping[str, Sequence[str]]) -> List[Issue]:
    """One warning per symbol with trading days missing inside its history."""
    return [
        issue(
            "missing_days",
            s,
            f"{len(v)} trading days with no bar: {', '.join(v[:5])}{' ...' if len(v) > 5 else ''}",
            "warning",
        )
        for s, v in sorted(gaps.items())
    ]


def lag(newest: Optional[str], through: dt.date) -> Optional[int]:
    """Trading days between a newest date and `through` (0: current), None when there is no data."""
    if newest is None:
        return None
    d = dt.date.fromisoformat(newest)
    return max(0, len(markets.trading_days(d, through)) - 1)


def staleness(
    bars: pd.DataFrame,
    symbols: Iterable[str],
    macro_newest: Mapping[str, Optional[str]],
    now: dt.datetime,
    cfg: Settings,
) -> List[Issue]:
    through = markets.last_closed_session(now)
    out = []
    newest = bars.groupby("symbol")["date"].max().to_dict() if not bars.empty else {}
    for s in symbols:
        n = lag(newest.get(s), through)
        if n is None:
            out.append(issue("staleness", s, "no prices at all"))
        elif n > cfg.stale_trading_days:
            out.append(issue("staleness", s, f"newest bar {newest[s]}, {n} trading days behind {through}"))
    for series, d in macro_newest.items():
        n = lag(d, through)
        if n is None or n > cfg.stale_trading_days + 1:  # yields post after the close: one more day of slack
            out.append(issue("staleness", series, f"newest value {d}, behind {through}"))
    return out


# Each scheduled job's slots, as (weekdays, UTC hours) and how long after a slot a success may come.
SCHEDULE: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], dt.timedelta]] = {
    "daily": ((0, 1, 2, 3, 4), (22,), dt.timedelta(hours=3)),
    "edgar": ((0, 1, 2, 3, 4), (0, 3, 6, 9, 12, 15, 18, 21), dt.timedelta(hours=2)),
    "weekly": ((5,), (6,), dt.timedelta(hours=6)),
}


def last_slot(job: str, now: dt.datetime) -> dt.datetime:
    """The latest scheduled start of `job` at or before now (UTC)."""
    days, hours, _ = SCHEDULE[job]
    t = now.astimezone(dt.UTC).replace(minute=0, second=0, microsecond=0)
    for _ in range(24 * 8):
        if t.weekday() in days and t.hour in hours and t <= now:
            return t
        t -= dt.timedelta(hours=1)
    raise ValueError(f"no slot for {job}")  # pragma: no cover  (every job has weekly slots)


def missed_runs(store: Store, now: dt.datetime) -> List[Issue]:
    """A scheduled job with no finished run (ok or partial) since its previous slot, once the slot's grace period
    is over: the check that catches a schedule that silently stopped. An "all" run counts for daily and edgar.
    Slots from before the database's first run are not held against it."""
    out: List[Issue] = []
    runs = store.runs()
    if not runs:
        return out
    born = min(parse_iso(r["started_at"]) for r in runs)
    for job, (_, _, grace) in SCHEDULE.items():
        slot = last_slot(job, now)
        if now - slot < grace:  # the newest slot may still be running: judge the one before
            slot = last_slot(job, slot - dt.timedelta(hours=1))
        if slot < born:
            continue
        done = [
            parse_iso(r["started_at"])
            for r in runs
            if r["job"] in (job, "all") and r["status"] in ("ok", "partial") and r["finished_at"]
        ]
        last = max(done, default=None)
        if last is None or last < slot - dt.timedelta(minutes=30):
            since = f" (last {last:%Y-%m-%d %H:%M})" if last else " (never)"
            out.append(
                issue(
                    "staleness",
                    f"job {job}",
                    f"no finished run since the {slot:%Y-%m-%d %H:%M} UTC slot{since}",
                    "warning",
                )
            )
    return out


def reconcile(rows: Sequence[Mapping[str, Any]], tolerance: float) -> List[Issue]:
    return [
        issue(
            "reconcile",
            f"{r['symbol']} {r['date']}",
            f"yahoo {r['yahoo']:.4f} vs cboe {r['cboe']:.4f} ({r['diff']:+.2%}, limit {tolerance:.1%})",
        )
        for r in rows
        if not r["ok"]
    ]


def point_in_time(store: Store) -> List[Issue]:
    out = []
    names = {r["name"] for r in store.query("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    for t in [*TABLES, *APPEND_ONLY]:
        for op in ("update", "delete"):
            if f"{t}_no_{op}" not in names:
                out.append(issue("point_in_time", t, f"append-only trigger {t}_no_{op} is missing"))
    early = store.query(
        "SELECT symbol, date, source, available_at FROM prices WHERE available_at < date || 'T20:00:00Z' LIMIT 20"
    )
    out += [
        issue(
            "point_in_time",
            f"{r['symbol']} {r['date']}",
            f"{r['source']} bar available at {r['available_at']}, " "before that day's close",
        )
        for r in early
    ]
    return out


def ipo(deals: Sequence[ipos.Deal]) -> List[Issue]:
    return [
        issue("ipo", f"{p['company']} (CIK {p['cik']})", f"{p['check']}: {p['detail']}", "warning")
        for p in ipos.check(deals)
    ]
