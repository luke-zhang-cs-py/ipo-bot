"""Reading the store back as analysis-ready frames, always as of a moment."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bot.store import Store

SOURCE_ORDER = ("yahoo", "cboe")  # per day, the first source that has a bar is the one used


def bars(store: Store, at: Optional[str] = None, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Daily bars as knowable at `at`: one row per (symbol, date), Yahoo's when it has one, else Cboe's.
    Columns: symbol, date, open, high, low, close, volume, source, available_at."""
    where = {"symbol": sorted(set(symbols))} if symbols is not None else None
    rows = store.asof("prices", at, where=where)
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume", "source", "available_at"]
        )
    df = pd.DataFrame(rows).drop(columns=["run_id"])
    df["rank"] = df["source"].map({s: i for i, s in enumerate(SOURCE_ORDER)}).fillna(len(SOURCE_ORDER))
    df = df.sort_values(["symbol", "date", "rank"]).drop_duplicates(["symbol", "date"]).drop(columns=["rank"])
    return df.reset_index(drop=True)


def closes(store: Store, at: Optional[str] = None, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Closes as a date x symbol frame (dates as ISO text, sorted), as knowable at `at`."""
    b = bars(store, at, symbols)
    if b.empty:
        return pd.DataFrame()
    return b.pivot(index="date", columns="symbol", values="close").sort_index()


def macro(store: Store, at: Optional[str] = None) -> pd.DataFrame:
    """Macro series as a date x series frame, as knowable at `at` (the first source per day wins)."""
    rows = store.asof("macro", at)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["series", "date", "available_at"]).drop_duplicates(["series", "date"])
    return df.pivot(index="date", columns="series", values="value").sort_index()


HISTORY = "wikipedia_history"  # monthly membership snapshots backfilled from the page's revision history
MEMBERSHIP_SOURCES = ("wikipedia", HISTORY, "config")


def _membership_rows(store: Store, at: Optional[str]) -> List[Dict[str, Any]]:
    """Every version of every membership row known by `at`, oldest first."""
    q = (
        "SELECT symbol, source, cik, added, member, available_at FROM universe WHERE source IN (?, ?, ?)"
        + (" AND available_at <= ?" if at is not None else "")
        + " ORDER BY available_at, rowid"
    )
    return store.query(q, (*MEMBERSHIP_SOURCES, *([at] if at is not None else [])))


RENAME_WINDOW = 35  # days between a ticker's leave and the same CIK's join under another for it to be a rename


class Membership:
    """Who was in the index on each day, from the dated membership rows (Wikipedia's monthly snapshots and the
    bot's own daily reads). A symbol's state on a day is its newest row dated on or before it. Before the first
    snapshot of all, the "Date added" column stands in (a member if added by then; with no date, if listed in
    that first snapshot).

    A rename is followed: a ticker leaving (its first non-member row after a member one) and the same CIK (read
    from that leave row) joining under another ticker within RENAME_WINDOW days of it, either side. The window
    lets the two sources disagree on the day: the daily read sees a rename on the day, the monthly snapshot at
    the next revision. The old ticker's days before the leave count for the new one, whose price history covers
    them; a ticker later reused by another company keeps only its days after the leave."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.config = {r["symbol"] for r in rows if r["source"] == "config" and r["member"] == 1}
        self.events: Dict[str, List[Tuple[str, bool]]] = {}
        self.added: Dict[str, str] = {}
        ciks: Dict[str, List[Optional[str]]] = {}
        for r in rows:
            if r["source"] == "config":
                continue
            self.events.setdefault(r["symbol"], []).append((r["available_at"][:10], r["member"] == 1))
            ciks.setdefault(r["symbol"], []).append(str(r["cik"]).lstrip("0") if r["cik"] else None)
            if r["added"]:
                self.added[r["symbol"]] = r["added"]
        self.first = min((ev[0][0] for ev in self.events.values()), default=None)
        self.renames = self._renames(ciks)
        self.renamed = {old: new for old, new, _ in self.renames}  # old ticker -> new (the latest rename)

    def _renames(self, ciks: Mapping[str, List[Optional[str]]]) -> List[Tuple[str, str, str]]:
        """(old ticker, new ticker, the day the old one left), oldest leave first."""
        leaves: List[Tuple[str, str, str]] = []  # (symbol, day, cik)
        joins: List[Tuple[str, str, str]] = []
        for sym, ev in self.events.items():
            for i, (day, flag) in enumerate(ev):
                if flag and (i == 0 or not ev[i - 1][1]):
                    if ciks[sym][i]:
                        joins.append((sym, day, str(ciks[sym][i])))
                elif not flag and i > 0 and ev[i - 1][1]:
                    cik = ciks[sym][i] or ciks[sym][i - 1]  # the leave row's own CIK, else the last member row's
                    if cik:
                        leaves.append((sym, day, cik))
        out = []
        for old, day, cik in sorted(leaves, key=lambda x: (x[1], x[0])):
            d0 = dt.date.fromisoformat(day)
            new = {
                s
                for s, d, c in joins
                if s != old and c == cik and abs((dt.date.fromisoformat(d) - d0).days) <= RENAME_WINDOW
            }
            if len(new) == 1:
                out.append((old, new.pop(), day))
        return out

    def symbols(self) -> List[str]:
        """Every symbol that was a member on some day (renamed tickers under their new name, unless the old
        ticker was a member again after its last rename)."""
        ever = {s for s, ev in self.events.items() if any(f for _, f in ev)}
        last = {old: until for old, _, until in self.renames}
        gone = {s for s, until in last.items() if not any(f and d >= until for d, f in self.events[s])}
        return sorted((ever | self.config) - gone)

    def _target(self, new: str, until: str) -> str:
        """Where a rename's days end up: the new ticker, or the name it was renamed to later (and so on)."""
        for old, nxt, when in self.renames:
            if old == new and when > until:
                return self._target(nxt, when)
        return new

    def _state(self, sym: str, days: np.ndarray) -> np.ndarray:
        ev = self.events.get(sym, [])
        when = np.array([d for d, _ in ev], dtype=str)
        flags = np.array([f for _, f in ev], dtype=bool)
        i = np.searchsorted(when, days, side="right") - 1 if len(ev) else np.full(len(days), -1)
        known = flags[np.maximum(i, 0)] if len(ev) else np.zeros(len(days), dtype=bool)
        added = self.added.get(sym)
        before = days < self.first if self.first is not None else np.ones(len(days), dtype=bool)
        in_first = bool(ev) and ev[0] == (self.first, True)  # listed in the first snapshot of all
        fallback = before & ((days >= added) if added else in_first)
        return np.where(i >= 0, known, fallback)

    def mask(self, dates: Sequence[str]) -> pd.DataFrame:
        """A date x symbol frame: True where the symbol was a member that day."""
        days = np.array(list(dates), dtype=str)
        cols: Dict[str, np.ndarray] = {}
        for s in self.symbols():
            cols[s] = np.ones(len(days), dtype=bool) if s in self.config else self._state(s, days)
            for old, _, until in self.renames:
                if old == s:  # its days before a rename belong to the new ticker
                    cols[s] = cols[s] & (days >= until)
        for old, new, until in self.renames:
            target = self._target(new, until)
            if target in cols:
                since = max((w for o, _, w in self.renames if o == old and w < until), default="")
                cols[target] = cols[target] | (self._state(old, days) & (days >= since) & (days < until))
        return pd.DataFrame(cols, index=pd.Index(list(dates), name="date"), dtype=bool)


def membership(store: Store, dates: Sequence[str], at: Optional[str] = None) -> pd.DataFrame:
    """Point-in-time index membership as a date x symbol frame of booleans, from the rows known by `at`."""
    return Membership(_membership_rows(store, at)).mask(dates)


def universe(store: Store, at: Optional[str] = None) -> List[str]:
    """Every symbol that was an index member (or a configured symbol) at some point, as known by `at`."""
    return Membership(_membership_rows(store, at)).symbols()


def members(store: Store, at: Optional[str] = None) -> List[str]:
    """Symbols in the universe at `at` (None: now), less those known delisted. Now, that is the newest read of
    the index (or the configured list), else the newest monthly snapshot; at a past moment, the membership on
    that day as recorded by then (monthly snapshots, then the "Date added" column before the first one)."""
    rows = store.asof("universe", at)
    delisted = {r["symbol"] for r in rows if r["source"] == "nasdaqtrader" and r["listed"] == 0}
    member = {r["symbol"] for r in rows if r["source"] in ("wikipedia", "config") and r["member"] == 1}
    if at is not None or not member:
        day = at[:10] if at is not None else "9999-12-31"
        m = membership(store, [day], at)
        member = {s for s in m.columns if m.iloc[0][s]}
    return sorted(member - delisted)


def splits(store: Store, at: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """symbol -> {date: ratio} of the splits on record at `at`."""
    out: Dict[str, Dict[str, float]] = {}
    for r in store.asof("actions", at, where={"kind": "split"}):
        out.setdefault(r["symbol"], {})[r["date"]] = float(r["value"])
    return out


def dividends(store: Store, at: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in store.asof("actions", at, where={"kind": "dividend"}):
        out.setdefault(r["symbol"], {})[r["date"]] = float(r["value"])
    return out
