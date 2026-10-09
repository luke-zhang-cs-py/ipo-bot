"""SQLite storage: append-only, point-in-time, with run metadata, and the lock that stops two updates overlapping.

Every data table is versioned. A row is keyed (a symbol and a date, say) and carries `available_at`, the moment
the bot could first have known it. A source that later changes a value (a split adjusts every older price, a
filing is amended) adds a new version; the first release stays. Reading "as of" a moment takes, per key, the
newest version available by then, so a backtest only ever sees what was knowable on the day. Triggers refuse
every UPDATE and DELETE, so history cannot be overwritten even by mistake.

The stamping rule for prices (see available()): a bar read soon after its close is stamped when it was read. A
bar read long after (a backfill) is stamped with its nominal publication time (16:00 New York plus the source's
delay) only if no split on record falls after its date. A bar with a later split has been rescaled by the source
(NVDA's 2018 close as served today reflects the 2021 and 2024 splits), so the number stored was not knowable on
its day: it is stamped with the time it was read. Readers that ask as of a past moment then see only bars as
they could have been known (features are log differences within one version, so a missing older version only
shortens the history they see, never mixes scales).

Membership snapshots backfilled from Wikipedia's page history are stamped with the revision's own timestamp:
that revision was public then, so its table was knowable at that moment.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import pathlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

Row = Dict[str, Any]


@dataclass(frozen=True)
class Table:
    """A versioned table: rows are identified by `key`; a change in any of `values` is a new version."""

    name: str
    key: Tuple[str, ...]
    values: Tuple[str, ...]
    types: Mapping[str, str]


def _t(name: str, key: Sequence[str], values: Sequence[str], **types: str) -> Table:
    return Table(name, tuple(key), tuple(values), types)


TABLES: Dict[str, Table] = {
    t.name: t
    for t in (
        # daily bars, as each source reports them (split-adjusted by the source as of available_at)
        _t(
            "prices",
            ["symbol", "date", "source"],
            ["open", "high", "low", "close", "volume"],
            open="REAL",
            high="REAL",
            low="REAL",
            close="REAL",
            volume="REAL",
        ),
        # splits (value = new shares per old share) and cash dividends (value = dollars per share)
        _t("actions", ["symbol", "date", "source", "kind"], ["value"], value="REAL"),
        # macro series: Treasury yields, the VIX
        _t("macro", ["series", "date", "source"], ["value"], value="REAL"),
        # who is in the universe: S&P 500 membership, listing status (member and listed are 0 or 1)
        _t(
            "universe",
            ["symbol", "source"],
            ["name", "cik", "sector", "exchange", "added", "member", "listed"],
            member="INTEGER",
            listed="INTEGER",
        ),
        # a company's EDGAR record: SIC code, tickers, its first annual or quarterly report (None: none yet)
        _t("companies", ["cik"], ["name", "sic", "tickers", "first_report"]),
        # SEC filings of the IPO pipeline (S-1, S-1/A, F-1, F-1/A, EFFECT, RW, 424B4)
        _t("filings", ["accession"], ["cik", "form", "filed", "company", "ticker", "sic", "file"]),
        # what the prospectus parsers read from a filing's document (JSON: range, offer, shares, lead, symbol)
        _t("documents", ["accession"], ["data"]),
        # an IPO's first trading day, read from a price source
        _t("first_trades", ["cik"], ["symbol", "date", "open", "close", "source"], open="REAL", close="REAL"),
        # where an incremental job got to (the EDGAR backfill's last day, say)
        _t("cursors", ["name"], ["value"]),
    )
}

# Tables that are written once per key, never versioned: predictions, their outcomes, trained models.
LEDGERS = """
CREATE TABLE IF NOT EXISTS predictions (pred_id TEXT PRIMARY KEY, model TEXT NOT NULL, kind TEXT NOT NULL,
    subject TEXT NOT NULL, event TEXT NOT NULL, target TEXT NOT NULL, prob REAL, value REAL,
    made_at TEXT NOT NULL, data TEXT, run_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outcomes (pred_id TEXT PRIMARY KEY REFERENCES predictions(pred_id), outcome REAL,
    value REAL, resolved_at TEXT NOT NULL, run_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS models (model_id TEXT PRIMARY KEY, kind TEXT NOT NULL, trained_at TEXT NOT NULL,
    params TEXT NOT NULL, metrics TEXT NOT NULL, adopted INTEGER NOT NULL, run_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, job TEXT NOT NULL, started_at TEXT NOT NULL,
    finished_at TEXT, status TEXT, version TEXT, summary TEXT);
"""
APPEND_ONLY = ("predictions", "outcomes", "models")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(t: dt.datetime) -> str:
    """A UTC moment as sortable text: 2026-10-08T22:00:00Z."""
    return t.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


LIVE_WINDOW = dt.timedelta(days=3)


def available(nominal: dt.datetime, fetched: dt.datetime, rescaled: bool = False) -> str:
    """When a value became knowable. Read soon after it was published, that is when the bot read it (never
    earlier than publication). Read long after (a backfill), the read time says nothing about the past, so the
    nominal publication time stands in (a close at 16:00 New York plus the source's delay), unless the value has
    since been rescaled (rescaled=True: a split on record after its date): then it is the read time, because
    the number as served was not what anyone could see on the day."""
    if fetched - nominal <= LIVE_WINDOW:
        return iso(max(nominal, fetched))
    return iso(fetched if rescaled else nominal)


class AppendOnlyError(Exception):
    """An attempt to change or delete history."""


class Store:
    """The bot's database. Rows go in with put() (versioned tables) or record() (ledgers), and come out as of a
    moment with asof()."""

    def __init__(self, path: pathlib.Path | str):
        self.path = pathlib.Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL" if str(path) != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self._create()

    def close(self) -> None:
        self.db.close()

    def _create(self) -> None:
        ddl = [LEDGERS]
        for t in TABLES.values():
            cols = [f"{c} TEXT NOT NULL" for c in t.key] + [f"{c} {t.types.get(c, 'TEXT')}" for c in t.values]
            ddl.append(
                f"CREATE TABLE IF NOT EXISTS {t.name} ({', '.join(cols)}, available_at TEXT NOT NULL, run_id TEXT NOT NULL);"
            )
            ddl.append(f"CREATE INDEX IF NOT EXISTS {t.name}_key ON {t.name} ({', '.join(t.key)}, available_at);")
        for name in [*TABLES, *APPEND_ONLY]:
            for op in ("UPDATE", "DELETE"):
                ddl.append(
                    f"CREATE TRIGGER IF NOT EXISTS {name}_no_{op.lower()} BEFORE {op} ON {name} "
                    f"BEGIN SELECT RAISE(ABORT, 'append-only: {op} on {name}'); END;"
                )
        # a run's row is finished once (its end time and status filled in) and never touched again
        ddl.append(
            "CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs WHEN OLD.finished_at IS NOT NULL "
            "BEGIN SELECT RAISE(ABORT, 'append-only: a finished run'); END;"
        )
        ddl.append(
            "CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs "
            "BEGIN SELECT RAISE(ABORT, 'append-only: DELETE on runs'); END;"
        )
        self.db.executescript("\n".join(ddl))

    # ------------------------------------------------------------------ runs

    def start_run(self, run_id: str, job: str, version: str, at: Optional[dt.datetime] = None) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO runs (run_id, job, started_at, version) VALUES (?, ?, ?, ?)",
                (run_id, job, iso(at or utc_now()), version),
            )

    def finish_run(
        self, run_id: str, status: str, summary: Mapping[str, Any], at: Optional[dt.datetime] = None
    ) -> None:
        try:
            with self.db:
                self.db.execute(
                    "UPDATE runs SET finished_at = ?, status = ?, summary = ? WHERE run_id = ?",
                    (iso(at or utc_now()), status, json.dumps(summary, sort_keys=True, default=str), run_id),
                )
        except sqlite3.IntegrityError as e:
            raise AppendOnlyError(str(e)) from e

    def runs(self, job: Optional[str] = None) -> List[Row]:
        q = "SELECT * FROM runs" + (" WHERE job = ?" if job else "") + " ORDER BY started_at, run_id"
        return [dict(r) for r in self.db.execute(q, (job,) if job else ())]

    # ------------------------------------------------------------------ versioned tables

    def put(self, table: str, rows: Iterable[Mapping[str, Any]], run_id: str, available_at: str) -> int:
        """Add the rows that are new or changed since their latest version; returns how many went in. Putting the
        same rows twice adds nothing (updates are idempotent).

        `available_at` is when the batch was read. A first release may carry its own, earlier available_at (the
        nominal publication time of backfilled history); a revision of a stored row never does: it became
        knowable only when it was read, so it gets at least the batch's time."""
        t = TABLES[table]
        rows = list(rows)
        if not rows:
            return 0
        latest = self._latest_for(t, {r[t.key[0]] for r in rows})
        fresh: List[Tuple[Any, ...]] = []
        for r in rows:
            k = tuple(str(r[c]) for c in t.key)
            v = tuple(r.get(c) for c in t.values)
            if latest.get(k) != v:
                own = r.get("available_at") or available_at
                when = own if k not in latest else max(own, available_at)
                latest[k] = v
                fresh.append((*k, *v, when, run_id))
        if fresh:
            cols = [*t.key, *t.values, "available_at", "run_id"]
            with self.db:
                self.db.executemany(
                    f"INSERT INTO {t.name} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", fresh
                )
        return len(fresh)

    def _latest_for(self, t: Table, firsts: Iterable[Any]) -> Dict[Tuple[str, ...], Tuple[Any, ...]]:
        out: Dict[Tuple[str, ...], Tuple[Any, ...]] = {}
        firsts = sorted({str(f) for f in firsts})
        for i in range(0, len(firsts), 500):
            chunk = firsts[i : i + 500]
            q = f"SELECT * FROM {t.name} WHERE {t.key[0]} IN ({', '.join('?' * len(chunk))}) ORDER BY rowid"
            for r in self.db.execute(q, chunk):
                out[tuple(r[c] for c in t.key)] = tuple(r[c] for c in t.values)
        return out

    def asof(
        self, table: str, at: Optional[str] = None, where: Optional[Mapping[str, Any]] = None, first: bool = False
    ) -> List[Row]:
        """Per key, the newest version available by `at` (None: everything stored so far). first=True takes each
        key's first release instead."""
        t = TABLES[table]
        conds, args = [], []
        if at is not None:
            conds.append("available_at <= ?")
            args.append(at)
        for c, v in (where or {}).items():
            if isinstance(v, (list, tuple, set)):
                vals = list(v)
                conds.append(f"{c} IN ({', '.join('?' * len(vals))})")
                args += [str(x) for x in vals]
            else:
                conds.append(f"{c} = ?")
                args.append(str(v))
        cond = (" WHERE " + " AND ".join(conds)) if conds else ""
        pick = "min" if first else "max"
        q = (
            f"SELECT * FROM {t.name} WHERE rowid IN (SELECT {pick}(rowid) FROM {t.name}{cond} GROUP BY {', '.join(t.key)}) "
            f"ORDER BY {', '.join(t.key)}"
        )
        return [dict(r) for r in self.db.execute(q, args)]

    def versions(self, table: str, where: Mapping[str, Any]) -> List[Row]:
        """Every version of the matching rows, oldest first."""
        t = TABLES[table]
        cond = " AND ".join(f"{c} = ?" for c in where)
        return [
            dict(r)
            for r in self.db.execute(
                f"SELECT * FROM {t.name} WHERE {cond} ORDER BY rowid", [str(v) for v in where.values()]
            )
        ]

    def last_date(self, table: str, where: Mapping[str, Any], column: str = "date") -> Optional[str]:
        """The latest `column` among the matching rows (where an incremental fetch picks up), or None."""
        cond = " AND ".join(f"{c} = ?" for c in where)
        r = self.db.execute(
            f"SELECT max({column}) FROM {table}" + (f" WHERE {cond}" if cond else ""), [str(v) for v in where.values()]
        ).fetchone()
        return r[0] if r and r[0] else None

    def cursor(self, name: str) -> Optional[str]:
        rows = self.asof("cursors", where={"name": name})
        return rows[0]["value"] if rows else None

    def set_cursor(self, name: str, value: str, run_id: str, at: str) -> None:
        self.put("cursors", [{"name": name, "value": value}], run_id, at)

    # ------------------------------------------------------------------ ledgers

    def record(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """Add ledger rows (predictions, outcomes, models) whose id is new; returns how many went in. A row that is
        already there is left as it was: a prediction, once made, is never revised."""
        if table not in APPEND_ONLY:
            raise KeyError(table)
        rows = list(rows)
        if not rows:
            return 0
        cols = list(rows[0])
        enc = [
            tuple(json.dumps(r[c], sort_keys=True) if isinstance(r[c], (dict, list)) else r[c] for c in cols)
            for r in rows
        ]
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", enc
            )
        return self.db.total_changes - before

    def query(self, sql: str, args: Sequence[Any] = ()) -> List[Row]:
        return [dict(r) for r in self.db.execute(sql, args)]


# ---------------------------------------------------------------------- the lock


class LockedError(Exception):
    """Another update holds the lock."""


@contextlib.contextmanager
def lock(path: pathlib.Path, stale_hours: float, now: Optional[dt.datetime] = None) -> Iterator[Optional[str]]:
    """Hold the update lock for the block. A lock left by a run that died (older than stale_hours) is taken over;
    the context value is then a warning saying so, otherwise None. A live lock raises LockedError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or utc_now()
    warning = None
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
            since = dt.datetime.fromisoformat(held["since"].replace("Z", "+00:00"))
        except (OSError, ValueError, KeyError):
            since, held = dt.datetime.fromtimestamp(0, dt.UTC), {}
        age = (now - since).total_seconds() / 3600
        if age < stale_hours:
            raise LockedError(
                f"another update holds {path} (pid {held.get('pid')}, since {held.get('since')})"
            ) from None
        warning = f"took over a stale lock from {held.get('since', 'an unreadable lock file')} ({age:.1f} h old)"
        path.unlink()
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "since": iso(now)}, f)
    try:
        yield warning
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
