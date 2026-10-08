"""The store: versions, first releases, point-in-time reads, append-only triggers, run metadata and the lock."""

import datetime as dt
import json
import sqlite3

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bot.store import AppendOnlyError, LockedError, Store, available, iso, lock, parse_iso

T1, T2, T3 = "2024-01-02T21:15:00Z", "2024-01-05T22:00:00Z", "2024-02-01T22:00:00Z"


def bar(close: float, date: str = "2024-01-02", symbol: str = "AAA") -> dict:
    return {
        "symbol": symbol,
        "date": date,
        "source": "yahoo",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": close,
        "volume": 10.0,
    }


def test_versions_keep_the_first_release(store: Store) -> None:
    assert store.put("prices", [bar(1.5)], "r", T1) == 1
    assert store.put("prices", [bar(1.5)], "r", T2) == 0  # unchanged: nothing new
    assert store.put("prices", [bar(0.75)], "r", T3) == 1  # rescaled after a split: a new version
    assert store.asof("prices", "2024-01-04T00:00:00Z")[0]["close"] == 1.5
    assert store.asof("prices")[0]["close"] == 0.75
    assert store.asof("prices", first=True)[0]["close"] == 1.5
    assert store.asof("prices", "2024-01-01T00:00:00Z") == []
    assert [v["close"] for v in store.versions("prices", {"symbol": "AAA"})] == [1.5, 0.75]


def test_row_available_at_wins_and_where_filters(store: Store) -> None:
    store.put("prices", [{**bar(1.0, symbol="A"), "available_at": T3}, bar(2.0, symbol="B")], "r", T1)
    assert [r["symbol"] for r in store.asof("prices", T2)] == ["B"]
    assert [r["symbol"] for r in store.asof("prices", where={"symbol": ["A", "B"]})] == ["A", "B"]
    assert [r["symbol"] for r in store.asof("prices", where={"symbol": "A"})] == ["A"]
    assert store.last_date("prices", {"symbol": "B"}) == "2024-01-02"
    assert store.last_date("prices", {}) == "2024-01-02"
    assert store.last_date("prices", {"symbol": "Z"}) is None
    assert store.put("prices", [], "r", T1) == 0


def test_history_cannot_be_changed(store: Store) -> None:
    store.put("prices", [bar(1.0)], "r", T1)
    store.start_run("r", "daily", "1.0")  # (a trigger fires per row: an empty table has nothing to protect)
    store.record(
        "predictions",
        [
            dict(
                pred_id="p",
                model="m",
                kind="stock",
                subject="A",
                event="up",
                target="t",
                prob=0.5,
                value=0.0,
                made_at=T1,
                data={"a": 1},
                run_id="r",
            )
        ],
    )
    for sql in (
        "UPDATE prices SET close = 2",
        "DELETE FROM prices",
        "UPDATE predictions SET prob = 1",
        "DELETE FROM predictions",
        "DELETE FROM runs",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.db.execute(sql)


def test_ledgers_never_revise(store: Store) -> None:
    row = dict(
        pred_id="p",
        model="m",
        kind="stock",
        subject="A",
        event="up",
        target="t",
        prob=0.5,
        value=None,
        made_at=T1,
        data={"x": [1]},
        run_id="r",
    )
    assert store.record("predictions", [row]) == 1
    assert store.record("predictions", [{**row, "prob": 0.9}]) == 0
    assert store.query("SELECT prob, data FROM predictions") == [{"prob": 0.5, "data": '{"x": [1]}'}]
    assert store.record("outcomes", []) == 0
    with pytest.raises(KeyError):
        store.record("prices", [row])


def test_runs_finish_once(store: Store) -> None:
    store.start_run("r1", "daily", "1.0", dt.datetime(2024, 1, 2, 22, tzinfo=dt.UTC))
    store.start_run("r2", "edgar", "1.0")
    store.finish_run("r1", "ok", {"n": 1}, dt.datetime(2024, 1, 2, 22, 5, tzinfo=dt.UTC))
    with pytest.raises(AppendOnlyError):
        store.finish_run("r1", "failed", {})
    assert [r["run_id"] for r in store.runs("daily")] == ["r1"]
    assert json.loads(store.runs("daily")[0]["summary"]) == {"n": 1}
    assert len(store.runs()) == 2


def test_cursors(store: Store) -> None:
    assert store.cursor("x") is None
    store.set_cursor("x", "a", "r", T1)
    store.set_cursor("x", "b", "r", T2)
    assert store.cursor("x") == "b"


def test_memory_store() -> None:
    s = Store(":memory:")
    assert s.put("macro", [{"series": "VIX", "date": "2024-01-02", "source": "cboe", "value": 13.0}], "r", T1) == 1
    s.close()


def test_available_uses_the_read_time_only_when_live() -> None:
    close = dt.datetime(2024, 1, 2, 21, 15, tzinfo=dt.UTC)
    assert available(close, close + dt.timedelta(hours=1)) == "2024-01-02T22:15:00Z"
    assert available(close, close - dt.timedelta(hours=1)) == "2024-01-02T21:15:00Z"  # never before publication
    assert available(close, close + dt.timedelta(days=30)) == "2024-01-02T21:15:00Z"  # a backfill
    assert parse_iso(iso(close)) == close


def test_lock(tmp_path) -> None:
    path = tmp_path / "update.lock"
    now = dt.datetime(2024, 1, 2, 22, tzinfo=dt.UTC)
    with lock(path, 6, now) as warning:
        assert warning is None and path.exists()
        with pytest.raises(LockedError), lock(path, 6, now + dt.timedelta(hours=1)):
            pass
        with lock(path, 6, now + dt.timedelta(hours=7)) as w2:  # the first holder died long ago
            assert w2 and "stale lock" in w2
        path.write_text("garbage")
        with lock(path, 6, now) as w3:  # an unreadable lock file counts as stale
            assert "unreadable" in str(w3)
    assert not path.exists()


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(st.sampled_from(["A", "B"]), st.integers(1, 5), st.floats(0.01, 1000)), min_size=1, max_size=30)
)
def test_put_is_idempotent_and_asof_is_monotone(rows) -> None:
    s = Store(":memory:")
    for i, (sym, day, close) in enumerate(rows):
        s.put("prices", [bar(close, f"2024-01-0{day}", sym)], "r", f"2024-02-{i % 28 + 1:02d}T00:00:00Z")
    n = s.query("SELECT count(*) AS n FROM prices")[0]["n"]
    latest = {(r["symbol"], r["date"]): r["close"] for r in s.asof("prices")}
    s.put("prices", [bar(c, d, sym) for (sym, d), c in latest.items()], "r", "2024-03-01T00:00:00Z")
    assert s.query("SELECT count(*) AS n FROM prices")[0]["n"] == n  # re-putting the latest values adds nothing
    expected = {}
    for sym, day, close in rows:
        expected[(sym, f"2024-01-0{day}")] = close
    assert latest == expected
    assert len(s.asof("prices", first=True)) == len(latest)
    s.close()
