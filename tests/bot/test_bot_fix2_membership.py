"""Point-in-time S&P 500 membership: monthly snapshots from Wikipedia's page history (answered by the synthetic
world's revision API, never the network), the membership mask the stock panel is filtered with, and the price
backfill of past members."""

import dataclasses
import datetime as dt
import json

import pytest
from conftest import Opener, make_cfg, make_ctx
from world import DELIST_DAY, HOLE_DAY, World, at

from bot import collect, data, tracking, update
from bot.adapters import wikipedia, wikipedia_revisions
from bot.adapters.base import SchemaError
from bot.http import Http

D = dt.date


def run(job, cfg, w, now=None):
    if now is not None:
        w.now = now
    return update.run(job, cfg, w.now, Http(cfg, opener=w.opener, sleep=lambda s: None))


def page(head, rows, table='<table class="wikitable sortable">'):
    html = table + "<tbody><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
    html += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
    return json.dumps({"parse": {"text": html}}).encode()


# ---------------------------------------------------------------------------- the adapters


def test_old_layouts_parse_through_the_aliases() -> None:
    body = page(
        ["Ticker symbol", "Company", "Date first added<sup>[3]</sup>", "CIK"],
        [["MMM", "3M", "", "0000066740"], ["ABT", "Abbott", "1964-03-31", ""], ["X"]],
    )
    rows = wikipedia.parse(body, min_members=2)
    assert rows == [
        {"symbol": "MMM", "name": "3M", "cik": "66740", "sector": None, "added": None},
        {"symbol": "ABT", "name": "Abbott", "cik": None, "sector": None, "added": "1964-03-31"},
    ]
    with pytest.raises(SchemaError, match="columns"):
        wikipedia.parse(page(["Company", "CIK"], [["a", "1"]]), min_members=1)
    with pytest.raises(SchemaError, match="no constituents"):
        wikipedia.parse(json.dumps({"parse": {"text": "<table class='other'></table>"}}).encode())
    assert "oldid=42" in wikipedia.url(oldid=42) and "page=List" in wikipedia.url()


def test_revision_lookup() -> None:
    assert "rvstart=2016-01-01T00:00:00Z" in wikipedia_revisions.url(at="2016-01-01T00:00:00Z")
    one = {"query": {"pages": [{"revisions": [{"revid": 7, "timestamp": "2015-12-30T01:02:03Z"}]}]}}
    assert wikipedia_revisions.parse(json.dumps(one).encode()) == [{"revid": 7, "timestamp": "2015-12-30T01:02:03Z"}]
    assert wikipedia_revisions.parse(json.dumps({"query": {"pages": [{"title": "x"}]}}).encode()) == []
    with pytest.raises(SchemaError):
        wikipedia_revisions.parse(json.dumps({"query": {"pages": []}}).encode())


# ---------------------------------------------------------------------------- the backfill


def test_monthly_snapshots_are_backfilled_once_and_point_in_time(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path)
    ctx = make_ctx(cfg, world.opener, world.now)
    stored = collect.update_membership_history(ctx)
    assert stored > 0 and not ctx.warnings
    assert ctx.store.cursor(data.HISTORY).startswith("2025-10|")
    rows = ctx.store.versions("universe", {"symbol": "HOLE", "source": data.HISTORY})
    assert rows[0]["member"] == 1 and rows[0]["available_at"] > HOLE_DAY.isoformat()  # joined after HOLE_DAY
    gone = ctx.store.versions("universe", {"symbol": "GONE", "source": data.HISTORY})
    assert [r["member"] for r in gone][-1] == 1  # the 2025-09-20 revision still lists it (delisted 09-30)
    # a month later the October revision drops it
    world.now = at(D(2025, 11, 21))
    ctx2 = make_ctx(cfg, world.opener, world.now, ctx.store)
    collect.update_membership_history(ctx2)
    gone = ctx.store.versions("universe", {"symbol": "GONE", "source": data.HISTORY})
    assert gone[-1]["member"] == 0 and gone[-1]["available_at"].startswith("2025-10-20")
    calls = len(world.calls)
    assert collect.update_membership_history(ctx2) == 0 and len(world.calls) == calls  # nothing missing


def test_budget_pauses_and_the_next_run_resumes(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, wiki_history_budget=9)
    ctx = make_ctx(cfg, world.opener, world.now)
    collect.update_membership_history(ctx)
    assert ctx.store.cursor(data.HISTORY).split("|")[0] < "2025-10"
    assert len([u for u in world.calls if "wikipedia" in u]) <= 9
    for _ in range(10):
        collect.update_membership_history(ctx)
    assert ctx.store.cursor(data.HISTORY).startswith("2025-10|")
    # no revision before the first one (2023-01-20): January 2023 costs one request and stores nothing
    assert all(r["available_at"] >= "2023-01-20" for r in ctx.store.asof("universe", where={"source": data.HISTORY}))


def test_a_broken_revision_is_skipped_and_a_failure_stops(tmp_path, world: World) -> None:
    world.broken_revs = {5003}  # April 2023's table is unreadable
    cfg = make_cfg(tmp_path)
    ctx = make_ctx(cfg, world.opener, world.now)
    world.fail["oldid=5010"] = 503  # the server fails on the November 2023 revision
    collect.update_membership_history(ctx)
    whats = [w["what"] for w in ctx.warnings]
    assert whats == ["membership_snapshot_skipped", "membership_history_failed"]
    assert ctx.store.cursor(data.HISTORY).startswith("2023-11|")  # stopped before December, to retry
    world.fail = {"action=query": 500}
    collect.update_membership_history(ctx)
    assert ctx.warnings[-1]["what"] == "membership_history_failed"
    world.fail = {}
    collect.update_membership_history(ctx)
    assert ctx.store.cursor(data.HISTORY).startswith("2025-10|")


# ---------------------------------------------------------------------------- the membership mask


def row(sym, at_, member, source="wikipedia_history", cik=None, added=None):
    return {
        "symbol": sym,
        "source": source,
        "name": sym,
        "cik": cik,
        "sector": None,
        "exchange": None,
        "added": added,
        "member": member,
        "listed": None,
        "available_at": at_,
    }


def put(store, rows):
    for r in sorted(rows, key=lambda r: r["available_at"]):
        store.put("universe", [r], "r", r["available_at"])


def test_mask_follows_snapshots_renames_and_date_added(store) -> None:
    put(
        store,
        [
            row("OLD", "2020-01-15T00:00:00Z", 1, cik="9", added="2001-01-01"),
            row("STAY", "2020-01-15T00:00:00Z", 1, added="2020-03-01"),
            row("NOADD", "2020-01-15T00:00:00Z", 1),
            row("OLD", "2020-06-15T00:00:00Z", 0, cik="9"),
            row("NEW", "2020-06-15T00:00:00Z", 1, cik="0009"),  # the same company under a new ticker
            row("JOIN", "2020-06-15T00:00:00Z", 1),
            row("LEFT", "2020-01-15T00:00:00Z", 1, cik="4"),
            row("LEFT", "2020-03-15T00:00:00Z", 0, cik="4"),
            row("CLS", "2020-06-15T00:00:00Z", 1, cik="4"),  # same CIK, but joined on another day: no rename
        ],
    )
    days = ["2020-01-02", "2020-02-03", "2020-04-01", "2020-07-01"]
    m = data.membership(store, days)
    assert list(m.columns) == ["CLS", "JOIN", "LEFT", "NEW", "NOADD", "STAY"]  # OLD is followed under NEW
    assert m["NEW"].tolist() == [True, True, True, True]  # OLD's days (and its date added) count for NEW
    assert m["STAY"].tolist() == [False, True, True, True]  # before the first snapshot: added 2020-03-01
    assert m["NOADD"].tolist() == [True, True, True, True]  # no date added: a member before the first snapshot
    assert m["JOIN"].tolist() == [False, False, False, True]
    assert m["LEFT"].tolist() == [True, True, False, False]  # in the first snapshot: a member before it too
    assert m["CLS"].tolist() == [False, False, False, True]
    assert data.members(store, "2020-04-01T22:00:00Z") == ["NOADD", "OLD", "STAY"]  # the rename not yet known
    assert data.members(store) == ["CLS", "JOIN", "NEW", "NOADD", "STAY"]  # no live read yet: the newest snapshot
    assert "JOIN" not in data.membership(store, days, at="2020-02-01T00:00:00Z").columns  # not known yet
    assert data.universe(store, at="2019-01-01T00:00:00Z") == []


def test_a_ticker_renamed_twice_and_a_duplicate_row(tmp_path, store) -> None:
    put(
        store,
        [
            row("A", "2020-01-15T00:00:00Z", 1, cik="1"),
            row("A", "2020-03-15T00:00:00Z", 0, cik="1"),
            row("B", "2020-03-15T00:00:00Z", 1, cik="1"),
            row("B", "2020-05-15T00:00:00Z", 0, cik="1"),
            row("C", "2020-05-15T00:00:00Z", 1, cik="1"),
        ],
    )
    assert data.membership(store, ["2020-02-03", "2020-04-01"]).to_dict("list") == {"C": [True, True]}
    ctx = make_ctx(make_cfg(tmp_path), Opener(), at(D(2025, 1, 2)))
    snap = [{"symbol": "X", "name": "X", "cik": None, "sector": None, "added": None}] * 2
    assert collect._put_snapshot(ctx, snap, "2025-01-01T00:00:00Z") == 1  # a symbol listed twice counts once


def test_config_symbols_are_always_members(store) -> None:
    put(store, [row("AAA", "2025-01-01T00:00:00Z", 1, source="config")])
    assert data.membership(store, ["2016-01-04"]).to_dict("list") == {"AAA": [True]}
    assert data.members(store) == ["AAA"] and data.members(store, "2016-01-04T22:00:00Z") == []


def test_panel_rows_only_while_a_member(tmp_path) -> None:
    w = World(now=at(D(2025, 10, 8)))
    cfg = make_cfg(tmp_path, history_start="2024-10-01")
    run("daily", cfg, w)
    ctx = make_ctx(cfg, Opener(), w.now)
    panel = tracking.stock_inputs(ctx.store)
    hole = panel[panel["symbol"] == "HOLE"]["date"]
    assert hole.min() > HOLE_DAY.isoformat() > panel["date"].min()  # the joiner only after it joined
    gone = panel[(panel["symbol"] == "GONE") & panel["r1"].notna()]
    assert gone["date"].max() == DELIST_DAY.isoformat()


# ---------------------------------------------------------------------------- leavers


def test_past_members_are_backfilled_once(tmp_path) -> None:
    w = World(now=at(D(2025, 10, 8)))
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    ctx = make_ctx(cfg, w.opener, w.now)
    collect.update_membership_history(ctx)
    ctx.store.put("universe", [row("ZZZ", ctx.at, 1)], "r", "2025-09-25T00:00:00Z")  # a ticker no source knows
    assert collect.backfill_leavers(ctx, ["AAA", "BBB", "DIVD", "HOLE", "SPLT"]) == 1  # GONE read, ZZZ not
    assert not ctx.warnings and ctx.store.cursor("leaver:ZZZ") == "failed"
    assert ctx.store.last_date("prices", {"symbol": "GONE"}) == DELIST_DAY.isoformat()
    calls = len(w.calls)
    assert collect.backfill_leavers(ctx, []) == 5 and len(w.calls) > calls  # the rest, never ZZZ or GONE again
    assert collect.backfill_leavers(ctx, []) == 0
    symbols = make_ctx(dataclasses.replace(cfg, symbols=("AAA",)), Opener(), w.now, ctx.store)
    assert collect.backfill_leavers(symbols, []) == 0


def test_daily_reports_leavers(tmp_path) -> None:
    w = World(now=at(D(2025, 12, 8)))
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    out = run("daily", cfg, w)
    assert out["daily"]["leavers_read"] == 1  # GONE left more than a month ago: read once as a past member
