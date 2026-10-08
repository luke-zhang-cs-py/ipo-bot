"""End to end, offline: whole updates against the synthetic world (tests/bot/world.py), never the network."""

import dataclasses
import datetime as dt
import json
import sqlite3

import pytest
from conftest import make_cfg
from world import DELIST_DAY, SPLIT_DAY, World, at

from bot import data, update
from bot.http import Http
from bot.store import LockedError, Store, lock

D = dt.date


def run(job, cfg, w, now=None):
    if now is not None:
        w.now = now
    return update.run(job, cfg, w.now, Http(cfg, opener=w.opener, sleep=lambda s: None))


def count(cfg, table):
    db = sqlite3.connect(cfg.data_dir / "bot.sqlite")
    try:
        return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        db.close()


def test_fresh_update_then_idempotent_rerun(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path)
    out = run("all", cfg, world)
    assert out["status"] in ("ok", "partial") and out["daily"]["symbols"] == 5
    assert out["daily"]["predictions"]["stock"] == 15  # 5 members x (model + 2 baselines)
    assert out["edgar"]["first_trades"] > 50 and count(cfg, "filings") > 400
    health = json.loads((tmp_path / "health" / "latest.json").read_text())
    assert health["coverage"]["missing_tickers"] == [] and health["coverage"]["current"] == 5
    assert (tmp_path / "reports" / "health.md").exists() and (tmp_path / "tracking" / "predictions.jsonl").exists()
    tables = ("prices", "macro", "filings", "documents", "predictions", "first_trades", "universe", "models")
    before = {t: count(cfg, t) for t in tables}
    calls = len(world.calls)
    again = run("all", cfg, world)  # the same moment again: nothing new
    assert {t: count(cfg, t) for t in tables} == before
    assert again["daily"]["price_sources"] == {"current": 6}  # nothing re-read: every symbol is up to date
    assert len(world.calls) - calls < 120
    lines = (tmp_path / "tracking" / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == before["predictions"]  # the committed mirror has each prediction once


def test_next_day_reads_only_new_data_and_scores(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path)
    run("all", cfg, world)
    world.calls.clear()
    out = run("daily", cfg, world, at(D(2025, 10, 9)))
    yahoo = [u for u in world.calls if "finance.yahoo" in u]
    assert all("period1=" in u for u in yahoo)
    starts = {int(u.split("period1=")[1].split("&")[0]) for u in yahoo}
    first = dt.datetime.fromtimestamp(min(starts), dt.UTC).date()
    assert first >= D(2025, 9, 30)  # a few sessions back for revisions, not the whole history
    assert out["daily"]["scored"] == 15  # yesterday's predictions resolved
    st = Store(cfg.data_dir / "bot.sqlite")
    outcomes = st.query("SELECT outcome FROM outcomes")
    assert all(o["outcome"] in (0.0, 1.0) for o in outcomes)
    st.close()


def test_split_rescales_history_as_a_new_version(tmp_path) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    w = World(now=at(D(2025, 5, 28)))
    run("daily", cfg, w)
    st = Store(cfg.data_dir / "bot.sqlite")
    before = st.asof("prices", where={"symbol": "SPLT", "source": "yahoo", "date": "2025-05-01"})[0]["close"]
    st.close()
    run("daily", cfg, w, at(D(2025, 6, 3)))  # the 2-for-1 split is known now
    st = Store(cfg.data_dir / "bot.sqlite")
    versions = st.versions("prices", {"symbol": "SPLT", "source": "yahoo", "date": "2025-05-01"})
    assert [v["close"] for v in versions] == [before, pytest.approx(before / 2)]  # first release kept
    old_view = st.asof("prices", "2025-05-30T00:00:00Z", where={"symbol": "SPLT", "date": "2025-05-01"})
    assert old_view[0]["close"] == before  # as of before the split, the bot saw the old scale
    assert st.asof("actions", where={"symbol": "SPLT", "kind": "split"})[0]["date"] == SPLIT_DAY.isoformat()
    issues = json.loads((tmp_path / "health" / "latest.json").read_text())["issues"]["all"]
    assert not [i for i in issues if i["check"] == "moves" and "SPLT" in i["subject"]]
    st.close()


def test_delisted_stock_is_kept(tmp_path) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    w = World(now=at(D(2025, 9, 26)))
    run("daily", cfg, w)
    run("daily", cfg, w, at(D(2025, 11, 14)))
    st = Store(cfg.data_dir / "bot.sqlite")
    assert "GONE" not in data.members(st)
    gone = st.asof("prices", where={"symbol": "GONE"})
    assert gone and max(r["date"] for r in gone) == DELIST_DAY.isoformat()  # its history stays
    uni = {r["symbol"]: r for r in st.asof("universe", where={"source": "wikipedia"})}
    assert uni["GONE"]["member"] == 0
    listed = {r["symbol"]: r for r in st.asof("universe", where={"source": "nasdaqtrader"})}
    assert listed["GONE"]["listed"] == 0
    health = json.loads((tmp_path / "health" / "latest.json").read_text())
    assert health["coverage"]["delisted_kept"] == 1
    st.close()


def test_sources_going_down_degrade_with_warnings(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    run("all", cfg, world)
    world.fail["query1.finance.yahoo.com"] = 503
    world.fail["home.treasury.gov"] = "timeout"
    world.fail["en.wikipedia.org"] = 429
    world.fail["daily-index"] = 500
    out = run("all", cfg, world, at(D(2025, 10, 9)))
    assert out["status"] == "partial"
    assert out["daily"]["price_sources"].get("cboe") == 6  # Cboe stood in for every symbol
    assert out["daily"]["macro"]["UST10Y"] == "fred"
    whats = {w["what"] for w in json.loads((tmp_path / "health" / "latest.json").read_text())["run_warnings"]}
    assert {"prices_fallback", "treasury_failed", "universe_not_refreshed", "filings_fallback"} <= whats
    st = Store(cfg.data_dir / "bot.sqlite")
    assert len(data.members(st)) == 5  # the universe last read is kept
    st.close()


def test_no_universe_ever_read_invents_none(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    world.fail["en.wikipedia.org"] = 429
    out = run("daily", cfg, world)
    assert out["daily"]["symbols"] == 0 and out["daily"]["predictions"]["stock"] == 0
    st = Store(cfg.data_dir / "bot.sqlite")
    assert data.members(st) == [] and {r["symbol"] for r in st.asof("prices")} == {"SPX"}
    st.close()


def test_schema_changes_are_caught_not_stored(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    world.schema.update(yahoo=True, cboe_vix=True, wikipedia=True)
    out = run("daily", cfg, world)
    assert out["status"] == "partial"
    warnings = json.loads((tmp_path / "health" / "latest.json").read_text())["run_warnings"]
    assert any("schema" in json.dumps(w) for w in warnings)
    assert count(cfg, "macro") > 0  # VIX fell back to FRED; yields still came from the Treasury
    st = Store(cfg.data_dir / "bot.sqlite")
    assert {r["source"] for r in st.asof("macro") if r["series"] == "VIX"} == {"fred"}
    st.close()


def test_everything_down_still_finishes_and_says_so(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-06-02", symbols=("AAA",))
    for host in ("yahoo", "cboe", "treasury", "fred", "nasdaqtrader", "sec.gov"):
        world.fail[host] = "reset"
    out = run("all", cfg, world)
    assert out["status"] == "partial" and out["health_status"] == "failing"
    health = json.loads((tmp_path / "health" / "latest.json").read_text())
    assert health["coverage"]["missing_tickers"] == ["AAA"]


def test_no_sec_user_agent_skips_edgar_only(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-06-02", sec_user_agent=None)
    out = run("all", cfg, world)
    assert out["edgar"] == {"skipped": True, "scored": 0, "predictions": {"ipo": 0}}
    assert out["daily"]["predictions"]["stock"] > 0
    assert not [u for u in world.calls if "sec.gov" in u]


def test_lock_blocks_an_overlapping_run(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path)
    with lock(tmp_path / "update.lock", 6, world.now), pytest.raises(LockedError):
        run("daily", cfg, world)
    (tmp_path / "update.lock").write_text(json.dumps({"pid": 1, "since": "2025-10-01T00:00:00Z"}))
    out = run("weekly", cfg, world)  # a lock a week old is from a run that died
    assert out["status"] == "partial"


def test_crash_marks_the_run_failed(tmp_path, world: World, monkeypatch) -> None:
    cfg = make_cfg(tmp_path)

    def boom(ctx):
        raise RuntimeError("disk full")

    monkeypatch.setattr(update, "weekly", boom)
    with pytest.raises(RuntimeError):
        run("weekly", cfg, world)
    st = Store(cfg.data_dir / "bot.sqlite")
    r = st.runs("weekly")[0]
    assert r["status"] == "failed" and "disk full" in r["summary"]
    st.close()
    with pytest.raises(ValueError):
        update.run("hourly", cfg)


def test_weekly_backtest_and_monthly_retrain(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path)
    run("all", cfg, world)
    out = run("weekly", cfg, world)
    assert out["weekly"]["leakage_passed"] and out["weekly"]["ipo_n"] > 20
    text = (tmp_path / "reports" / "backtest.md").read_text()
    assert "Diebold-Mariano" in text and "Leakage test" in text and "| model vs base: brier |" in text
    res = json.loads((tmp_path / "backtest" / "latest.json").read_text())
    assert res["ipo"]["beats_best_baseline"]  # the synthetic IPOs carry a real signal (the range revision)
    assert not res["stock"]["beats_best_baseline"]  # the synthetic stocks are random walks: no edge to find
    out = run("monthly", cfg, world)
    assert "waiting" in out["monthly"]["stock"]["reason"]  # no out-of-sample outcomes yet for the current model
    for day in (D(2025, 10, 31), D(2025, 11, 10)):
        run("all", cfg, world, at(day))
    out = run("monthly", cfg, world, at(D(2025, 11, 10), 23))
    stock = out["monthly"]["stock"]
    assert stock["holdout_from"] > "2025-10-08" and stock["current"] and isinstance(stock["adopted"], bool)
    st = Store(cfg.data_dir / "bot.sqlite")
    assert len(st.query("SELECT * FROM models WHERE kind = 'stock'")) == 2
    st.close()


def test_record_then_replay_gives_the_same_database(tmp_path, world: World) -> None:
    rec = tmp_path / "recorded"
    cfg_a = make_cfg(tmp_path / "a", record_dir=rec, history_start="2025-06-02")
    run("all", cfg_a, world)
    cfg_b = make_cfg(tmp_path / "b", replay_dir=rec, history_start="2025-06-02")
    out = update.run("all", cfg_b, world.now)  # the real client, reading only the recorded files
    assert out["status"] in ("ok", "partial")
    for t in ("prices", "macro", "filings", "documents", "first_trades"):
        assert count(cfg_a, t) == count(cfg_b, t), t
    sa, sb = Store(cfg_a.data_dir / "bot.sqlite"), Store(cfg_b.data_dir / "bot.sqlite")
    pa = sa.query("SELECT pred_id, prob FROM predictions ORDER BY pred_id")
    pb = sb.query("SELECT pred_id, prob FROM predictions ORDER BY pred_id")
    sa.close()
    sb.close()
    assert [p["pred_id"].rsplit(":", 1)[0] for p in pa] == [p["pred_id"].rsplit(":", 1)[0] for p in pb]
    assert [p["prob"] for p in pa] == [p["prob"] for p in pb]  # deterministic


@pytest.mark.parametrize(
    "now",
    [
        at(D(2025, 3, 10), 1),  # the night DST starts (still Sunday evening in New York)
        at(D(2025, 11, 3), 3),  # the morning after DST ends
        at(D(2025, 7, 4), 22),  # a holiday
        at(D(2025, 12, 27), 12),  # a Saturday
    ],
)
def test_runs_on_awkward_moments(tmp_path, now) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    out = run("all", cfg, World(now=now))
    assert out["status"] in ("ok", "partial")
    health = json.loads((tmp_path / "health" / "latest.json").read_text())
    assert health["coverage"]["stale"] == []


def test_symbols_override(tmp_path, world: World) -> None:
    cfg = dataclasses.replace(make_cfg(tmp_path, history_start="2025-06-02"), symbols=("AAA", "BBB"))
    out = run("daily", cfg, world)
    assert out["daily"]["symbols"] == 2 and not [u for u in world.calls if "wikipedia" in u]
