"""Edge cases and error paths the end-to-end runs do not reach on their own."""

import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from conftest import Opener, make_cfg, make_ctx
from world import World, at

from bot import checks, collect, edgar, evaluate, ipos, models, tracking, update
from bot.adapters import sec_search
from bot.http import Http, SourceError

D = dt.date


def wctx(tmp_path, w, **kw):
    cfg = make_cfg(tmp_path, **kw)
    return make_ctx(cfg, w.opener, w.now)


# ---------------------------------------------------------------------------- collection


def test_split_and_dividend_explain_big_moves() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "A",
            "date": ["2024-01-02", "2024-01-03"],
            "open": 1,
            "high": 1,
            "low": 1,
            "close": [100.0, 40.0],
            "volume": 1,
        }
    )
    assert checks.moves(bars, {"A": {"2024-01-03": 2.5}}, {}, 0.5) == []


def test_listing_files_down_and_duplicates(tmp_path, world: World) -> None:
    world.fail["nasdaqtrader"] = 404
    world.schema["duplicate"] = True
    ctx = wctx(tmp_path, world, history_start="2025-09-01")
    syms = collect.update_universe(ctx)
    collect.update_prices(ctx, syms)
    collect.update_macro(ctx)
    whats = [w["what"] for w in ctx.warnings]
    assert "listings_not_refreshed" in whats and whats.count("duplicate_rows") >= 2


# ---------------------------------------------------------------------------- EDGAR


def test_search_pages(tmp_path) -> None:
    def page(url):
        off = int(url.split("from=")[1])
        if off >= 200:
            return SourceError("http", "500")
        hits = [
            {
                "_id": f"0000000001-25-{off + i:06d}:a.htm",
                "_source": {
                    "ciks": ["1"],
                    "display_names": ["X  (CIK 1)"],
                    "form": "S-1",
                    "file_date": "2025-01-02",
                    "adsh": f"0000000001-25-{off + i:06d}",
                },
            }
            for i in range(100)
        ]
        return json.dumps({"hits": {"total": {"value": 250}, "hits": hits}}).encode()

    ctx = make_ctx(make_cfg(tmp_path), Opener({"efts": page}), at(D(2025, 1, 3)))
    first = sec_search.parse(page("from=0"))[0]
    rows = edgar._search_all(ctx, D(2025, 1, 2), list(first), 250)
    assert len(rows) == 200  # the third page failed: the two read are kept


def test_backfill_stops_at_history_start_and_without_cursor(tmp_path, world: World) -> None:
    ctx = wctx(tmp_path, world, history_start="2025-07-15")
    budget = edgar.Budget(ctx)
    assert edgar.backfill(ctx, budget) == []  # no cursor yet: the daily step sets it
    ctx.store.set_cursor("edgar_backfill", "2025-10-01", "r", ctx.at)
    assert edgar.backfill(ctx, budget) == ["2025Q3"]  # and not 2025Q2: before history_start's quarter
    world.fail["full-index"] = 500
    ctx.store.set_cursor("edgar_backfill", "2025-04-01", "r", ctx.at)
    ctx2 = wctx(tmp_path, world, history_start="2024-01-02")
    assert edgar.backfill(ctx2, edgar.Budget(ctx2)) == [] and ctx2.warnings[0]["what"] == "backfill_failed"


def test_budget_runs_out(tmp_path, world: World) -> None:
    ctx = wctx(tmp_path, world, sec_budget=60)
    out = edgar.update(ctx)
    assert out["sec_requests"] <= 60 + 2
    ctx.store.close()


def test_company_lookups_fail_and_page(tmp_path) -> None:
    recent = {"form": ["S-1"], "filingDate": ["2025-03-01"], "accessionNumber": ["a"], "primaryDocument": ["a.htm"]}
    routes = {
        "CIK0000000005.json": json.dumps(
            {
                "name": "Five",
                "sic": "2834",
                "tickers": [],
                "filings": {"recent": recent, "files": [{"name": "CIK0000000005-submissions-001.json"}]},
            }
        ).encode(),
        "submissions-001": json.dumps({"form": ["10-Q", "8-K"], "filingDate": ["2019-05-01", "2018-01-01"]}).encode(),
        "CIK0000000006.json": SourceError("http", "500"),
    }
    ctx = make_ctx(make_cfg(tmp_path), Opener(routes), at(D(2025, 3, 5)))
    ctx.store.put(
        "filings",
        [
            {
                "accession": "a",
                "cik": "5",
                "form": "S-1",
                "filed": "2025-03-01",
                "company": "Five",
                "ticker": "",
                "sic": None,
                "file": "a.htm",
            },
            {
                "accession": "b",
                "cik": "6",
                "form": "S-1",
                "filed": "2025-03-02",
                "company": "Six",
                "ticker": "",
                "sic": None,
                "file": "b.htm",
            },
        ],
        "r",
        ctx.at,
    )
    assert edgar.companies(ctx, edgar.Budget(ctx)) == 1
    co = ctx.store.asof("companies")[0]
    assert co["first_report"] == "2019-05-01"  # found on an older page: a follow-on
    assert ctx.warnings[0]["what"] == "company_failed"
    deal = edgar.deals_asof(ctx.store)[0]
    assert deal.ipo is False
    ctx.cfg = dataclasses.replace(ctx.cfg, sec_budget=0)
    assert edgar.companies(ctx, edgar.Budget(ctx)) == 0 and edgar.documents(ctx, edgar.Budget(ctx)) == 0


def test_document_read_fails(tmp_path) -> None:
    ctx = make_ctx(make_cfg(tmp_path), Opener({"edgar/data": SourceError("http", "500")}), at(D(2025, 3, 5)))
    ctx.store.put(
        "filings",
        [
            {
                "accession": "0000000007-25-000001",
                "cik": "7",
                "form": "S-1",
                "filed": "2025-03-01",
                "company": "Seven",
                "ticker": "",
                "sic": None,
                "file": "s.htm",
            }
        ],
        "r",
        ctx.at,
    )
    assert edgar.documents(ctx, edgar.Budget(ctx)) == 0 and ctx.warnings[0]["what"] == "document_failed"


def test_first_trade_window_and_splits(tmp_path, world: World) -> None:
    ctx = wctx(tmp_path, world)
    end = D(2025, 10, 8)
    old = ipos.Deal(cik="1", company="Old", foreign=False, registered="2025-05-01", priced="2025-06-02", symbol="AAA")
    assert edgar._first_trade(ctx, old, D(2025, 5, 26), end) is None  # trading long before: an uplisting
    early = ipos.Deal(cik="2", company="S", foreign=False, registered="2022-12-01", priced="2023-01-03", symbol="SPLT")
    day, row = edgar._first_trade(ctx, early, D(2022, 12, 27), end)
    raw = world.raw["SPLT"][D(2023, 1, 3)][3]
    assert day == D(2023, 1, 3) and row["close"] == pytest.approx(raw, abs=1e-3)  # the later 2-for-1 split is undone
    late = ipos.Deal(cik="3", company="L", foreign=False, registered="2022-10-01", priced="2022-11-01", symbol="BBB")
    assert edgar._first_trade(ctx, late, D(2022, 10, 25), end) is None  # first bar months after pricing
    none = ipos.Deal(cik="4", company="N", foreign=False, registered="2025-01-02", priced="2025-02-03", symbol="NOPE")
    world.fail["cdn.cboe.com"] = 403
    assert edgar._first_trade(ctx, none, D(2025, 1, 27), end) is None


def test_old_unfound_deals_are_retried_weekly(tmp_path, world: World) -> None:
    ctx = wctx(tmp_path, world)
    ctx.store.put(
        "filings",
        [
            {
                "accession": f"0000000009-24-00000{i}",
                "cik": "9",
                "form": f,
                "filed": d,
                "company": "Nine",
                "ticker": "",
                "sic": None,
                "file": "x.htm",
            }
            for i, (f, d) in enumerate((("S-1", "2024-01-02"), ("424B4", "2024-02-01")))
        ],
        "r",
        ctx.at,
    )
    ctx.store.put(
        "documents",
        [{"accession": "0000000009-24-000001", "data": json.dumps({"offer": 10, "symbol": "NOPE"})}],
        "r",
        ctx.at,
    )
    edgar.first_trades(ctx)
    calls = len(world.calls)
    edgar.first_trades(ctx)
    assert len(world.calls) == calls  # tried today: not again until next week


# ---------------------------------------------------------------------------- tracking


def test_ipo_predictions_made_before_listing_then_scored(tmp_path) -> None:
    w = World(now=at(D(2025, 10, 8)))
    deal = next(
        d for d in w.deals if d.trade and d.trade > D(2025, 6, 1) and not d.follow_on and "Acq" not in d.company
    )
    cfg = make_cfg(tmp_path)
    before = dt.datetime.combine(deal.trade, dt.time(12), tzinfo=dt.UTC)  # the morning it lists
    w.now = before
    out = update.run("all", cfg, before, Http(cfg, opener=w.opener, sleep=lambda s: None))
    assert out["edgar"]["predictions"]["ipo"] >= 3 or out["daily"]["predictions"]["ipo"] >= 3
    w.now = at(deal.trade + dt.timedelta(days=1))
    out = update.run("edgar", cfg, w.now, Http(cfg, opener=w.opener, sleep=lambda s: None))
    assert out["edgar"]["scored"] >= 3


def test_withdrawn_ipo_prediction_is_void(tmp_path) -> None:
    ctx = make_ctx(make_cfg(tmp_path), Opener(), at(D(2025, 3, 10)))
    f = [("S-1", "2025-01-02", "a"), ("EFFECT", "2025-02-03", "b"), ("RW", "2025-02-20", "c")]
    ctx.store.put(
        "filings",
        [
            {
                "accession": acc,
                "cik": "5",
                "form": form,
                "filed": d,
                "company": "Five",
                "ticker": "",
                "sic": None,
                "file": "x",
            }
            for form, d, acc in f
        ],
        "r",
        ctx.at,
    )
    ctx.store.record(
        "predictions",
        [
            {
                "pred_id": "ipo:5:m",
                "model": "m",
                "kind": "ipo",
                "subject": "5",
                "event": "first_day_pop",
                "target": "first_trade",
                "prob": 0.3,
                "value": 0.1,
                "made_at": "2025-02-04T00:00:00Z",
                "data": {},
                "run_id": "r",
            }
        ],
    )
    assert tracking.score(ctx) == 1 and ctx.store.query("SELECT outcome FROM outcomes") == [{"outcome": None}]


def test_model_exists_but_no_close_today(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    update.run("daily", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))
    ctx = make_ctx(cfg, world.opener, at(D(2025, 10, 9)))  # a day later, before that day's prices are read
    assert tracking.predict_stocks(ctx) == 0 and ctx.warnings[-1]["reason"].startswith("no closes")


def test_watch_with_only_baselines_resolved(store) -> None:
    store.record(
        "predictions",
        [
            {
                "pred_id": "p",
                "model": "base_rate",
                "kind": "stock",
                "subject": "A",
                "event": "e",
                "target": "2025-01-03",
                "prob": 0.5,
                "value": 0,
                "made_at": "x",
                "data": {},
                "run_id": "r",
            }
        ],
    )
    store.record("outcomes", [{"pred_id": "p", "outcome": 1.0, "value": 0.0, "resolved_at": "x", "run_id": "r"}])
    out = tracking.watch(store, "stock", 3, 0.1)
    assert len(out["periods"]) == 1 and out["warnings"] == []


def test_retrain_when_fitting_fails(tmp_path, world: World, monkeypatch) -> None:
    cfg = make_cfg(tmp_path)
    update.run("all", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))
    ctx = make_ctx(cfg, world.opener, at(D(2025, 11, 20)))
    for day in (D(2025, 10, 31), D(2025, 11, 20)):
        world.now = at(day)
        update.run("daily", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))

    def fail(*a, **k):
        raise ValueError("one class only")

    monkeypatch.setattr(models, "fit", fail)
    assert tracking.retrain(ctx, "stock") == {"adopted": False, "reason": "one class only"}


# ---------------------------------------------------------------------------- evaluation


def test_newton_stops_at_its_iteration_limit() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert np.isfinite(models.fit_logit(x, np.array([0, 0, 1, 1.0]), iters=1)).all()


def test_leakage_cutoff_too_early_to_fit_and_ipo_without_macro() -> None:
    idx = [f"2024-01-{d:02d}" for d in (2, 3, 4, 5, 8, 9, 10)]
    closes = pd.DataFrame({"A": np.linspace(10, 11, 7), "SPX": np.linspace(100, 101, 7)}, index=idx)
    out = evaluate.leakage_stocks(closes, pd.DataFrame(), [idx[3]])
    assert out["passed"] and out["checks"][0]["prediction_diff"] == 0.0
    assert evaluate.leakage_ipos([], pd.DataFrame(), ["2024-01-01"], 0.2)["passed"]


def test_a_failed_leakage_test_is_a_warning(tmp_path, world: World, monkeypatch) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    update.run("daily", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))
    monkeypatch.setattr(evaluate, "leakage_stocks", lambda *a, **k: {"passed": False, "worst_diff": 1.0, "checks": []})
    out = update.run("weekly", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))
    assert out["status"] == "partial" and not out["weekly"]["leakage_passed"]
    assert "FAILED" in (tmp_path / "reports" / "backtest.md").read_text()


def test_older_pages_without_a_report(tmp_path) -> None:
    recent = {"form": ["S-1"], "filingDate": ["2025-03-03"], "accessionNumber": ["c"], "primaryDocument": ["c.htm"]}
    eight = {
        "name": "Eight",
        "sic": "2834",
        "tickers": [],
        "filings": {"recent": recent, "files": [{"name": "eight-a.json"}, {"name": "eight-b.json"}]},
    }
    routes = {
        "CIK0000000008.json": json.dumps(eight).encode(),
        "eight-a": json.dumps({"form": ["8-K"], "filingDate": ["2010-01-01"]}).encode(),
        "eight-b": SourceError("http", "500"),
    }
    ctx = make_ctx(make_cfg(tmp_path), Opener(routes), at(D(2025, 3, 5)))
    row = {
        "accession": "c",
        "cik": "8",
        "form": "S-1",
        "filed": "2025-03-03",
        "company": "Eight",
        "ticker": "",
        "sic": None,
        "file": "c.htm",
    }
    ctx.store.put("filings", [row], "r", ctx.at)
    assert edgar.companies(ctx, edgar.Budget(ctx)) == 1
    assert ctx.store.asof("companies")[0]["first_report"] is None  # no report on any page read: a first-time filer


def test_dividend_after_listing_is_not_a_split(tmp_path, world: World) -> None:
    ctx = wctx(tmp_path, world)
    div = ipos.Deal(cik="5", company="D", foreign=False, registered="2022-12-01", priced="2023-01-03", symbol="DIVD")
    _, row = edgar._first_trade(ctx, div, D(2022, 12, 27), D(2025, 10, 8))
    assert row["close"] == pytest.approx(world.raw["DIVD"][D(2023, 1, 3)][3])


def test_too_little_history_for_a_first_model(tmp_path, world: World) -> None:
    cfg = make_cfg(tmp_path, history_start="2025-10-06", symbols=("AAA",))
    update.run("daily", cfg, world.now, Http(cfg, opener=world.opener, sleep=lambda s: None))
    ctx = make_ctx(cfg, world.opener, world.now)
    assert tracking.predict_stocks(ctx) == 0 and ctx.warnings[-1]["what"] == "no_model"


def test_filings_outside_a_deal_are_ignored() -> None:
    fs = [
        {"accession": a, "cik": "7", "form": f, "filed": d, "company": "Acme", "sic": None}
        for a, f, d in (("0", "RW", "2024-12-01"), ("1", "S-1", "2025-01-02"), ("2", "S-1MEF", "2025-02-12"))
    ]
    deal = ipos.build(fs, {}, {}, {})[0]
    assert deal.withdrawn is None and deal.status(D(2025, 3, 1)) == "on_file"
