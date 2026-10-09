"""Live stock predictions only between a close and the next open, the first model fitted on the backtest's window,
point-in-time stamps for rescaled backfilled bars, and the tolerant pop test when IPO predictions are scored."""

import datetime as dt
import json

import numpy as np
import pandas as pd
from conftest import Opener, make_cfg, make_ctx
from world import SPLIT_DAY, World, at

from bot import collect, data, markets, tracking, update
from bot.http import Http
from bot.store import available, iso

D = dt.date


def test_no_stock_predictions_during_a_session(tmp_path) -> None:
    w = World(now=at(D(2025, 10, 9), 16, 49))  # Thursday, mid-session
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    out = update.run("daily", cfg, w.now, Http(cfg, opener=w.opener, sleep=lambda s: None))
    assert out["daily"]["predictions"]["stock"] == 0 and out["status"] == "ok"  # skipped, not a warning
    early = markets.close_utc(D(2025, 10, 8)) + dt.timedelta(minutes=5)  # before the closes are published
    assert tracking.predict_stocks(make_ctx(cfg, Opener(), early)) == 0
    weekend = make_ctx(cfg, Opener(), at(D(2025, 10, 11), 12))  # Saturday: Friday's close, Monday's open ahead
    assert tracking.predict_stocks(weekend) == 0 and weekend.warnings[-1]["reason"].startswith("no closes")
    ok = make_ctx(cfg, Opener(), at(D(2025, 10, 9), 0, 30))  # after Wednesday's close, before Thursday's open
    assert tracking.predict_stocks(ok) == 15
    targets = {r["target"] for r in ok.store.query("SELECT target FROM predictions")}
    assert targets == {"2025-10-09"}


def test_first_stock_model_uses_the_backtest_window(tmp_path, monkeypatch) -> None:
    w = World(now=at(D(2025, 10, 9), 16, 49))  # mid-session: the daily run fits no stock model
    cfg = make_cfg(tmp_path, history_start="2025-01-02")
    update.run("daily", cfg, w.now, Http(cfg, opener=w.opener, sleep=lambda s: None))
    monkeypatch.setattr(tracking, "STOCK_TRAIN_DAYS", 40)
    seen = []
    fit = tracking.models.fit
    monkeypatch.setattr(tracking.models, "fit", lambda kind, df, *a, **k: seen.append(df) or fit(kind, df, *a, **k))
    ctx = make_ctx(cfg, Opener(), w.now)
    tracking._ensure_model(ctx, "stock", tracking.stock_inputs(ctx.store), tracking.features.STOCK_FEATURES)
    assert seen[-1]["date"].nunique() == 40
    note = json.loads(ctx.store.query("SELECT metrics FROM models")[0]["metrics"])["note"]
    assert "last 40 sessions" in note


def test_backfilled_bars_before_a_later_split_carry_the_read_time(tmp_path) -> None:
    nominal = dt.datetime(2018, 5, 1, 20, 15, tzinfo=dt.UTC)
    fetched = dt.datetime(2026, 10, 8, 22, tzinfo=dt.UTC)
    assert available(nominal, fetched) == iso(nominal)
    assert available(nominal, fetched, rescaled=True) == iso(fetched)
    assert available(fetched, fetched, rescaled=True) == iso(fetched)  # read live: the read time either way
    w = World(now=at(D(2025, 10, 8)))
    cfg = make_cfg(tmp_path, history_start="2025-05-01")
    ctx = make_ctx(cfg, w.opener, w.now)
    collect.update_symbol(ctx, "SPLT", markets.last_closed_session(w.now))
    rows = {r["date"]: r["available_at"] for r in ctx.store.asof("prices", where={"symbol": "SPLT"})}
    before = markets.previous_trading_day(SPLIT_DAY).isoformat()
    assert rows[before] == ctx.at  # rescaled by the split: not knowable as such on the day
    assert rows[SPLIT_DAY.isoformat()] < ctx.at  # the split day itself and later: the nominal close time
    # as of a past moment, only the unsplit-era bars knowable then are seen (none of the rescaled ones)
    past = data.closes(ctx.store, "2025-07-01T00:00:00Z", ["SPLT"])
    assert past.index.min() == SPLIT_DAY.isoformat()


def test_ipo_outcome_uses_the_tolerant_pop_test(tmp_path, monkeypatch) -> None:
    ctx = make_ctx(make_cfg(tmp_path), Opener(), at(D(2025, 3, 10)))

    class Deal:
        first_day_return = 12 / 10 - 1  # 0.19999999999999996: a 20% pop

    monkeypatch.setattr(tracking.edgar, "deals_asof", lambda store, at=None: [type("X", (Deal,), {"cik": "5"})()])
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
    assert tracking.score(ctx) == 1 and ctx.store.query("SELECT outcome FROM outcomes") == [{"outcome": 1.0}]


def test_ipos_are_predicted_only_with_an_effect_notice(tmp_path, monkeypatch) -> None:
    ctx = make_ctx(make_cfg(tmp_path), Opener(), at(D(2025, 3, 10)))
    rows = pd.DataFrame(
        {
            "cik": ["1", "2", "3"],
            "company": ["Effective", "Prospectus only", "Old"],
            "moment": ["2025-03-07", "2025-03-07", "2024-01-02"],
            "trade_date": [None, None, "2024-01-03"],
            "y": [np.nan, np.nan, 1.0],
            "ret": [np.nan, np.nan, 0.3],
        }
    )

    class Deal:
        def __init__(self, cik, effective, state):
            self.cik, self.effective, self.state = cik, effective, state

        def status(self, today):
            return self.state

    deals = [Deal("1", "2025-03-07", "effective"), Deal("2", None, "priced"), Deal("3", "2024-01-02", "trading")]

    class Model:
        model_id = "m"

        def prob(self, df):
            return np.full(len(df), 0.4)

        def value(self, df):
            return np.full(len(df), 0.1)

    monkeypatch.setattr(tracking, "ipo_inputs", lambda store, pop, at=None: rows)
    monkeypatch.setattr(tracking.edgar, "deals_asof", lambda store, at=None: deals)
    monkeypatch.setattr(tracking, "_ensure_model", lambda *a: Model())
    assert tracking.predict_ipos(ctx) == 3  # deal 1 only, by the model and both baselines
    assert {r["subject"] for r in ctx.store.query("SELECT subject FROM predictions")} == {"1"}
