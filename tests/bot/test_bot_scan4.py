"""Review fixes: leavers retried after a transient failure, renames seen by both membership sources, a reused
ticker, the shrinkage recorded and a constant model warned about, the Wikipedia budget counted in requests sent,
and the leakage test run on the membership-masked panel."""

import datetime as dt
import io
import urllib.error
from email.message import Message

import numpy as np
import pandas as pd
from conftest import Opener, make_cfg, make_ctx
from world import DELIST_DAY, World, at

from bot import collect, data, evaluate, features, health, models, tracking

D = dt.date
COLLECTING = ["AAA", "BBB", "DIVD", "HOLE", "SPLT"]


def row(sym, at_, member, source="wikipedia_history", cik=None):
    return {
        "symbol": sym,
        "source": source,
        "name": sym,
        "cik": cik,
        "sector": None,
        "exchange": None,
        "added": None,
        "member": member,
        "listed": None,
        "available_at": at_,
    }


def put(store, rows):
    for r in sorted(rows, key=lambda r: r["available_at"]):
        store.put("universe", [r], "r", r["available_at"])


# ---------------------------------------------------------------------------- leavers


def _leaver_world(tmp_path):
    w = World(now=at(D(2025, 10, 8)))
    cfg = make_cfg(tmp_path, history_start="2025-06-02")
    ctx = make_ctx(cfg, w.opener, w.now)
    collect.update_membership_history(ctx)
    return w, cfg, ctx


def test_a_leaver_is_retried_after_a_transient_failure(tmp_path) -> None:
    w, cfg, ctx = _leaver_world(tmp_path)
    w.fail["GONE"] = 503  # both sources busy for this ticker
    assert collect.backfill_leavers(ctx, COLLECTING) == 0
    assert ctx.store.cursor("leaver:GONE") == "retry:1:2025-10-09"
    assert ctx.warnings[-1] == {"what": "leavers_missing_prices", "n": 1, "symbols": ["GONE"]}
    calls = len(w.calls)
    assert collect.backfill_leavers(ctx, COLLECTING) == 0 and len(w.calls) == calls  # not due again today
    w.fail = {}
    later = make_ctx(cfg, w.opener, at(D(2025, 10, 9)), ctx.store)
    assert collect.backfill_leavers(later, COLLECTING) == 1
    assert later.store.cursor("leaver:GONE") == "yahoo" and not later.warnings
    assert later.store.last_date("prices", {"symbol": "GONE"}) == DELIST_DAY.isoformat()


def test_a_leaver_is_given_up_after_its_tries_and_warned_about(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(collect, "LEAVER_TRIES", 2)
    w, cfg, ctx = _leaver_world(tmp_path)
    w.fail["GONE"] = "timeout"
    collect.backfill_leavers(ctx, COLLECTING)
    second = make_ctx(cfg, w.opener, at(D(2025, 10, 10)), ctx.store)
    assert collect.backfill_leavers(second, COLLECTING) == 0
    assert second.store.cursor("leaver:GONE") == "gave_up"
    w.fail = {}
    third = make_ctx(cfg, w.opener, at(D(2025, 11, 10)), ctx.store)
    assert collect.backfill_leavers(third, COLLECTING) == 0  # never read again, but still a warning every run
    assert third.warnings[-1]["what"] == "leavers_missing_prices"


def test_a_failure_recorded_before_retries_existed_is_retried(tmp_path) -> None:
    w, _, ctx = _leaver_world(tmp_path)
    ctx.store.set_cursor("leaver:GONE", "failed", "old", "2025-10-01T00:00:00Z")
    w.fail["GONE"] = 429
    assert collect.backfill_leavers(ctx, COLLECTING) == 0
    assert ctx.store.cursor("leaver:GONE") == "retry:2:2025-10-10"  # the old failure counts as the first try
    assert collect._leaver_tries(None) == 0 and not collect._leaver_due("yahoo", "2025-10-08")


def test_only_a_final_failure_settles_a_leaver() -> None:
    assert collect._gone([("yahoo", "not_found: x"), ("cboe", "blocked: 403")])
    assert collect._gone([("yahoo", "schema: x")])
    assert not collect._gone([("yahoo", "not_found: x"), ("cboe", "timeout: x")])
    assert not collect._gone([("yahoo", "blocked: 403")])


# ---------------------------------------------------------------------------- renames


def test_a_rename_recorded_by_both_sources_on_different_days_is_followed(store) -> None:
    put(
        store,
        [
            row("OLD", "2020-01-15T00:00:00Z", 1, cik="9"),
            row("OLD", "2026-09-01T00:00:00Z", 1, source="wikipedia", cik="9"),
            row("OLD", "2026-09-21T22:00:00Z", 0, source="wikipedia", cik="9"),  # the daily read sees it that day
            row("NEW", "2026-09-21T22:00:00Z", 1, source="wikipedia", cik="9"),
            row("OLD", "2026-09-28T00:00:00Z", 0, cik="9"),  # the monthly snapshot a week later
            row("NEW", "2026-09-28T00:00:00Z", 1, cik="9"),
        ],
    )
    assert data.universe(store) == ["NEW"]
    assert data.membership(store, ["2025-01-02", "2026-10-01"]).to_dict("list") == {"NEW": [True, True]}


def test_a_rename_survives_the_old_ticker_being_reused(store) -> None:
    put(
        store,
        [
            row("A", "2020-01-15T00:00:00Z", 1, cik="1"),
            row("A", "2020-03-15T00:00:00Z", 0, cik="1"),
            row("B", "2020-03-15T00:00:00Z", 1, cik="1"),
            row("A", "2021-03-15T00:00:00Z", 1, cik="2"),  # another company takes the ticker
        ],
    )
    m = data.membership(store, ["2020-02-03", "2020-04-01", "2021-04-01"])
    assert m.to_dict("list") == {"A": [False, False, True], "B": [True, True, True]}


def test_a_leave_without_a_cik_on_its_row_uses_the_last_member_rows() -> None:
    rows = [
        row("A", "2020-01-15T00:00:00Z", 1, cik="1"),
        row("A", "2020-03-15T00:00:00Z", 0),
        row("B", "2020-04-01T00:00:00Z", 1, cik="0001"),  # 17 days later, inside the window
        row("C", "2020-01-15T00:00:00Z", 1),
        row("C", "2020-03-15T00:00:00Z", 0),  # no CIK anywhere: never a rename
        row("D", "2020-03-15T00:00:00Z", 1),  # no CIK on the join: not a candidate
    ]
    assert data.Membership(rows).renamed == {"A": "B"}


def test_two_joins_of_the_same_cik_are_not_a_rename() -> None:
    rows = [
        row("A", "2020-01-15T00:00:00Z", 1, cik="1"),
        row("A", "2020-03-15T00:00:00Z", 0, cik="1"),
        row("B", "2020-03-15T00:00:00Z", 1, cik="1"),
        row("C", "2020-03-20T00:00:00Z", 1, cik="1"),
    ]
    assert data.Membership(rows).renamed == {}


def test_a_rename_into_a_ticker_renamed_earlier_is_dropped() -> None:
    rows = [
        row("X", "2020-01-06T00:00:00Z", 1, cik="1"),
        row("OLD", "2019-12-01T00:00:00Z", 1, cik="1"),
        row("NEW", "2020-02-09T00:00:00Z", 1, cik="1"),
        row("NEW", "2020-02-10T00:00:00Z", 0, cik="1"),  # NEW -> X (X joined 35 days before)
        row("OLD", "2020-03-15T00:00:00Z", 0, cik="1"),  # OLD -> NEW, which is no longer a column
    ]
    mem = data.Membership(rows)
    assert mem.renamed == {"NEW": "X", "OLD": "NEW"}
    assert list(mem.mask(["2020-03-01"]).columns) == ["X"]


# ---------------------------------------------------------------------------- shrinkage


def _noise_panel(n_dates=400):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for s in range(20):
            r = rng.normal(0, 0.01)
            rows.append(
                dict(date=d, symbol=str(s), ret=r, y=float(r > 0), **{f: rng.normal() for f in features.STOCK_FEATURES})
            )
    return pd.DataFrame(rows)


def test_shrinkage_is_chosen_and_applied_around_the_same_center() -> None:
    df = _noise_panel()
    m = models.fit("stock", df, features.STOCK_FEATURES, df["date"].max())
    assert m.shrink == 0.0 and m.center == models._logit(float(df["y"].mean()))
    p = m.prob(df.head(50))
    assert np.allclose(p, p[0])  # a constant model: every prediction the base rate


def test_a_constant_model_is_recorded_and_warned_about(tmp_path, monkeypatch) -> None:
    df = _noise_panel()
    ctx = make_ctx(make_cfg(tmp_path), Opener(), at(D(2025, 1, 2)))
    m = tracking._ensure_model(ctx, "stock", df, features.STOCK_FEATURES)
    assert m is not None and m.shrink == 0.0
    assert ctx.warnings[-1]["what"] == "constant_model"
    assert '"shrink": 0.0' in ctx.store.query("SELECT metrics FROM models")[0]["metrics"]
    report = health.build(ctx)
    assert report["models"]["stock"]["shrink"] == 0.0 and report["models"]["ipo"] is None
    assert any("(constant: the base rate)" in line for line in health.markdown(report))
    monkeypatch.setattr(tracking, "stock_inputs", lambda store, at=None: df)
    (tmp_path / "fresh").mkdir()
    later = make_ctx(make_cfg(tmp_path / "fresh"), Opener(), at(D(2025, 6, 2)))
    out = tracking.retrain(later, "stock")  # no current model: the candidate is adopted
    assert out["shrink"] == 0.0 and out["adopted"] and later.warnings[-1]["what"] == "constant_model"


# ---------------------------------------------------------------------------- the Wikipedia budget


def test_the_history_budget_counts_retried_requests(tmp_path) -> None:
    w = World(now=at(D(2025, 10, 8)))
    flip = {"fail": True}

    def flaky(req, timeout):  # every Wikipedia read fails once before it answers
        if "wikipedia.org" in req.full_url:
            flip["fail"] = not flip["fail"]
            if not flip["fail"]:
                raise urllib.error.HTTPError(req.full_url, 503, "busy", Message(), io.BytesIO(b""))
        return w.opener(req, timeout)

    ctx = make_ctx(make_cfg(tmp_path, history_start="2025-01-02", wiki_history_budget=9), flaky, w.now)
    collect.update_membership_history(ctx)
    assert ctx.http.requests <= 9  # each read cost two requests: two months, not four
    assert ctx.store.cursor(data.HISTORY).startswith("2025-02|")


# ---------------------------------------------------------------------------- the leakage test, masked


def test_the_stock_leakage_test_fits_on_member_rows_only(monkeypatch) -> None:
    idx = list(pd.bdate_range("2024-01-02", periods=300).strftime("%Y-%m-%d"))
    rng = np.random.default_rng(1)
    closes = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for s in ("A", "B", "C", "SPX")}, index=idx
    )
    mask = pd.DataFrame({"A": True, "B": True, "C": False}, index=pd.Index(idx, name="date"))
    seen = []
    real = models.fit

    def spy(kind, df, feats, through, l2=1.0):
        seen.append(set(df["symbol"]))
        return real(kind, df, feats, through, l2=l2)

    monkeypatch.setattr(models, "fit", spy)
    out = evaluate.leakage_stocks(closes, pd.DataFrame(), [idx[250]], mask=mask)
    assert out["passed"] and seen and all(s == {"A", "B"} for s in seen)
