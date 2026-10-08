"""Unit tests for the checks, IPO deals, statistics, features, models, tracking and settings."""

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import Opener, make_cfg, make_ctx
from hypothesis import given, settings
from hypothesis import strategies as st

from bot import checks, config, evaluate, features, ipos, models, prospectus, stats, tracking
from bot.store import Store

D = dt.date
NOW = dt.datetime(2025, 10, 8, 22, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------- checks


def frame(rows):
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def test_dedupe() -> None:
    rows, issues = checks.dedupe([{"a": 1, "v": 1}, {"a": 1, "v": 2}, {"a": 2, "v": 3}], ("a",))
    assert [r["v"] for r in rows] == [2, 3] and issues[0]["check"] == "duplicates"


def test_price_checks() -> None:
    bars = frame(
        [
            ("A", "2024-01-02", 1, 2, 0.5, 1.5, 1),
            ("A", "2024-01-03", 1, 2, 0.5, -1, 1),
            ("A", "2024-01-04", 1, 1, 2, 1.5, 1),
            ("A", "2024-01-05", 1, 2, 1, 3, 1),
        ]
    )
    out = checks.prices(bars)
    assert sorted({i["subject"] for i in out}) == ["A 2024-01-03", "A 2024-01-04", "A 2024-01-05"]
    assert checks.prices(frame([])) == []


def test_moves_explained_by_actions() -> None:
    bars = frame(
        [
            ("A", "2024-01-02", 1, 1, 1, 100, 1),
            ("A", "2024-01-03", 1, 1, 1, 50, 1),  # unadjusted 2:1 split
            ("A", "2024-01-04", 1, 1, 1, 20, 1),  # a 60% special dividend
            ("A", "2024-01-05", 1, 1, 1, 80, 1),
        ]
    )  # unexplained
    out = checks.moves(bars, {"A": {"2024-01-03": 2.0}}, {"A": {"2024-01-04": 30.0}}, 0.5)
    assert [i["subject"] for i in out] == ["A 2024-01-05"]


def test_missing_days_and_staleness(cfg) -> None:
    bars = frame(
        [("A", "2025-10-01", 1, 1, 1, 1, 1), ("A", "2025-10-03", 1, 1, 1, 1, 1), ("B", "2025-10-08", 1, 1, 1, 1, 1)]
    )
    gaps = checks.missing_days(bars, D(2025, 10, 8))
    assert gaps == {"A": ["2025-10-02"]} and checks.gap_issues(gaps)[0]["subject"] == "A"
    many = checks.gap_issues({"A": [f"2025-01-0{i}" for i in range(1, 8)]})
    assert many[0]["detail"].endswith("...")
    out = checks.staleness(
        bars, ["A", "B", "C"], {"VIX": "2025-10-08", "UST10Y": None, "UST2Y": "2025-09-01"}, NOW, cfg
    )
    assert {i["subject"] for i in out} == {"A", "C", "UST10Y", "UST2Y"}
    assert checks.staleness(frame([]), ["A"], {}, NOW, cfg)[0]["detail"] == "no prices at all"
    assert checks.lag(None, D(2025, 1, 2)) is None and checks.lag("2025-01-02", D(2025, 1, 2)) == 0


def test_missed_runs(store: Store) -> None:
    assert checks.missed_runs(store, NOW) == []  # a database with no runs yet
    store.start_run("a", "all", "1", dt.datetime(2025, 10, 6, 22, tzinfo=dt.UTC))
    store.finish_run("a", "ok", {})
    late = dt.datetime(2025, 10, 8, 2, tzinfo=dt.UTC)  # past the 7 Oct 22:00 slot's grace
    subjects = {i["subject"] for i in checks.missed_runs(store, late)}
    assert "job daily" in subjects and "job edgar" in subjects and "job weekly" not in subjects
    store.start_run("b", "daily", "1", dt.datetime(2025, 10, 7, 22, tzinfo=dt.UTC))
    store.finish_run("b", "partial", {})
    assert "job daily" not in {i["subject"] for i in checks.missed_runs(store, late)}
    assert checks.last_slot("weekly", NOW).weekday() == 5


def test_reconcile_and_point_in_time(store: Store) -> None:
    rows = [
        {"symbol": "A", "date": "d", "yahoo": 10.0, "cboe": 10.2, "diff": 0.02, "ok": False},
        {"symbol": "B", "date": "d", "yahoo": 10.0, "cboe": 10.0, "diff": 0.0, "ok": True},
    ]
    assert [i["subject"] for i in checks.reconcile(rows, 0.005)] == ["A d"]
    assert checks.point_in_time(store) == []
    store.put(
        "prices",
        [
            {
                "symbol": "A",
                "date": "2025-10-08",
                "source": "yahoo",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "available_at": "2025-10-08T10:00:00Z",
            }
        ],
        "r",
        "x",
    )
    store.db.execute("DROP TRIGGER prices_no_delete")
    out = checks.point_in_time(store)
    assert {i["subject"] for i in out} == {"prices", "A 2025-10-08"}


# ---------------------------------------------------------------------------- IPO deals


def filing(acc, cik, form, filed, company="Acme Inc."):
    return {"accession": acc, "cik": cik, "form": form, "filed": filed, "company": company, "sic": None}


def test_deal_assembly_and_status() -> None:
    fs = [
        filing("1", "7", "S-1", "2025-01-02"),
        filing("2", "7", "S-1/A", "2025-01-20"),
        filing("3", "7", "S-1/A", "2025-02-03"),
        filing("4", "7", "EFFECT", "2025-02-10"),
        filing("5", "7", "424B4", "2025-02-11"),
        filing("6", "8", "EFFECT", "2025-02-11"),  # no registration: not a deal
        filing("7", "9", "F-1", "2025-01-02", "Funded Trust"),
        filing("8", "10", "S-1", "2025-01-02", "Blank Acquisition Corp"),
        filing("9", "11", "S-1", "2025-01-02", "Old Co"),
        filing("10", "12", "S-1", "2025-01-02", "Quit Co"),
        filing("11", "12", "RW", "2025-03-03", "Quit Co"),
        filing("12", "13", "S-1", "2025-01-02", "Late Co"),
        filing("13", "13", "RW", "2025-06-01", "Late Co"),
    ]
    docs = {
        "2": json.dumps({"range": [10, 12], "shares": 5e6, "lead": "Goldman Sachs", "symbol": None}),
        "3": json.dumps({"range": [12, 14], "shares": None, "lead": None, "symbol": "ACME"}),
        "5": json.dumps({"offer": 15.0, "shares": 6e6, "lead": None, "symbol": None}),
    }
    cos = {
        "11": {"sic": "2834", "first_report": "2020-03-01", "tickers": "[]"},
        "13": {"sic": "2834", "first_report": None, "tickers": '["LATE"]'},
    }
    trades = {
        "7": {"date": "2025-02-12", "close": 18.0, "open": 16.0, "symbol": "ACME"},
        "13": {"date": "2025-03-03", "close": 5.0, "open": 5.0, "symbol": "LATE"},
    }
    deals = {d.cik: d for d in ipos.build(fs, docs, cos, trades)}
    assert "8" not in deals
    acme = deals["7"]
    assert acme.initial_range == (10.0, 12.0) and acme.latest_range == (12.0, 14.0)
    assert (acme.effective, acme.priced, acme.offer, acme.shares, acme.lead, acme.symbol) == (
        "2025-02-10",
        "2025-02-11",
        15.0,
        6e6,
        "Goldman Sachs",
        "ACME",
    )
    assert acme.first_day_return == pytest.approx(0.2) and acme.status(D(2025, 3, 1)) == "trading"
    assert deals["9"].ipo is False and "fund" in str(deals["9"].why_not)
    assert deals["10"].ipo is False and deals["11"].ipo is False
    assert deals["12"].status(D(2025, 4, 1)) == "withdrawn"
    assert deals["13"].withdrawn is None and deals["13"].symbol == "LATE"  # an RW after listing is something else
    blank = ipos.Deal(cik="1", company="X", foreign=False, registered="2025-01-02")
    assert blank.initial_range is None and blank.latest_range is None and blank.first_day_return is None
    assert blank.status(D(2025, 2, 1)) == "on_file"
    blank.effective = "2025-01-10"
    assert blank.status(D(2025, 1, 20)) == "effective" and blank.status(D(2025, 3, 1)) == "postponed"
    blank.priced = "2025-01-11"
    assert blank.status(D(2025, 1, 20)) == "priced"
    counts = ipos.counts(list(deals.values()), D(2025, 4, 1))
    assert counts["trading"] == 2 and counts["withdrawn"] == 1


def test_ipo_checks() -> None:
    d = ipos.Deal(
        cik="1",
        company="X",
        foreign=False,
        registered="2025-01-02",
        ranges=[("2025-01-10", 10, 12)],
        offer=40.0,
        priced="2025-02-03",
        effective="2025-02-10",
        first_trade={"date": "2025-02-07", "close": 1.0},
    )
    names = {p["check"] for p in ipos.check([d])}
    assert names == {"offer_vs_range", "pricing_vs_424b4", "trade_before_pricing"}
    d2 = ipos.Deal(
        cik="2", company="Y", foreign=False, registered="2025-01-02", first_trade={"date": "2025-02-07", "close": 1}
    )
    assert [p["check"] for p in ipos.check([d2])] == ["missing_offer"]
    spac = ipos.Deal(
        cik="3",
        company="Z",
        foreign=False,
        registered="2025-01-02",
        ipo=False,
        offer=1000.0,
        ranges=[("2025-01-03", 1, 2)],
    )
    assert ipos.check([spac]) == [] and checks.ipo([d2])[0]["severity"] == "warning"


# ---------------------------------------------------------------------------- prospectus parsers


def test_prospectus_parsers() -> None:
    t = "The initial public offering price is $17.00 per share. A fee of $1.50 per share is paid."
    assert prospectus.offer_price(t) == 17.0 and prospectus.offer_price(t, (20, 22)) == 17.0
    assert prospectus.offer_price("Underwriting discount $1.50 per share", (10, 12)) is None
    assert prospectus.offer_price("nothing here") is None
    assert prospectus.price_range("between $14.00 and $16.00 per share") == (14.0, 16.0)
    assert prospectus.price_range("the offering price will be between $3 and $5") == (3.0, 5.0)
    assert prospectus.price_range("an assumed initial public offering price of $4.00") == (4.0, 4.0)
    assert prospectus.price_range("between $16.00 and $14.00 per share") is None
    assert prospectus.price_range("an assumed offering price of $9999.00") is None
    assert prospectus.shares_offered("12,500,000 Shares Class A") == 12_500_000
    assert prospectus.shares_offered("10,000 Shares") is None and prospectus.shares_offered("") is None
    assert prospectus.listing_symbol("listed under the symbol “SNAP.”") == "SNAP"
    assert prospectus.listing_symbol("no symbol") is None
    assert prospectus.lead_bank("Merrill Lynch and Goldman Sachs") == "BofA Securities"
    assert prospectus.lead_bank("no banks") is None
    assert prospectus.text_of(b"<p>a&nbsp;b</p><script>x</script>") == " a b "


# ---------------------------------------------------------------------------- statistics


def test_scores_and_calibration() -> None:
    p, y = np.array([0.9, 0.1, 0.8, 0.3]), np.array([1, 0, 0, 1])
    s = stats.scores(p, y, np.zeros(4), np.array([0.1, -0.1, 0.0, 0.2]))
    assert s["brier"] == pytest.approx(np.mean((p - y) ** 2)) and s["hit_rate"] == 0.5 and s["n"] == 4
    assert s["rmse"] == pytest.approx(math.sqrt(np.mean(np.array([0.1, -0.1, 0.0, 0.2]) ** 2)))
    cal = stats.calibration(np.array([0.05, 0.95, 1.0]), np.array([0, 1, 1]))
    assert cal["ece"] == pytest.approx((0.05 + 2 * 0.025) / 3) and len(cal["table"]) == 2
    assert stats.log_loss(np.array([0.0]), np.array([1]))[0] == pytest.approx(-math.log(stats.EPS))


def test_diebold_mariano_and_bootstrap() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(1.0, 1, 400), rng.normal(1.3, 1, 400)
    dm = stats.diebold_mariano(a, b)
    assert dm["mean_diff"] < 0 and dm["p"] < 0.01 and dm["n"] == 400
    same = stats.diebold_mariano(a, a)
    assert same["p"] == 1.0 and math.isnan(same["stat"])
    shifted = stats.diebold_mariano(a, a - 1)
    assert shifted["p"] == 0.0  # a constant difference: no variance, certainly different
    assert math.isnan(stats.diebold_mariano([1, 2], [1, 1])["p"]) and math.isnan(
        stats.diebold_mariano([], [])["mean_diff"]
    )
    boot = stats.block_bootstrap(a, b, reps=500)
    assert boot["hi"] < 0 and boot["p"] <= 0.01 and boot == stats.block_bootstrap(a, b, reps=500)  # seeded
    assert math.isnan(stats.block_bootstrap([1], [0])["lo"]) and math.isnan(stats.block_bootstrap([], [])["mean_diff"])
    assert stats.newey_west(np.ones(5), 2) == 0.0


def test_holm_and_by_period() -> None:
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert math.isnan(adj["d"])
    assert stats.by_period(["b", "a", "a"], [3.0, 1.0, 2.0]) == [1.5, 3.0]


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=1, max_size=50))
def test_score_bounds(rows) -> None:
    p = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=float)
    s = stats.scores(p, y, p, y)
    assert 0 <= s["brier"] <= 1 and 0 <= s["hit_rate"] <= 1 and 0 <= s["ece"] <= 1 and s["log_loss"] >= 0
    assert np.isfinite(s["log_loss"])


# ---------------------------------------------------------------------------- models and baselines


def test_fit_and_round_trip() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + rng.normal(0, 0.5, 400) > 0).astype(float)
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y, "ret": x[:, 0] * 0.1})
    df.loc[0, "b"] = np.nan
    m = models.fit("t", df, ("a", "b"), "2025-01-01")
    p = m.prob(df)
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.75 and np.all((p >= 0) & (p <= 1))
    assert np.corrcoef(m.value(df), df["ret"])[0, 1] > 0.99
    again = models.Model.from_params(json.loads(json.dumps(m.params())))
    assert np.array_equal(again.prob(df), p) and again.model_id == m.model_id
    with pytest.raises(ValueError):
        models.fit("t", df.head(10), ("a", "b"), "x")
    flat = df.assign(c=1.0, d=np.nan)
    assert np.isfinite(models.fit("t", flat, ("a", "c", "d"), "x").prob(flat)).all()  # constant and empty columns


def test_baselines_use_only_known_outcomes() -> None:
    y = [1, 1, 0, 1]
    out = models.base_rate(y, ["A"] * 4, ["d1", "d1", "d2", "d3"], ["d2", "d2", "d3", None], prior=0.5, min_obs=1)
    # an outcome counts only from the day after it is known (strictly before the prediction's date): same-day
    # outcomes are never used, which for stocks costs one day of history and can never leak
    assert list(out) == [0.5, 0.5, 0.5, 1.0]
    pooled = models.base_rate([1, 0, 1], ["A", "B", "C"], ["1", "2", "3"], ["1", "2", "3"], prior=0.3, min_obs=2)
    assert list(pooled) == [0.3, 0.3, 0.5]
    recent = models.recent_mean(
        [0.1, 0.2, 0.3, np.nan], ["1", "2", "3", "4"], ["1", "2", "3", "4"], prior=9.0, window=2, min_obs=2
    )
    assert list(recent) == [9.0, 9.0, pytest.approx(0.15), pytest.approx(0.25)]


# ---------------------------------------------------------------------------- features


def test_macro_asof_takes_what_was_published() -> None:
    rows = pd.DataFrame(
        [
            {"series": "UST10Y", "date": "2025-01-02", "value": 4.0, "available_at": "2025-01-02T23:00:00Z"},
            {"series": "UST10Y", "date": "2025-01-03", "value": 4.1, "available_at": "2025-01-03T23:00:00Z"},
            {
                "series": "UST10Y",
                "date": "2025-01-02",
                "value": 9.9,
                "available_at": "2025-02-01T00:00:00Z",
            },  # a revision
        ]
    )
    t = [dt.datetime(2025, 1, 2, 22, tzinfo=dt.UTC), dt.datetime(2025, 1, 3, 23, tzinfo=dt.UTC)]
    out = features.macro_asof(rows, t)
    assert np.isnan(out["UST10Y"].iloc[0]) and out["UST10Y"].iloc[1] == 4.1
    assert features.macro_asof(pd.DataFrame(), t).shape == (2, 0)


def test_panels_handle_missing_inputs() -> None:
    assert features.stock_panel(pd.DataFrame(), pd.DataFrame()).empty
    closes = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=["2025-01-02", "2025-01-03", "2025-01-06"])
    p = features.stock_panel(closes, pd.DataFrame(), with_target=False)
    assert "y" not in p and p["mkt_r1"].isna().all() and p["vix"].isna().all() and p["slope"].isna().all()
    assert features.ipo_rows([], pd.DataFrame(), 0.2).empty
    d = ipos.Deal(
        cik="1",
        company="X",
        foreign=True,
        registered="2025-01-02",
        ranges=[("2025-01-10", 10, 12)],
        effective="2025-02-01",
        shares=None,
        lead="Goldman Sachs",
    )
    row = features.ipo_rows([d], pd.DataFrame(), 0.2).iloc[0]
    assert np.isnan(row["log_size"]) and np.isnan(row["heat"]) and row["top_bank"] == 1 and row["foreign"] == 1
    assert np.isnan(row["y"])


# ---------------------------------------------------------------------------- evaluation pieces


def test_walkforward_edges() -> None:
    empty = evaluate.stock_walkforward(pd.DataFrame(columns=["date", "symbol", *features.STOCK_FEATURES, "y", "ret"]))
    assert empty.empty and "p_model" in empty
    one_class = pd.DataFrame(
        {
            "date": [f"2025-01-{i:02d}" for i in range(1, 31)] * 2,
            "symbol": ["A"] * 30 + ["B"] * 30,
            **{f: 0.0 for f in features.STOCK_FEATURES},
            "y": 1.0,
            "ret": 0.01,
        }
    )
    assert evaluate.stock_walkforward(one_class, warmup=5, refit_every=5).empty  # never fits: one outcome only
    assert evaluate.ipo_walkforward(pd.DataFrame()).empty
    rows = pd.DataFrame(
        {
            "cik": [str(i) for i in range(60)],
            "company": "X",
            "moment": [f"2024-{1 + i // 6:02d}-15" for i in range(60)],
            **{f: 0.0 for f in features.IPO_FEATURES},
            "y": 1.0,
            "ret": 0.3,
            "trade_date": [f"2024-{1 + i // 6:02d}-16" for i in range(60)],
        }
    )
    assert evaluate.ipo_walkforward(rows, min_train=5).empty  # every IPO popped: nothing to fit
    assert evaluate.report(pd.DataFrame({"y": [], "ret": [], "date": []}), "date", ("model",))["n"] == 0
    assert evaluate.markdown({"n": 0}, "T", ())[-2].startswith("No resolved")


def test_leakage_test_catches_a_leaky_feature(monkeypatch) -> None:
    idx = [d.isoformat() for d in __import__("bot").markets.trading_days(D(2024, 1, 2), D(2024, 12, 31))]
    rng = np.random.default_rng(1)
    closes = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for s in ("A", "B", "SPX")}, index=idx
    )
    honest = evaluate.leakage_stocks(closes, pd.DataFrame(), [idx[200]])
    assert honest["passed"]
    real = features.stock_panel

    def peeking(closes, macro_rows, symbols=None, with_target=True):
        p = real(closes, macro_rows, symbols, with_target)
        p["r1"] = p["ret"] if "ret" in p else p["r1"]  # tomorrow's return as a "feature"
        return p

    monkeypatch.setattr(features, "stock_panel", peeking)
    leaky = evaluate.leakage_stocks(closes, pd.DataFrame(), [idx[200]])
    assert not leaky["passed"] and leaky["worst_diff"] > 0
    assert evaluate._same(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": [1.0, 2.0]}), ["a"]) == math.inf
    assert evaluate._same(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": [np.nan]}), ["a"]) == math.inf
    assert evaluate._same(pd.DataFrame({"a": []}), pd.DataFrame({"a": []}), ["a"]) == 0.0


def test_ipo_leakage_with_macro_and_json() -> None:
    d = ipos.Deal(
        cik="1",
        company="X",
        foreign=False,
        registered="2024-01-02",
        ranges=[("2024-01-10", 10, 12), ("2024-03-01", 12, 14)],
        effective="2024-02-01",
        offer=11.0,
        first_trade={"date": "2024-02-02", "close": 13.0},
    )
    macro = pd.DataFrame(
        [
            {"series": "VIX", "date": "2024-01-31", "value": 14.0, "available_at": "2024-01-31T21:30:00Z"},
            {"series": "VIX", "date": "2024-03-31", "value": 20.0, "available_at": "2024-03-31T21:30:00Z"},
        ]
    )
    assert evaluate.leakage_ipos([d], macro, ["2024-02-15"], 0.2)["passed"]
    text = evaluate.to_json({"a": np.float64(1.5), "b": np.int64(2), "c": np.bool_(True), "d": d})
    assert json.loads(text)["d"]["cik"] == "1"
    with pytest.raises(TypeError):
        evaluate.to_json({"x": object()})


# ---------------------------------------------------------------------------- tracking


def pred(pid, model, subject, target, prob, made_at="2025-10-01T22:00:00Z", kind="stock", date="2025-10-01"):
    return {
        "pred_id": pid,
        "model": model,
        "kind": kind,
        "subject": subject,
        "event": "close_up",
        "target": target,
        "prob": prob,
        "value": 0.0,
        "made_at": made_at,
        "data": {"date": date},
        "run_id": "r",
    }


def test_output_checks() -> None:
    good = [pred("1", "m", "A", "t", 0.5)]
    tracking._check_outputs(good, {"A"}, 1)
    with pytest.raises(tracking.PredictionError, match="non-finite"):
        tracking._check_outputs([{**good[0], "prob": float("nan")}], {"A"}, 1)
    with pytest.raises(tracking.PredictionError, match="probability"):
        tracking._check_outputs([{**good[0], "prob": 1.5}], {"A"}, 1)
    with pytest.raises(tracking.PredictionError, match="answered"):
        tracking._check_outputs(good, {"A", "B"}, 1)


def test_watch_warns_after_three_bad_periods_and_on_drift(store: Store) -> None:
    rows, outs = [], []
    for w, day in enumerate(("2025-09-02", "2025-09-09", "2025-09-16", "2025-09-23")):
        for i in range(30):
            y = float(i % 2)
            for model, p in (("stock-x", 0.9 if y == 0 else 0.1), ("base_rate", 0.5), ("random_walk", 0.5)):
                pid = f"{model}:{w}:{i}"
                rows.append(pred(pid, model, f"S{i}", day, p))
                outs.append({"pred_id": pid, "outcome": y, "value": 0.0, "resolved_at": "x", "run_id": "r"})
    store.record("predictions", rows)
    store.record("outcomes", outs)
    out = tracking.watch(store, "stock", 3, 0.10)
    assert len(out["periods"]) == 4 and len(out["warnings"]) == 2
    assert any("each of the last 3" in w for w in out["warnings"]) and any(
        "calibration drift" in w for w in out["warnings"]
    )
    assert tracking.watch(store, "ipo", 3, 0.1) == {"periods": [], "warnings": []}
    assert tracking.period_of("ipo", pd.Series({"made_at": "2025-05-02T00:00:00Z"})) == "2025-Q2"


def test_score_voids_a_target_with_no_close(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    ctx = make_ctx(cfg, Opener(), NOW)
    ctx.store.record(
        "predictions",
        [
            pred("p1", "m", "ZZZ", "2025-09-02", 0.5, date="2025-09-01"),
            pred("p2", "m", "ZZZ", "2025-10-08", 0.5, date="2025-10-07"),
            {**pred("p3", "m", "404", "first_trade", 0.5, kind="ipo")},
        ],
    )
    assert tracking.score(ctx) == 1  # the old target is void; the recent one and the unknown IPO still wait
    assert ctx.store.query("SELECT pred_id, outcome FROM outcomes") == [{"pred_id": "p1", "outcome": None}]
    assert tracking.score(ctx) == 0


def test_predictions_need_data(tmp_path) -> None:
    ctx = make_ctx(make_cfg(tmp_path), Opener(), NOW)
    assert tracking.predict_stocks(ctx) == 0 and ctx.warnings[0]["what"] == "no_stock_predictions"
    assert tracking.predict_ipos(ctx) == 0
    assert tracking.retrain(ctx, "stock")["reason"] == "not enough history"
    assert tracking.retrain(ctx, "ipo")["reason"] == "not enough IPOs"


# ---------------------------------------------------------------------------- settings


def test_settings_from_environment_and_env_file(tmp_path, monkeypatch) -> None:
    s = config.settings(
        {
            "BOT_DATA_DIR": str(tmp_path),
            "SEC_USER_AGENT": "Jane Doe jane@example.com",
            "BOT_REPLAY": "r",
            "BOT_RECORD": "w",
            "BOT_SYMBOLS": "aapl, msft,",
            "BOT_HISTORY_START": "2020-01-02",
            "BOT_SEC_BUDGET": "10",
        }
    )
    assert (s.data_dir, s.sec_user_agent, s.symbols, s.history_start, s.sec_budget) == (
        tmp_path,
        "Jane Doe jane@example.com",
        ("AAPL", "MSFT"),
        "2020-01-02",
        10,
    )
    assert str(s.replay_dir) == "r" and str(s.record_dir) == "w" and s.tracking_dir is None
    bare = config.settings({"SEC_USER_AGENT": "no-contact"})
    assert bare.sec_user_agent is None and bare.symbols is None and bare.data_dir == config.ROOT / "data"
    assert bare.tracking_dir == config.ROOT / "tracking" / "bot"
    env = tmp_path / ".env"
    env.write_text('# comment\n\nSEC_USER_AGENT="A B a@b.c"\nBROKEN LINE\n', encoding="utf-8")
    assert config._env_file(env) == {"SEC_USER_AGENT": "A B a@b.c"}
    assert config._env_file(tmp_path / "missing") == {}
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert config.settings().sec_user_agent == "A B a@b.c"
