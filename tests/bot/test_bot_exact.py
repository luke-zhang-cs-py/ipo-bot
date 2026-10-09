"""Exact-value tests for the pure core, written against independent references (dateutil's Easter, brute-force
weekday counting, scipy's optimiser, closed-form formulas) and at every boundary, so that a changed operator,
constant or message is caught. These are what the weekly mutation run (mutmut) leans on."""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from dateutil.easter import easter as reference_easter
from scipy import optimize
from scipy import stats as sps

from bot import checks, ipos, markets, models, prospectus, stats
from bot.store import TABLES, available, iso, parse_iso

D = dt.date
UTC = dt.UTC


# ---------------------------------------------------------------------------- the calendar


def test_easter_matches_dateutil_for_four_centuries() -> None:
    for y in range(1800, 2200):
        assert markets.easter(y) == reference_easter(y), y


@pytest.mark.parametrize("year", [2023, 2024, 2025, 2026, 2027, 2028])
def test_nth_weekday_by_brute_force(year: int) -> None:
    for month in range(1, 13):
        days = [D(year, month, d) for d in range(1, 32) if _valid(year, month, d)]
        for weekday in range(7):
            matching = [d for d in days if d.weekday() == weekday]
            for n in (1, 2, 3, 4):
                assert markets._nth_weekday(year, month, weekday, n) == matching[n - 1]
            assert markets._nth_weekday(year, month, weekday, -1) == matching[-1]


def _valid(y: int, m: int, d: int) -> bool:
    try:
        D(y, m, d)
        return True
    except ValueError:
        return False


def test_observed_rule() -> None:
    assert markets._observed(D(2026, 7, 4)) == D(2026, 7, 3)  # Saturday -> Friday
    assert markets._observed(D(2027, 7, 4)) == D(2027, 7, 5)  # Sunday -> Monday
    assert markets._observed(D(2025, 7, 4)) == D(2025, 7, 4)  # a Friday stays


@pytest.mark.parametrize(
    "year,expected",
    [
        (
            2025,
            [
                D(2025, 1, 1),
                D(2025, 1, 20),
                D(2025, 2, 17),
                D(2025, 4, 18),
                D(2025, 5, 26),
                D(2025, 6, 19),
                D(2025, 7, 4),
                D(2025, 9, 1),
                D(2025, 11, 27),
                D(2025, 12, 25),
            ],
        ),
        (
            2027,
            [
                D(2027, 1, 1),
                D(2027, 1, 18),
                D(2027, 2, 15),
                D(2027, 3, 26),
                D(2027, 5, 31),
                D(2027, 6, 18),
                D(2027, 7, 5),
                D(2027, 9, 6),
                D(2027, 11, 25),
                D(2027, 12, 24),
            ],
        ),
    ],
)
def test_holidays_exactly(year, expected) -> None:
    assert sorted(markets.holidays(year)) == expected


def test_quarters() -> None:
    starts = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 7, 8: 7, 9: 7, 10: 10, 11: 10, 12: 10}
    for month, first in starts.items():
        d = D(2025, month, 15)
        assert markets.quarter_start(d) == D(2025, first, 1)
        assert markets.quarter_label(d) == f"2025Q{(first - 1) // 3 + 1}"
    assert [markets.next_quarter(D(2025, m, 2)) for m in (1, 4, 7, 10)] == [
        D(2025, 4, 1),
        D(2025, 7, 1),
        D(2025, 10, 1),
        D(2026, 1, 1),
    ]


def test_close_and_open_exactly() -> None:
    assert markets.close_utc(D(2025, 3, 7)) == dt.datetime(2025, 3, 7, 21, tzinfo=UTC)  # EST
    assert markets.close_utc(D(2025, 3, 10)) == dt.datetime(2025, 3, 10, 20, tzinfo=UTC)  # EDT from 9 March
    assert markets.close_utc(D(2025, 11, 3)) == dt.datetime(2025, 11, 3, 21, tzinfo=UTC)  # EST from 2 November
    assert markets.open_utc(D(2025, 1, 2)) == dt.datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    exactly = dt.datetime(2025, 3, 10, 20, tzinfo=UTC)
    assert markets.last_closed_session(exactly) == D(2025, 3, 10)
    assert markets.last_closed_session(exactly - dt.timedelta(seconds=1)) == D(2025, 3, 7)


# ---------------------------------------------------------------------------- the store's helpers


def test_store_helpers() -> None:
    assert iso(dt.datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2025-01-02T03:04:05Z"
    assert parse_iso("2025-01-02T03:04:05Z") == dt.datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    nominal = dt.datetime(2025, 1, 2, 21, tzinfo=UTC)
    assert available(nominal, nominal + dt.timedelta(days=3)) == "2025-01-05T21:00:00Z"  # 3 days exactly: still live
    assert available(nominal, nominal + dt.timedelta(days=3, seconds=1)) == "2025-01-02T21:00:00Z"
    t = TABLES["prices"]
    assert t.key == ("symbol", "date", "source") and t.values == ("open", "high", "low", "close", "volume")
    assert t.types["close"] == "REAL" and TABLES["universe"].types["member"] == "INTEGER"


# ---------------------------------------------------------------------------- statistics, against the formulas


def test_scores_exactly() -> None:
    p, y = np.array([0.8, 0.3, 0.6, 0.5]), np.array([1.0, 0.0, 0.0, 1.0])
    v, a = np.array([0.1, -0.2, 0.0, 0.3]), np.array([0.0, 0.0, 0.1, 0.1])
    s = stats.scores(p, y, v, a)
    assert s["brier"] == pytest.approx(np.mean((p - y) ** 2), abs=1e-15)
    assert s["log_loss"] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), abs=1e-15)
    assert s["hit_rate"] == 0.625  # 0.8 right, 0.3 right, 0.6 wrong, 0.5 calls neither side: half a hit
    assert s["mae"] == pytest.approx(np.mean(np.abs(v - a))) and s["rmse"] == pytest.approx(
        np.sqrt(np.mean((v - a) ** 2))
    )
    assert s["event_rate"] == 0.5 and s["n"] == 4
    assert stats.brier(np.array([0.25]), np.array([1]))[0] == 0.5625
    assert stats.log_loss(np.array([1.0]), np.array([0]))[0] == pytest.approx(-math.log(stats.EPS))


def test_calibration_table_exactly() -> None:
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.999])
    y = np.array([0, 1, 0, 1, 1])
    cal = stats.calibration(p, y)
    assert [(r["lo"], r["hi"], r["n"]) for r in cal["table"]] == [(0.0, 0.1, 1), (0.1, 0.2, 2), (0.9, 1.0, 2)]
    assert cal["table"][1]["mean_p"] == pytest.approx(0.15) and cal["table"][1]["rate"] == 0.5
    expected = (1 * 0.05 + 2 * abs(0.15 - 0.5) + 2 * abs((0.95 + 0.999) / 2 - 1)) / 5
    assert cal["ece"] == pytest.approx(expected)


def _reference_nw(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    gamma = [np.sum(d[k:] * d[: n - k]) / n for k in range(lags + 1)]
    return gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))


def test_newey_west_and_dm_match_the_formulas() -> None:
    rng = np.random.default_rng(11)
    a, b = rng.normal(0.30, 0.1, 120), rng.normal(0.31, 0.1, 120)
    d = a - b
    for lags in (0, 1, 3, 7):
        assert stats.newey_west(d, lags) == pytest.approx(_reference_nw(d, lags), rel=1e-12)
    n, h = len(d), 1
    lags = max(h - 1, int(n ** (1 / 3)))
    stat = d.mean() / math.sqrt(_reference_nw(d, lags) / n) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = stats.diebold_mariano(a, b)
    assert dm["stat"] == pytest.approx(stat, rel=1e-12) and dm["mean_diff"] == pytest.approx(d.mean())
    assert dm["p"] == pytest.approx(2 * sps.t.sf(abs(stat), n - 1), rel=1e-12) and dm["n"] == 120
    h3 = stats.diebold_mariano(a, b, h=3)
    lags3 = max(2, int(n ** (1 / 3)))
    stat3 = d.mean() / math.sqrt(_reference_nw(d, lags3) / n) * math.sqrt((n + 1 - 6 + 6 / n) / n)
    assert h3["stat"] == pytest.approx(stat3, rel=1e-12)
    three = stats.diebold_mariano([1.0, 2.0, 4.0], [1.0, 1.0, 1.0])
    assert three["n"] == 3 and not math.isnan(three["stat"])


def test_block_bootstrap_matches_a_reference() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.0, 1, 64), rng.normal(0.4, 1, 64)
    d = a - b
    out = stats.block_bootstrap(a, b, reps=300, seed=9)
    block = max(1, round(64 ** (1 / 3)))
    gen = np.random.default_rng(9)
    starts = gen.integers(0, 64 - block + 1, size=(300, math.ceil(64 / block)))
    means = np.array([np.concatenate([d[s : s + block] for s in row])[:64].mean() for row in starts])
    assert out["mean_diff"] == pytest.approx(d.mean())
    assert out["lo"] == pytest.approx(np.percentile(means, 2.5)) and out["hi"] == pytest.approx(
        np.percentile(means, 97.5)
    )
    p = max(float((np.abs(means - d.mean()) >= abs(d.mean())).mean()), 1 / 300)
    assert out["p"] == pytest.approx(p)
    assert stats.block_bootstrap(a, b, reps=300, block=8, seed=9) != out  # an explicit block length is used
    assert stats.block_bootstrap(a, a, reps=50)["p"] == 1.0
    assert not math.isnan(stats.block_bootstrap([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], reps=20)["lo"])


def test_holm_exactly() -> None:
    adj = stats.holm({"a": 0.02, "b": 0.01, "c": 0.04})
    assert adj == {"b": pytest.approx(0.03), "a": pytest.approx(0.04), "c": pytest.approx(0.04)}
    assert stats.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}
    assert stats.by_period(["x"], [2.5]) == [2.5]


# ---------------------------------------------------------------------------- models, against scipy


def test_logit_matches_scipy() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 3))
    y = (x @ np.array([1.0, -0.5, 0.0]) + rng.logistic(size=300) > 0).astype(float)
    l2 = 2.0
    X = np.column_stack([np.ones(300), x])

    def loss(w: np.ndarray) -> float:
        z = X @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + l2 / 2 * np.sum(w[1:] ** 2))

    ref = optimize.minimize(loss, np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    assert models.fit_logit(x, y, l2=l2) == pytest.approx(ref, abs=1e-5)


def test_ridge_matches_the_closed_form() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 2))
    y = x @ np.array([0.3, -0.1]) + 0.05 + rng.normal(0, 0.01, 50)
    X = np.column_stack([np.ones(50), x])
    pen = np.diag([0.0, 1.5, 1.5])
    assert models.fit_ridge(x, y, l2=1.5) == pytest.approx(np.linalg.solve(X.T @ X + pen, X.T @ y), abs=1e-8)


def test_fit_standardizes_and_clips() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan] * 10, "b": [5.0] * 40, "y": [0.0, 1.0] * 20, "ret": [0.0] * 40})
    m = models.fit("t", df, ("a", "b"), "2025-01-01")
    assert m.mean == pytest.approx([2.0, 5.0]) and m.std == pytest.approx([math.sqrt(2 / 3), 1.0])
    assert m.n_train == 40 and m.trained_through == "2025-01-01" and m.kind == "t"
    far = pd.DataFrame({"a": [1e9, -1e9, np.nan], "b": [5.0, 5.0, 5.0]})
    z = m._x(far)
    assert z[:, 0].tolist() == [models.CLIP, -models.CLIP, 0.0]
    with pytest.raises(ValueError, match="30 with both outcomes"):
        models.fit("t", df.head(29), ("a", "b"), "x")
    with pytest.raises(ValueError):
        models.fit("t", df.assign(y=1.0), ("a", "b"), "x")
    assert models.fit("t", df.head(30), ("a", "b"), "x").n_train == 30


def test_baselines_at_their_thresholds() -> None:
    y = [1.0] * 20 + [0.0]
    t = [f"{i:03d}" for i in range(21)]
    known = [f"{i:03d}" for i in range(21)]
    out = models.base_rate(y, ["A"] * 21, [f"{i + 1:03d}" for i in range(21)], known, prior=0.3, min_obs=20)
    assert out[18] == 0.3 and out[19] == 1.0  # 19 known outcomes: the prior; 20: the rate
    assert models.recent_mean([1.0, 3.0, 5.0], t[:3], t[:3], prior=-1.0, window=2, min_obs=1).tolist() == [
        -1.0,
        1.0,
        2.0,
    ]


# ---------------------------------------------------------------------------- checks: boundaries and messages


def bars(rows):
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def test_price_check_boundaries_and_messages() -> None:
    ok = bars([("A", "d1", 10, 10.5, 9.5, 10.5 * 1.005, 1)])  # within the half-percent rounding slack
    assert checks.prices(ok) == []
    out = checks.prices(
        bars([("A", "d2", 10, 10.5, 9.5, 10.5 * 1.006, 1), ("B", "d3", 0, 1, 1, 1, 1), ("C", "d4", 1, 1, 0, 1, 1)])
    )
    assert [(i["subject"], i["severity"]) for i in out] == [("B d3", "error"), ("C d4", "error"), ("A d2", "warning")]
    assert out[0] == {
        "check": "prices",
        "severity": "error",
        "subject": "B d3",
        "detail": "non-positive price (close 1.0)",
    }
    assert out[2]["detail"] == f"close {10.5 * 1.006} outside low 9.5 - high 10.5"
    nan_range = bars([("A", "d", None, None, None, 5.0, None)])
    assert checks.prices(nan_range) == []


def test_move_limit_is_inclusive_and_messages_exact() -> None:
    b = bars([("A", "d1", 1, 1, 1, 100.0, 1), ("A", "d2", 1, 1, 1, 150.0, 1), ("A", "d3", 1, 1, 1, 226.0, 1)])
    out = checks.moves(b, {}, {}, 0.5)
    assert out == [
        {"check": "moves", "severity": "warning", "subject": "A d3", "detail": "close moved +51% from 150 to 226"}
    ]
    near = bars([("A", "d1", 1, 1, 1, 100.0, 1), ("A", "d2", 1, 1, 1, 47.0, 1)])
    assert checks.moves(near, {"A": {"d2": 2.0}}, {}, 0.5) == []  # 0.47 * 2 = 0.94: within 10% of the ratio
    assert checks.moves(near, {"A": {"d2": 2.5}}, {}, 0.5) != []  # 0.47 * 2.5 = 1.175: not explained
    drop = bars([("A", "d1", 1, 1, 1, 100.0, 1), ("A", "d2", 1, 1, 1, 38.0, 1)])
    assert checks.moves(drop, {}, {"A": {"d2": 58.0}}, 0.5) == []  # -62% against a 58% dividend
    assert checks.moves(drop, {}, {"A": {"d2": 50.0}}, 0.5) != []  # 12 points off: not explained


def test_gap_and_duplicate_messages() -> None:
    assert checks.gap_issues({"A": ["2025-01-02", "2025-01-03"]}) == [
        {
            "check": "missing_days",
            "severity": "warning",
            "subject": "A",
            "detail": "2 trading days with no bar: 2025-01-02, 2025-01-03",
        }
    ]
    five = checks.gap_issues({"A": [f"d{i}" for i in range(5)]})[0]["detail"]
    assert not five.endswith("...")
    _, issues = checks.dedupe([{"a": 1, "b": 2}, {"a": 1, "b": 2}], ("a", "b"))
    assert issues == [
        {"check": "duplicates", "severity": "warning", "subject": "1/2", "detail": "two rows for one key in one read"}
    ]


def test_staleness_boundaries(cfg) -> None:
    now = dt.datetime(2025, 10, 8, 22, tzinfo=UTC)  # the last closed session is 8 October
    b = bars([("ONE", "2025-10-07", 1, 1, 1, 1, 1), ("TWO", "2025-10-06", 1, 1, 1, 1, 1)])
    out = checks.staleness(b, ["ONE", "TWO"], {"VIX": "2025-10-06", "UST3M": "2025-10-03"}, now, cfg)
    assert [i["subject"] for i in out] == ["TWO", "UST3M"]  # one session behind is allowed; macro has one more
    assert out[0]["detail"] == "newest bar 2025-10-06, 2 trading days behind 2025-10-08"
    assert checks.lag("2025-10-03", D(2025, 10, 8)) == 3


def test_last_slot_and_missed_run_boundaries(store) -> None:
    assert checks.last_slot("daily", dt.datetime(2025, 10, 8, 22, 0, tzinfo=UTC)) == dt.datetime(
        2025, 10, 8, 22, tzinfo=UTC
    )
    assert checks.last_slot("daily", dt.datetime(2025, 10, 8, 21, 59, tzinfo=UTC)) == dt.datetime(
        2025, 10, 7, 22, tzinfo=UTC
    )
    assert checks.last_slot("daily", dt.datetime(2025, 10, 11, 12, tzinfo=UTC)) == dt.datetime(
        2025, 10, 10, 22, tzinfo=UTC
    )
    assert checks.last_slot("edgar", dt.datetime(2025, 10, 8, 5, 30, tzinfo=UTC)) == dt.datetime(
        2025, 10, 8, 3, tzinfo=UTC
    )
    assert checks.last_slot("weekly", dt.datetime(2025, 10, 8, tzinfo=UTC)) == dt.datetime(2025, 10, 4, 6, tzinfo=UTC)
    store.start_run("born", "weekly", "1", dt.datetime(2025, 10, 1, tzinfo=UTC))
    store.finish_run("born", "ok", {})
    store.start_run("d", "daily", "1", dt.datetime(2025, 10, 7, 21, 30, tzinfo=UTC))  # 30 minutes early: counts
    store.finish_run("d", "ok", {})
    store.start_run("x", "daily", "1", dt.datetime(2025, 10, 8, 22, tzinfo=UTC))  # failed: does not count
    store.finish_run("x", "failed", {})
    store.start_run("e", "edgar", "1", dt.datetime(2025, 10, 7, 23, 29, tzinfo=UTC))
    late = dt.datetime(2025, 10, 8, 1, 0, tzinfo=UTC)  # daily's 22:00 slot is 3 h old: its grace is over
    out = {i["subject"]: i for i in checks.missed_runs(store, late)}
    assert "job daily" not in out
    assert (
        out["job edgar"]["detail"] == "no finished run since the 2025-10-07 21:00 UTC slot (never)"
    )  # e never finished
    assert out["job edgar"]["severity"] == "warning"
    later = dt.datetime(2025, 10, 9, 2, 0, tzinfo=UTC)
    out = {i["subject"]: i for i in checks.missed_runs(store, later)}
    assert out["job daily"]["detail"] == "no finished run since the 2025-10-08 22:00 UTC slot (last 2025-10-07 21:30)"


def test_point_in_time_uses_the_earliest_close(store) -> None:
    row = {
        "symbol": "A",
        "date": "2025-10-08",
        "source": "yahoo",
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
    }
    store.put("prices", [{**row, "available_at": "2025-10-08T20:00:00Z"}], "r", "x")
    assert checks.point_in_time(store) == []
    store.put("prices", [{**row, "date": "2025-10-09", "available_at": "2025-10-09T19:59:59Z"}], "r", "x")
    assert [i["subject"] for i in checks.point_in_time(store)] == ["A 2025-10-09"]


def test_reconcile_message() -> None:
    row = {"symbol": "A", "date": "d", "yahoo": 10.0, "cboe": 10.1, "diff": 0.01, "ok": False}
    assert checks.reconcile([row], 0.005)[0]["detail"] == "yahoo 10.0000 vs cboe 10.1000 (+1.00%, limit 0.5%)"


# ---------------------------------------------------------------------------- IPO deals: boundaries


def deal(**kw):
    base = dict(cik="1", company="X", foreign=False, registered="2025-01-02", ranges=[("2025-01-10", 10.0, 12.0)])
    base.update(kw)
    return ipos.Deal(**base)


def test_offer_range_bounds_are_inclusive() -> None:
    assert ipos.check([deal(offer=5.0), deal(offer=24.0)]) == []
    names = [p["check"] for p in ipos.check([deal(offer=4.99), deal(offer=24.01)])]
    assert names == ["offer_vs_range", "offer_vs_range"]
    assert ipos.check([deal(offer=4.0)])[0]["detail"] == "offer 4.0 against a 10.0-12.0 range"


def test_pricing_gap_of_two_sessions_is_allowed() -> None:
    ft = {"date": "2025-02-05", "close": 10.0}
    assert ipos.check([deal(offer=11.0, priced="2025-02-03", first_trade=ft)]) == []  # Mon -> Wed: 2 sessions
    late = ipos.check([deal(offer=11.0, priced="2025-02-03", first_trade={"date": "2025-02-06", "close": 10.0})])
    assert [p["check"] for p in late] == ["pricing_vs_424b4"]
    early = ipos.check([deal(offer=11.0, priced="2025-02-06", first_trade={"date": "2025-02-03", "close": 10.0})])
    assert [p["check"] for p in early] == ["pricing_vs_424b4"]  # the 424B4 long after the trade, too
    on_day = ipos.check([deal(offer=11.0, effective="2025-02-03", first_trade={"date": "2025-02-03", "close": 1.0})])
    assert on_day == []
    before_reg = ipos.check([deal(offer=11.0, first_trade={"date": "2025-01-01", "close": 1.0})])
    assert [p["check"] for p in before_reg] == ["trade_before_pricing"]


def test_build_reads_each_form_once() -> None:
    def f(acc, form, filed):
        return {"accession": acc, "cik": "1", "form": form, "filed": filed, "company": "Co", "sic": None}

    fs = [
        f("a", "S-1", "2025-01-02"),
        f("b", "EFFECT", "2025-02-01"),
        f("c", "EFFECT", "2025-02-05"),
        f("d", "424B4", "2025-02-02"),
        f("e", "424B4", "2025-02-06"),
        f("g", "EFFECT", "2024-12-01"),
    ]
    docs = {"d": '{"offer": 9.0, "symbol": "CO", "lead": "Cantor", "shares": 1000000}', "e": '{"offer": 99.0}'}
    d = ipos.build(fs, docs, {}, {})[0]
    assert (d.effective, d.priced, d.offer, d.symbol, d.lead, d.shares) == (
        "2025-02-01",
        "2025-02-02",
        9.0,
        "CO",
        "Cantor",
        1e6,
    )
    assert d.registered == "2025-01-02" and d.company == "Co" and d.foreign is False and d.ipo is True
    sic = ipos.build([{**fs[0], "sic": "6770"}], {}, {}, {})[0]
    assert (sic.ipo, sic.why_not) == (False, "blank-check company")
    follow = ipos.build(fs[:1], {}, {"1": {"sic": "2834", "first_report": "2025-01-01", "tickers": []}}, {})[0]
    assert (follow.ipo, follow.why_not) == (False, "already reporting (a follow-on or an uplisting)")
    same_day = ipos.build(fs[:1], {}, {"1": {"sic": "2834", "first_report": "2025-01-02", "tickers": []}}, {})[0]
    assert same_day.ipo is True  # a report filed the same day as the registration is not "before" it
    listed = ipos.build(fs[:1], {}, {"1": {"sic": "", "first_report": None, "tickers": ["ZZ", "YY"]}}, {})[0]
    assert listed.symbol == "ZZ"


# ---------------------------------------------------------------------------- prospectus bounds


def test_prospectus_bounds() -> None:
    assert prospectus._plausible(0.5) and prospectus._plausible(500.0)
    assert not prospectus._plausible(0.49) and not prospectus._plausible(500.01)
    assert prospectus.offer_price("The initial public offering price is $0.50 per share") == 0.5
    assert prospectus.offer_price("The initial public offering price is $500.01 per share") is None
    assert prospectus.offer_price("initial public offering price of $12.00", (5, 6)) == 12.0  # 2 x 6: the top
    assert prospectus.offer_price("initial public offering price of $12.01", (5, 6)) is None
    assert prospectus.offer_price("initial public offering price of $2.50", (5, 6)) == 2.5
    assert prospectus.price_range("between $0.50 and $500.00 per share") == (0.5, 500.0)
    assert prospectus.price_range("between $0.49 and $1.00 per share") is None
    assert prospectus.price_range("an assumed offering price of $7.00") == (7.0, 7.0)
    assert prospectus.shares_offered("50,000 Shares") == 50_000 and prospectus.shares_offered("49,999 Shares") is None
    assert prospectus.shares_offered("5,000,000,001 Shares") is None
    assert prospectus.listing_symbol('under the trading symbol "ABC-"') == "ABC"
    assert prospectus.lead_bank("UBS and Goldman Sachs") == "UBS" and prospectus.lead_bank("Leerink") == "SVB Leerink"
    assert prospectus.text_of("<style>x</style>A&amp;B\xa0 C") == " A&B C"


# ---------------------------------------------------------------------------- price ranges from real filings

# Excerpts of real S-1/A text (as text_of reads it). Snowflake's second amendment moved the range to $100-110 but its
# cover still shows $75-85; the body's assumed midpoint ($105) and the fee table's maximum ($110) tell the truth.
SNOWFLAKE_S1A2 = (
    "CALCULATION OF REGISTRATION FEE Title of each Class of Securities to be Registered Amount to be Registered (1) "
    "Proposed Maximum Offering Price Per Share (2) Proposed Maximum Aggregate Offering Price (1)(2) Amount of "
    "Registration Fee Class A common stock, par value $0.0001 per share 32,200,000 $110.00 $3,542,000,000 "
    "It is currently estimated that the initial public offering price will be between $75.00 and $85.00 per share. "
    "Based on an assumed initial public offering price of $105.00 per share, which is the midpoint of the price range "
    "set forth on the cover page of this prospectus, each of Salesforce Ventures LLC and Berkshire Hathaway Inc. "
    "assuming an initial public offering price of $105.00 per share, which is the midpoint of the price range set forth"
)
# Lyft's last amendment: the "ff" ligature is lost ("o_ering"), so only a tolerant pattern finds the $70-72 range,
# and the old parser fell back to the assumed $71 as a point range.
LYFT_S1A3 = (
    "Prior to this o_ering, there has been no public market for our Class A common stock. It is currently estimated "
    "that the initial public o_ering price per share will be between $70.00 and $72.00. We have been approved to list "
    "based upon the assumed initial public offering price of $71.00 per share, which is the midpoint of the estimated "
    "offering price range set forth on the cover page of this prospectus"
)
# A first S-1 filed before the range is set: blanks where the amounts go.
BLANK_S1 = (
    "the initial public offering price will be between $ and $ per share. Each $1.00 increase or decrease in the "
    "assumed initial public offering price per share of $ , which is the midpoint of the price range set forth"
)


def test_price_range_prefers_the_stated_midpoint_and_tolerates_lost_ligatures() -> None:
    assert prospectus.range_reading(SNOWFLAKE_S1A2) == ((100.0, 110.0), "midpoint")
    assert prospectus.price_range(SNOWFLAKE_S1A2) == (100.0, 110.0)
    assert prospectus.range_reading(LYFT_S1A3) == ((70.0, 72.0), "stated")
    assert prospectus.range_reading(BLANK_S1) is None and prospectus.offer_price(BLANK_S1) is None
    # no fee table: the stated range's width, centred on the midpoint
    no_fee = SNOWFLAKE_S1A2.split("$3,542,000,000 ")[1]
    assert prospectus.range_reading(no_fee) == ((100.0, 110.0), "midpoint")
    # a fee table with no amount above the midpoint: the width again
    stale_fee = SNOWFLAKE_S1A2.replace("$110.00", "$85.00")
    assert prospectus.range_reading(stale_fee) == ((100.0, 110.0), "midpoint")
    # a range that cannot be rebuilt around the midpoint (it would go below $0.50): the midpoint as an assumed price
    tiny = "between $1.00 and $9.00 per share. an assumed offering price of $2.00 per share, which is the midpoint of the price range"
    assert prospectus.range_reading(tiny) == ((2.0, 2.0), "assumed")
    # within rounding of the midpoint: the stated range stands
    assert (
        prospectus.range_reading("between $4.00 and $4.25 per share; $4.13, which is the midpoint of the price range")[
            1
        ]
        == "stated"
    )
    assert prospectus.range_reading("an assumed offering price of $7.00") == ((7.0, 7.0), "assumed")
    assert prospectus.range_reading("an oﬀering price will be between $3 and $5") == ((3.0, 5.0), "stated")
    assert prospectus.shares_offered(",,,,,, Shares") is None


def _moved(lo: float, hi: float, mid: float, fee: float) -> str:
    return (
        f"Proposed Maximum Offering Price Per Share (2) Class A common stock 10,000,000 ${fee:.2f} "
        f"the initial public offering price will be between ${lo:.2f} and ${hi:.2f} per share. "
        f"an assumed initial public offering price of ${mid:.2f} per share, which is the midpoint of the price range"
    )


def test_a_range_cut_down_keeps_its_width_and_ignores_the_fee_tables_old_maximum() -> None:
    # cover $17-19 (stale), midpoint $15, fee table still at the old $19: about $14-16, not $11-19
    assert prospectus.range_reading(_moved(17, 19, 15, 19)) == ((14.0, 16.0), "midpoint")
    assert prospectus.range_reading(_moved(17, 19, 15, 16.5)) == ((14.0, 16.0), "midpoint")  # even when it would fit


def test_a_range_moved_up_takes_the_fee_tables_maximum_only_near_the_stated_half_width() -> None:
    assert prospectus.range_reading(_moved(75, 85, 105, 111)) == ((99.0, 111.0), "midpoint")  # 6 off: within half
    assert prospectus.range_reading(_moved(75, 85, 105, 107.5)) == ((102.5, 107.5), "midpoint")  # 2.5 off: the edge
    assert prospectus.range_reading(_moved(75, 85, 105, 107.49)) == ((100.0, 110.0), "midpoint")  # just past it
    assert prospectus.range_reading(_moved(75, 85, 105, 150)) == ((100.0, 110.0), "midpoint")  # far off: the width
