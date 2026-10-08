"""The benchmark's statistics and walk-forward discipline, and the forecast ledger. Offline: made-up prices."""
import datetime as dt
import json
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "evaluation"))

import benchmark  # noqa: E402
import tools  # noqa: E402


def test_ranks_and_spearman():
    assert benchmark.ranks([10, 30, 20, 20]) == [1, 4, 2.5, 2.5]
    assert benchmark.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert benchmark.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert benchmark.spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_newey_west_widens_the_error_for_overlapping_windows():
    rng = random.Random(1)
    noise = [rng.gauss(0, 1) for _ in range(300)]
    smooth = [sum(noise[k:k + 12]) / 12 + 0.05 for k in range(288)]     # overlapping like 12-month returns
    assert abs(benchmark.newey_west_t(smooth, 11)) < abs(benchmark.newey_west_t(smooth, 0))


def test_max_drawdown():
    assert benchmark.max_drawdown([1, 2, 1, 3, 1.5]) == pytest.approx(-0.5)


def fake_panel(n_months=120, seed=3):
    rng = random.Random(seed)
    months = [f"{2010 + m // 12}-{m % 12 + 1:02d}" for m in range(n_months)]
    px = {}
    for s, drift in (("UP", 0.02), ("MID", 0.0), ("DOWN", -0.02), ("X", 0.005), ("Y", -0.005), ("Z", 0.01)):
        p, path = 100.0, []
        for _ in months:
            p *= 1 + drift + rng.gauss(0, 0.01)
            path.append(p)
        px[s] = path
    return months, px


def test_a_persistent_trend_is_found_and_random_is_not(monkeypatch):
    months, px = fake_panel()
    monkeypatch.setitem(benchmark.UNIVERSES, "fake", list(px))
    monkeypatch.setattr(benchmark, "panel", lambda symbols, refresh=False: (months, px))
    _, per = benchmark.run("fake")
    rows = {r["algo"]: r for r in benchmark.summary(per, 12)}
    assert rows["Momentum 12-1"]["ic"] > 0.8 and rows["Historical mean"]["hit"] > 0.9
    assert abs(rows["Random (no skill)"]["ic"]) < 0.2


def test_no_algorithm_sees_the_future(monkeypatch):
    # Changing prices after month i must not change any score made at month i.
    months, px = fake_panel()
    i = 80
    before = {a: [f(px[s], i) for s in px] for a, f in benchmark.ALGORITHMS.items()}
    later = {s: p[:i + 1] + [x * 3 for x in p[i + 1:]] for s, p in px.items()}
    after = {a: [f(later[s], i) for s in later] for a, f in benchmark.ALGORITHMS.items()}
    assert before == after


def test_record_forecast_logs_and_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEDGER", tmp_path / "ledger.jsonl")
    ok = {"symbol": "aapl", "rating": "Overweight", "expected_return_pct": 18.5, "price": 200.0, "price_date": "2026-10-05"}
    text, err = tools.run_tool("record_forecast", ok)
    assert not err
    row = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert row["symbol"] == "AAPL" and row["horizon_months"] == 12 and row["date"] == dt.date.today().isoformat()
    full = {**ok, "bull_value": 260, "bull_prob": 25, "bear_value": 150, "bear_prob": 20}
    assert not tools.run_tool("record_forecast", full)[1]
    last = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["bear_value"] == 150 and last["bull_prob"] == 25
    for bad in ({"rating": "Strong Buy"}, {"price": 0}, {"price_date": "last week"}, {"symbol": "A; rm"},
                {"bull_prob": 140}, {"bear_value": 300, "bull_value": 200}, {"bear_value": -1}):
        assert tools.run_tool("record_forecast", {**ok, **bad})[1], bad


def test_the_ledger_scores_matured_forecasts_and_lists_pending(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    rows = [{"date": "2025-01-15", "symbol": "AAA", "rating": "Overweight", "expected_return_pct": 20, "price": 100,
             "price_date": "2025-01-14", "horizon_months": 12},
            {"date": dt.date.today().isoformat(), "symbol": "BBB", "rating": "Underweight", "expected_return_pct": -12,
             "price": 50, "price_date": dt.date.today().isoformat(), "horizon_months": 12}]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(benchmark, "LEDGER", ledger)
    monkeypatch.setattr(benchmark, "fetch", lambda s, refresh=False: {"2025-12": 130.0})
    monkeypatch.setattr(benchmark, "entry_factor", lambda s, day: 1.0)
    text = benchmark.score_ledger()
    assert "1 matured, 1 pending" in text and "got +30.0%" in text and "Direction right on 1 of 1" in text


def test_a_split_during_the_horizon_is_not_a_loss_and_unpriced_forecasts_are_not_misses(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    rows = [{"date": "2025-01-15", "symbol": s, "rating": "Overweight", "expected_return_pct": 20, "price": 400,
             "price_date": "2025-01-14", "horizon_months": 12} for s in ("AAA", "BBB")]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(benchmark, "LEDGER", ledger)
    # AAA split 4-for-1 during the year and ends at 110 in adjusted terms: up 10%, not down 72.5%. BBB has no price.
    monkeypatch.setattr(benchmark, "fetch", lambda s, refresh=False: {"2025-12": 110.0} if s == "AAA" else {})
    monkeypatch.setattr(benchmark, "entry_factor", lambda s, day: 0.25)
    text = benchmark.score_ledger()
    assert "got +10.0%" in text and "BBB: no price" in text
    assert "Direction right on 1 of 1." in text                  # BBB was never scored, so it is not a miss


def test_the_entry_factor_undoes_later_splits_and_carries_the_dividend_adjustment():
    day = lambda d: int(dt.datetime(2025, 1, d, tzinfo=dt.timezone.utc).timestamp())
    res = {"timestamp": [day(13), day(14), day(15)],
           "indicators": {"quote": [{"close": [99.0, 100.0, 101.0]}], "adjclose": [{"adjclose": [97.02, 98.0, 98.98]}]},
           "events": {"splits": {"a": {"date": day(15), "numerator": 4, "denominator": 1},
                                 "b": {"date": day(2), "numerator": 2, "denominator": 1}}}}   # before: already in the price
    f = benchmark.chart_factor(res, dt.date(2025, 1, 14))
    assert abs(f - 0.98 / 4) < 1e-12
    assert benchmark.chart_factor(res, dt.date(2025, 1, 1)) is None   # no close on or before the day


def test_the_stress_test_keeps_its_forecasts_out_of_the_real_ledger():
    src = (pathlib.Path(__file__).resolve().parents[1] / "evaluation" / "stress_test.py").read_text(encoding="utf-8")
    assert 'tools.LEDGER = out / "ledger.jsonl"' in src


def test_the_luck_band_is_wider_for_a_slow_signal(monkeypatch):
    months, px = fake_panel(n_months=150)
    monkeypatch.setitem(benchmark.UNIVERSES, "fake", list(px))
    monkeypatch.setattr(benchmark, "panel", lambda symbols, refresh=False: (months, px))
    keep = {}
    benchmark.run("fake", keep=keep)
    slow = benchmark.luck_band(keep, "Historical mean", 12, seeds=60)
    fast = benchmark.luck_band(keep, "Random (no skill)", 12, seeds=60)
    assert slow[0] < 0 < slow[1] and fast[0] < 0 < fast[1]
    assert slow[1] - slow[0] > 1.5 * (fast[1] - fast[0])
