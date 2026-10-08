"""benchmark.py offline: the month-end fetch and its cache, the entry factor's fetch and memo, the statistics'
edge cases, the ledger's maturity, the shared pct, and the full report on a made-up universe."""
import datetime as dt
import json
import math
import runpy
import sys

import pytest

from test_eval1_cov_helpers import ROOT, Served, chart, no_network, plain_streams

import benchmark as bm  # noqa: E402
from test_benchmark import fake_panel  # noqa: E402


def test_fetch_keeps_the_last_close_of_each_month_drops_this_month_and_caches_for_the_day(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "CACHE", tmp_path / "cache")
    this_month = dt.date.today().replace(day=1).isoformat()
    days = ["2020-01-30", "2020-01-31", "2020-02-28", this_month]
    served = Served(chart(days, [1.0] * 4, [1.0] * 4, [10.0, 11.0, None, 12.0]))
    monkeypatch.setattr(bm.urllib.request, "urlopen", served)
    assert bm.fetch("XLK") == {"2020-01": 11.0}                 # Feb has no adjusted close; this month is not over
    monkeypatch.setattr(bm.urllib.request, "urlopen", no_network)
    assert bm.fetch("XLK") == {"2020-01": 11.0}                 # today's cache
    monkeypatch.setattr(bm.urllib.request, "urlopen", served)
    bm.fetch("XLK", refresh=True)                                # --refresh ignores the cache
    path = tmp_path / "cache" / "XLK.json"
    path.write_text(json.dumps({"fetched": "2000-01-01", "months": {}}), encoding="utf-8")
    assert bm.fetch("XLK") == {"2020-01": 11.0} and len(served.urls) == 3     # a stale cache is fetched again


def test_entry_factor_fetches_once_per_symbol_and_day(monkeypatch):
    monkeypatch.setattr(bm, "_factors", {})
    payload = chart(["2025-01-14", "2025-01-15"], [100.0, 100.0], [100.0, None], [98.0, 99.0],
                    splits={"2025-01-15": (4, 1)})
    served = Served(payload)
    monkeypatch.setattr(bm.urllib.request, "urlopen", served)
    assert bm.entry_factor("AAA", "2025-01-15") == pytest.approx(0.98)   # the 15th has no close: the 14th's
    assert bm.entry_factor("AAA", "2025-01-15") == pytest.approx(0.98) and len(served.urls) == 1
    assert bm.entry_factor("AAA", "2025-01-14") == pytest.approx(0.98 / 4)  # the split after it is undone


def test_chart_factor_runs_off_the_end_and_ignores_a_split_without_ratio():
    payload = chart(["2025-01-13", "2025-01-14"], [1, 1], [100.0, 100.0], [99.0, 98.0],
                    splits={"2025-01-20": (0, 1)})["chart"]["result"][0]
    assert bm.chart_factor(payload, dt.date(2025, 3, 1)) == pytest.approx(0.98)
    del payload["events"]
    assert bm.chart_factor(payload, dt.date(2025, 3, 1)) == pytest.approx(0.98)


def test_panel_keeps_only_the_months_every_symbol_has(monkeypatch):
    prices = {"A": {"2020-01": 1.0, "2020-02": 2.0}, "B": {"2020-02": 3.0, "2020-03": 4.0}}
    monkeypatch.setattr(bm, "fetch", lambda s, refresh=False: prices[s])
    assert bm.panel(["A", "B"]) == (["2020-02"], {"A": [2.0], "B": [3.0]})


def test_the_statistics_on_too_little_or_constant_data():
    assert math.isnan(bm.newey_west_t([1.0, 2.0], 0))
    assert math.isnan(bm.newey_west_t([1.0, 1.0, 1.0], 1))
    assert bm.newey_west_t([1.0, 2.0, 3.0, 4.0], 0) == pytest.approx(2.5 / math.sqrt(1.25 / 4))


def test_pct_one_formatter_for_every_report():
    assert bm.pct(0.1234) == "+12.3%" and bm.pct(-0.1234, 2) == "-12.34%"
    assert bm.pct(-0.0001) == "0.0%" and bm.pct(0) == "0.0%"
    assert bm.pct(None) == bm.pct(float("nan")) == bm.pct(float("-inf")) == "n/a"


def test_maturity_and_is_due():
    made, due, end = bm.maturity({"date": "2024-11-20", "horizon_months": 3})
    assert (made, due, end) == (dt.date(2024, 11, 20), dt.date(2025, 2, 1), "2025-01")
    assert bm.maturity({"date": "2024-01-31", "horizon_months": 12})[1:] == (dt.date(2025, 1, 1), "2024-12")
    assert bm.is_due(dt.date(2025, 2, 1), today=dt.date(2025, 2, 1))
    assert not bm.is_due(dt.date(2025, 3, 1), today=dt.date(2025, 2, 27))
    assert bm.is_due(dt.date(2000, 1, 1)) and not bm.is_due(dt.date.today() + dt.timedelta(days=40))


def write_ledger(tmp_path, monkeypatch, rows):
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(bm, "LEDGER", path)


def test_score_ledger_without_a_ledger_and_without_an_adjusted_price(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "LEDGER", tmp_path / "missing.jsonl")
    assert bm.score_ledger().startswith("No bot forecasts logged yet")
    write_ledger(tmp_path, monkeypatch, [{"date": "2024-01-15", "symbol": "AAA", "rating": "Overweight",
                                          "expected_return_pct": 10, "price": 100, "horizon_months": 12}])
    monkeypatch.setattr(bm, "fetch", lambda s, refresh=False: {"2024-12": 120.0})
    asked = []
    monkeypatch.setattr(bm, "entry_factor", lambda s, day: asked.append(day))        # None: no close on file
    text = bm.score_ledger()
    assert "1 matured, 0 pending" in text and "AAA: no adjusted price for 2024-01-15" in text
    assert asked == ["2024-01-15"] and "Direction right" not in text and "Next to mature" not in text


def report_universe(monkeypatch, n_months=110):
    months, px = fake_panel(n_months=n_months)
    monkeypatch.setattr(bm, "UNIVERSES", {"Toy": list(px)})
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (months, {s: px[s] for s in symbols}))
    band = bm.luck_band
    monkeypatch.setattr(bm, "luck_band", lambda keep, algo, h, seeds=None: band(keep, algo, h, seeds=20))
    return months, px


def test_the_report_and_main_on_a_made_up_universe(tmp_path, monkeypatch):
    months, px = report_universe(monkeypatch)
    monkeypatch.setattr(bm, "ROOT", tmp_path)
    monkeypatch.setattr(bm, "LEDGER", tmp_path / "none.jsonl")
    out = plain_streams(monkeypatch)
    assert bm.main(["--ledger"]) == 0
    saved = list((tmp_path / "bench_runs").glob("*.md"))
    text = saved[0].read_text(encoding="utf-8")
    assert len(saved) == 1 and text in out.getvalue()
    assert f"## Toy (6 names, predictions every month-end {months[bm.LOOKBACK]} to {months[-2]})" in text
    assert "### 12-month horizon (" in text and "### 1-month horizon (" in text
    _, per = bm.run("Toy")
    for r in bm.summary(per, 12):
        assert f"| {r['algo']} | {r['ic']:+.3f} |" in text
    labels, _ = bm.blocks(per, 12)
    assert labels[0] == months[bm.LOOKBACK][:4] + "-" + str(int(months[bm.LOOKBACK][:4]) + 2)[2:]
    curves = bm.backtest(per, months)
    assert "Equal-weight universe" in curves and f"| Equal-weight universe | {bm.pct(curves['Equal-weight universe']['cagr'])} |" in text
    assert "## The bot's own forecasts" in text and "No bot forecasts logged yet" in text
    assert "yes, better" in text                                             # momentum on trending stocks
    assert "## The bot's own forecasts" not in bm.report()                  # only with --ledger


def test_backtest_sharpe_is_nan_without_volatility():
    per = {1: {"Flat": [{"top_ret": 0.01, "all_ret": 0.01}] * 12}}
    out = bm.backtest(per, None)
    assert out["Flat"]["cagr"] == pytest.approx(1.01 ** 12 - 1) and out["Flat"]["mdd"] == 0.0
    assert math.isnan(out["Flat"]["sharpe"]) and math.isnan(out["Equal-weight universe"]["sharpe"])


def test_without_truststore_the_module_still_loads(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)
    ns = runpy.run_path(str(ROOT / "evaluation" / "benchmark.py"), run_name="benchmark_without_truststore")
    assert ns["BLOCK_YEARS"] == bm.BLOCK_YEARS
