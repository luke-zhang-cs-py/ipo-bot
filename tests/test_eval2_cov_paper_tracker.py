"""paper_replay's command line and Yahoo reader, and the tracker's run, report and edge cases, offline: Yahoo is a
fake, the universe is cut to a few symbols, and every file goes to tmp_path."""
import datetime as dt
import io
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import paper  # noqa: E402
import paper_replay as R  # noqa: E402
import tracker as T  # noqa: E402


def bars(closes, lows=None, start=dt.date(2025, 1, 6)):
    """[(date, open, high, low, close, volume)] on weekdays; each day opens at yesterday's close."""
    out, day = [], start
    for k, c in enumerate(closes):
        while day.weekday() > 4:
            day += dt.timedelta(days=1)
        o = closes[k - 1] if k else c
        lo = lows[k] if lows and lows[k] is not None else min(o, c) * 0.995
        out.append((day.isoformat(), o, max(o, c) * 1.005, lo, c, 5_000_000))
        day += dt.timedelta(days=1)
    return out


def rising(n=260, step=0.002):
    return [100 * (1 + step) ** k for k in range(n)]


@pytest.fixture
def universe(monkeypatch):
    def use(symbols):
        monkeypatch.setitem(R.bm.UNIVERSES, "US large caps", symbols)
    return use


# ----------------------------------------------------------------------------- paper_replay

def test_bars_reads_yahoo_dropping_incomplete_days_and_gives_up_after_three_failures(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    t = [int(dt.datetime(2025, 3, d, tzinfo=dt.timezone.utc).timestamp()) for d in (3, 4)]
    body = {"chart": {"result": [{"timestamp": t, "indicators": {"quote": [
        {"open": [1.0, 2.0], "high": [1.5, None], "low": [0.5, 1.0], "close": [1.2, 2.1], "volume": [100, 200]}]}}]}}
    monkeypatch.setattr(R.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(json.dumps(body).encode()))
    assert R.bars("AAPL", days=30) == [("2025-03-03", 1.0, 1.5, 0.5, 1.2, 100)]
    calls = []

    def down(req, timeout):
        calls.append(1)
        raise OSError("offline")
    monkeypatch.setattr(R.urllib.request, "urlopen", down)
    assert R.bars("AAPL") == [] and len(calls) == 3


def test_replay_needs_some_prices():
    with pytest.raises(ValueError, match="no price history"):
        R.replay({}, days=3)


def test_fills_text_states_what_happened_to_planned_orders():
    assert R.fills_text({"fill_rate_of_planned_shares": None}) == "No orders were planned."
    none = R.fills_text({"fill_rate_of_planned_shares": 0.0, "mean_slippage_vs_signal": None})
    assert none.startswith("Fill rate of planned shares: 0.0%") and "nothing filled" in none
    some = R.fills_text({"fill_rate_of_planned_shares": 1.0, "mean_slippage_vs_signal": 0.001, "worst_slippage_vs_signal": 0.002,
                         "fees": 0.0, "median_latency_s": 0.0005})
    assert "+0.10%" in some and "worst +0.20%" in some and "0.5 ms" in some


def test_main_writes_the_report_with_stops_missing_symbols_and_warnings(tmp_path, monkeypatch, universe):
    universe(["AAPL", "XOM", "MSFT"])
    monkeypatch.setattr(R, "ROOT", tmp_path)
    lows = [None] * 257 + [1.0, None, None]                          # bought at bar 257's open, stopped the same day
    prices = {"AAPL": bars(rising(), lows=lows), "XOM": bars([100.0] * 260), "MSFT": bars([100.0] * 100), "SPY": bars(rising())}
    monkeypatch.setattr(R, "bars", lambda s, days=420: prices[s])
    divergence = paper.divergence
    monkeypatch.setattr(R.paper, "divergence", lambda lines: {**divergence(lines), "warnings": ["a warning"]})
    out = io.StringIO()                                              # no reconfigure(): skipped
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    R.main(["--days", "3"])
    text = out.getvalue()
    assert "(3 trading days)" in text and "(no prices for MSFT)" in text and "Warnings: a warning" in text
    assert "## Stops hit" in text and "AAPL:" in text and "Fill rate of planned shares" in text
    report = list((tmp_path / "bench_runs").glob("paper_replay_*.md"))
    assert len(report) == 1 and report[0].read_text(encoding="utf-8") in text
    assert list((tmp_path / "forecasts" / "paper").glob("replay_*.jsonl"))


def test_main_with_no_signals_and_with_no_spy(tmp_path, monkeypatch, universe):
    universe(["XOM"])
    monkeypatch.setattr(R, "ROOT", tmp_path)
    prices = {"XOM": bars([100.0] * 260), "SPY": bars([100.0] * 260)}
    monkeypatch.setattr(R, "bars", lambda s, days=420: prices[s])
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="ascii"))
    R.main([])
    R.main([])                                                       # the same day again: the old log is replaced
    sys.stdout.flush()
    text = raw.getvalue().decode("utf-8")
    assert "(10 trading days)" in text and "No orders were planned." in text and "None." in text
    assert len(list((tmp_path / "forecasts" / "paper").glob("replay_*.jsonl"))) <= 1
    prices["SPY"] = []
    with pytest.raises(SystemExit, match="no prices for SPY"):
        R.main([])


# ----------------------------------------------------------------------------- tracker

def test_accordance_on_too_little_history_gives_nans_not_crashes():
    spy = {"2026-01-05": 100.0}
    a = T.accordance([{"day": "2026-01-05", "equity": 1000.0, "cash": 1000.0}], spy, spy)
    assert a["days"] == 1 and all(math.isnan(a[k]) for k in ("beta", "correlation", "tracking_error", "excess_per_year"))
    assert a["invested_when_spy_above_200d"] is None and a["invested_by_regime"] == {"unlabelled": (0.0, 1)}
    assert "beta_ci" not in a
    days = [(dt.date(2026, 1, 5) + dt.timedelta(days=k)).isoformat() for k in range(5)]
    spy = {d: 100.0 + k for k, d in enumerate(days)}
    a = T.accordance([{"day": d, "equity": 1000.0, "cash": 1000.0} for d in days], spy, spy)
    assert a["beta"] == 0.0 and math.isnan(a["correlation"]) and a["tracking_error"] > 0
    assert T.drawdown([100, 50, 100]) == -0.5 and T.returns([1, 2]) == [1.0]


def test_render_says_na_where_nothing_could_be_measured():
    e = {"run": "2026-10-08T12:00:00+00:00", "from": "2026-01-02", "to": "2026-10-07", "days": 1, "account_return": 0.0,
         "spy_return": float("nan"), "account_drawdown": None, "spy_drawdown": 0.0, "correlation": float("nan"),
         "beta": float("nan"), "tracking_error": float("nan"), "excess_per_year": float("nan"),
         "invested_when_spy_above_200d": None, "invested_when_spy_below_200d": None, "days_above_200d": 0,
         "days_below_200d": 0, "invested_by_regime": {"unlabelled": (0.25, 1)}, "signals": 0, "outcomes": {},
         "fill_rate": None, "mean_slippage": None, "stops_hit": 0, "lookahead_found": 0, "lookahead_checked": 0}
    text = T.render(e)
    assert "above its 200-day average: n/a on average (0 days); below it: n/a (0 days)" in text
    assert "unlabelled: 25.0% (1 d)" in text and "Fill rate of planned shares n/a" in text
    assert "95% CI" not in text
    assert "(95% CI 0.100 to 0.200)" in T.render({**e, "beta_ci": (0.1, 0.2)})
    assert T.level(float("nan")) == "n/a" and T.pct(None) == "n/a" and T.rng_text(None, str) == ""


def test_run_records_one_line_per_market_day_and_main_prints_the_report(tmp_path, monkeypatch, universe):
    universe(["AAPL", "XOM"])
    prices = {"AAPL": bars(rising(300)), "XOM": bars([100.0] * 300), "SPY": bars(rising(300, 0.001))}
    monkeypatch.setattr(R, "bars", lambda s, days=420: prices[s])
    monkeypatch.setattr(T, "TRACK_START", prices["SPY"][-15][0])
    monkeypatch.setattr(T, "ROOT", tmp_path)
    monkeypatch.setattr(T, "OUT", tmp_path / "tracking")
    e = T.run()
    assert e["days"] == 15 and e["universe"] == 2 and e["missing"] == [] and e["lookahead_found"] == 0
    assert "beta_ci" in e and e["from"] == prices["SPY"][-15][0] and e["to"] == prices["SPY"][-1][0]
    track = tmp_path / "tracking" / "track.json"
    T.run()                                                         # the same last day: replaced, not duplicated
    assert len(track.read_text(encoding="utf-8").splitlines()) == 1
    old = json.loads(track.read_text(encoding="utf-8"))
    track.write_text(json.dumps({**old, "to": "2000-01-01"}) + "\n", encoding="utf-8")
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    T.main()                                                        # a new day: appended
    assert len(track.read_text(encoding="utf-8").splitlines()) == 2
    assert out.getvalue().startswith("# Tracking: the paper bot against the market")
    assert (tmp_path / "tracking" / "README.md").read_text(encoding="utf-8") in out.getvalue()
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii"))
    T.main()
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").startswith("# Tracking")
