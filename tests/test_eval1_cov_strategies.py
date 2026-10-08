"""strategies.py end to end, offline: the Yahoo fetch and its cache, the two price modes, the engine's corner
cases (no cash to buy, volume caps on both sides), the metrics with no or only winning trades, the nudges, and
the full report on a synthetic market."""
import datetime as dt
import json
import math
import runpy
import sys

import pytest

from test_eval1_cov_helpers import ROOT, Served, chart, no_network, plain_streams

import strategies as S  # noqa: E402


def flat_market(prices):
    """Each day opens at the previous close."""
    dates = [(dt.date(2010, 1, 4) + dt.timedelta(days=k)).isoformat() for k in range(len(prices))]
    return dates, {"SPY": [(prices[max(0, k - 1)], p) for k, p in enumerate(prices)]}


# ----------------------------------------------------------------------------- data

def test_daily_fetches_adjusts_the_open_caches_for_the_day_and_refetches_a_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    days = ["2020-01-02", "2020-01-03", "2020-01-06"]
    served = Served(chart(days, [10.0, None, 30.0], [20.0, 25.0, 30.0], [10.0, 12.5, 30.0],
                          volume=[500, 600, None], dividends={"2020-01-02": 0.5}))
    monkeypatch.setattr(S.urllib.request, "urlopen", served)
    full = S.daily("SPY", full=True)
    # day 1: adjustment 10/20 applied to the open too; day 2 has no open and is dropped; day 3 has no volume
    assert full == [("2020-01-02", 5.0, 10.0, 10.0, 20.0, 500, 0.5), ("2020-01-06", 30.0, 30.0, 30.0, 30.0, 0, 0.0)]
    assert len(served.urls) == 1 and "SPY" in served.urls[0] and "events=div%2Csplit" in served.urls[0]
    cache = json.loads((tmp_path / "bench_data" / "daily2_SPY.json").read_text(encoding="utf-8"))
    assert cache["fetched"] == dt.date.today().isoformat()
    monkeypatch.setattr(S.urllib.request, "urlopen", no_network)          # today's cache: no fetch
    assert S.daily("SPY") == [("2020-01-02", 5.0, 10.0), ("2020-01-06", 30.0, 30.0)]
    assert S.daily("SPY", full=True)[0] == full[0]
    cache["fetched"] = "2000-01-01"                                         # yesterday's (or older): fetched again
    (tmp_path / "bench_data" / "daily2_SPY.json").write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(S.urllib.request, "urlopen", served)
    assert S.daily("SPY") == [("2020-01-02", 5.0, 10.0), ("2020-01-06", 30.0, 30.0)]
    assert len(served.urls) == 2


def synthetic_full(symbol, full=True, n=820):
    """Business days from mid-2014 (so both halves around SPLIT exist): a rising, wavy SPY with a few sharp
    drops, a slow IEF; quarterly dividends in the split-adjusted columns."""
    out, d, k = [], dt.date(2014, 6, 2), 0
    while k < n:
        if d.weekday() < 5:
            if symbol == "SPY":
                c = 100 * math.exp(0.0004 * k + 0.08 * math.sin(k / 40)) * (0.94 if k % 97 == 50 else 1.0)
            else:
                c = 100 + 0.01 * k + 0.5 * math.sin(k / 15)
            o = c * (1 + 0.001 * math.sin(k))
            div = 0.4 if k % 63 == 10 else 0.0
            out.append((d.isoformat(), o, c, o * 1.01, c * 1.01, 2_000_000, div))
            k += 1
        d += dt.timedelta(days=1)
    return out


def test_aligned_keeps_the_common_dates_and_pays_dividends_only_in_split_adjusted_mode(monkeypatch):
    def daily(symbol, full=False):
        rows = synthetic_full(symbol, n=10)
        return rows[1:] if symbol == "IEF" else rows                           # IEF misses the first day
    monkeypatch.setattr(S, "daily", daily)
    dates, bars, extras = S.aligned("SPY", "IEF")
    spy = synthetic_full("SPY", n=10)
    assert dates == [r[0] for r in spy[1:]] and bars["SPY"][0] == (spy[1][1], spy[1][2])
    assert all(v == 0.0 for v in extras["SPY"]["div"]) and extras["IEF"]["volume"][0] == 2_000_000
    d2, b2, e2 = S.aligned("SPY", "IEF", mode="split_adjusted")
    assert b2["SPY"][0] == (spy[1][3], spy[1][4]) and e2["SPY"]["div"] == [r[6] for r in spy[1:]]


def test_without_truststore_the_module_still_loads(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)                    # import truststore -> ImportError
    ns = runpy.run_path(str(ROOT / "evaluation" / "strategies.py"), run_name="strategies_without_truststore")
    assert ns["FEE"] == S.FEE and callable(ns["simulate"])


# ----------------------------------------------------------------------------- the bots and the engine

def test_dca_always_wants_all_spy_and_the_grid_works_without_dates():
    assert S.dca({"amount": 50.0})["contribution"] == 50.0
    assert S.dca()["decide"](3, {}, None) == {"SPY": 1.0}
    bot = S.grid()
    assert bot["decide"](0, {"SPY": [200.0]}, None) == {"SPY": 0.5}           # no dates: one range for all of it
    assert bot["decide"](1, {"SPY": [200.0, 230.0]}, None) == {"SPY": 0.0}    # above the top: holds nothing


def test_mean_reversion_leaves_after_max_hold_days_even_below_the_mean():
    bot = S.mean_reversion({"max_hold": 3})
    xs = [100.0 + (k % 2) * 0.5 for k in range(30)] + [92.0]
    assert bot["decide"](30, {"SPY": xs}, {"SPY": 0.0}) == {"SPY": 1.0}      # in on the drop
    xs += [92.5, 92.6]
    assert bot["decide"](32, {"SPY": xs}, {"SPY": 1.0}) == {"SPY": 1.0}      # still below the mean, 2 days in
    xs += [92.7]
    assert bot["decide"](33, {"SPY": xs}, {"SPY": 1.0}) == {"SPY": 0.0}      # 3 days: out, wherever it is


def test_a_buy_that_cannot_pay_its_commission_is_skipped():
    dates, bars = flat_market([100.0] * 5)
    hold = {"name": "hold", "assets": ["SPY"], "decide": lambda i, c, held: {"SPY": 1.0}}
    run = S.simulate(hold, dates, bars, start_cash=2.0, per_share=True, cost_x=3.0)   # $3 minimum > $2 cash
    assert run["curve"] == [2.0] * 5 and run["trades"] == [] and run["commissions"] == 0.0
    assert run["exposure"] == 0.0


def test_volume_caps_bind_on_buys_and_sells_and_impact_costs_both_ways():
    dates, bars = flat_market([100.0] * 25)
    extras = {"SPY": {"volume": [1000] * 25, "div": [0.0] * 25}}
    in_then_out = {"name": "x", "assets": ["SPY"], "decide": lambda i, c, held: {"SPY": 1.0 if i < 10 else 0.0}}
    run = S.simulate(in_then_out, dates, bars, cost_x=0, extras=extras, volume_share=True)
    impact = S.PRICE_IMPACT * (25 / 1000) ** 2                                # 25 shares = 2.5% of the day
    assert run["capped_fills"] >= 6                                           # 3+ capped buys, 3+ capped sells
    assert run["trades"][0]["pnl"] == pytest.approx(25 * 100 * ((1 - impact) - (1 + impact)))
    assert all(t["pnl"] < 0 for t in run["trades"])                           # flat prices: impact is the only cost
    assert 9_990 < run["curve"][-1] < 10_000
    same = S.simulate(in_then_out, dates, bars, costs=S.Costs(cost_x=0, volume_share=True), extras=extras)
    assert same == run                                                        # a Costs or its fields: one result


def test_costs_rejects_a_setting_it_does_not_have():
    dates, bars = flat_market([100.0] * 3)
    with pytest.raises(TypeError):
        S.simulate(S.trend(), dates, bars, fees=0.0)


# ----------------------------------------------------------------------------- metrics and the report cells

def run_of(trades, curve=(100.0, 101.0)):
    return {"curve": list(curve), "flows": [0.0] * len(curve), "dates": ["2020-01-01", "2021-01-01"][:len(curve)],
            "exposure": 1.0, "trades": [{"pnl": p} for p in trades]}


def test_metrics_and_cells_without_trades_and_with_only_winners():
    none = S.metrics(run_of([]))
    assert none["trades"] == 0 and math.isnan(none["win_rate"]) and math.isnan(none["profit_factor"])
    assert math.isnan(none["expectancy"]) and none["avg_win"] == 0.0 and none["avg_loss"] == 0.0
    assert none["cagr"] == pytest.approx(1.01 ** (365.25 / 366) - 1)          # +1% over 2020's 366 days
    assert S.trade_cells(none) == "n/a | | | | "
    winners = S.metrics(run_of([5.0, 15.0]))
    assert math.isinf(winners["profit_factor"]) and winners["expectancy"] == 10.0
    assert S.trade_cells(winners) == "100.0% | $10.00 | $0.00 | no losses | $10.00"
    flat = S.metrics(run_of([0.0]))                                           # a scratch trade is no win
    assert math.isnan(flat["profit_factor"]) and "n/a" in S.trade_cells(flat)
    mixed = S.metrics(run_of([30.0, -10.0]))
    assert S.trade_cells(mixed) == "50.0% | $30.00 | $10.00 | 3.00 | $10.00"


def test_pct_is_the_benchmark_formatter_at_two_places():
    assert S.pct(0.01234) == "+1.23%" and S.pct(-0.5) == "-50.00%"
    assert S.pct(None) == S.pct(float("nan")) == S.pct(float("inf")) == "n/a"
    assert S.pct(-0.00001) == "0.00%"                                         # no "-0.00%"


def test_window_edge_side_flips_and_buy_and_hold():
    dates, bars = flat_market([100.0, 110.0, 121.0, 133.1])
    d, b = S.window(dates, bars, lo=dates[1], hi=dates[3])
    assert d == dates[1:3] and b["SPY"] == bars["SPY"][1:3]
    assert S.window(dates, bars) == (dates, bars)
    bh = S.buy_and_hold(dates, bars, cost_x=0)
    assert bh["curve"][-1] == pytest.approx(10_000 * 133.1 / 100)            # in at the first open (100)
    assert S.edge({"cagr": 0.05}, {"cagr": 0.08}) == pytest.approx(-0.03)
    assert (S.side(0.0005), S.side(0.01), S.side(-0.01)) == (0, 1, -1)
    assert S.flips(0.0, [("x", 0.5)]) == []                                   # a level base has nothing to flip


def test_a_nudge_that_would_put_the_fast_average_past_the_slow_one_is_dropped(monkeypatch):
    monkeypatch.setattr(S, "NUDGE", 3.0)
    labels = [label for label, _ in S.nudges("trend")]
    assert labels == ["fast 50->-100", "slow 200->-400", "slow 200->800"]   # fast 50 -> 200 is not < slow
    assert S.nudges("dca") == []
    monkeypatch.setattr(S, "NUDGE", 0.2)
    assert ("drift 0.05->0.04", {"drift": pytest.approx(0.04)}) == S.nudges("rebalancing")[0]


# ----------------------------------------------------------------------------- the report

def test_the_report_on_a_synthetic_market(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "daily", synthetic_full)
    monkeypatch.setattr(S, "ROOT", tmp_path)
    out = plain_streams(monkeypatch)
    S.main()
    saved = list((tmp_path / "bench_runs").glob("strategies_*.md"))
    assert len(saved) == 1
    text = saved[0].read_text(encoding="utf-8")
    assert text in out.getvalue() and f"Saved to {saved[0]}" in out.getvalue()
    dates, bars, _ = S.aligned("SPY", "IEF")
    bh = S.metrics(S.buy_and_hold(dates, bars))
    assert f"Daily bars {dates[0]} to {dates[-1]}" in text and "a round trip costs 0.30%" in text
    assert "costs $3.00 each, $120 in all" in text
    assert f"| Buy and hold SPY | {S.pct(bh['net_return'])} | {S.pct(bh['cagr'])} |" in text
    for key, make in S.BOTS.items():
        m = S.metrics(S.simulate(make(), dates, bars))
        assert f"| {make()['name']} | {S.pct(m['net_return'])} | {S.pct(m['cagr'])} | {S.pct(S.edge(m, bh))} |" in text
    stress = text.split("## Stress tests")[1].split("## Regimes")[0]
    assert stress.count("| no parameters |") == 1                            # DCA has none to nudge
    assert stress.count(" | holds |") + stress.count(" | fragile: flips under ") + stress.count("level with holding SPY") == 5
    regimes = text.split("## Regimes")[1].split("## Modelling")[0]
    assert "| DCA |" not in regimes and regimes.count("| Trend 50/200 |") == 1
    assert "a yr (" in regimes and "too few days" in regimes
    cross = text.split("## Modelling cross-checks")[1]
    assert cross.count("(dividends $") == 5 and cross.count("capped)") == 5 and cross.count("(commissions $") == 5
