"""The trading-bot lab on synthetic prices: costs, fills at the next open (never at the close that signalled),
each bot's rule, and the metrics. Offline."""
import datetime as dt
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import strategies as S  # noqa: E402


def market(prices, ief=None):
    """Bars where each day opens at the previous close (so fill prices are easy to check)."""
    dates = [(dt.date(2010, 1, 4) + dt.timedelta(days=k)).isoformat() for k in range(len(prices))]
    bars = {"SPY": [(prices[max(0, k - 1)], p) for k, p in enumerate(prices)]}
    ief = ief or [100.0] * len(prices)
    bars["IEF"] = [(ief[max(0, k - 1)], p) for k, p in enumerate(ief)]
    return dates, bars


def test_the_cost_example_from_the_brief():
    per, total, rate = S.friction(1000, fee=0.001, slippage=0.0005, round_trips=40)
    assert abs(rate - 0.003) < 1e-12 and abs(per - 3.0) < 1e-9 and abs(total - 120.0) < 1e-9


def test_a_signal_fills_at_the_next_open_never_at_its_own_close():
    prices = [100.0] * 220 + [100.0 + k for k in range(1, 60)]
    dates, bars = market(prices)
    run = S.simulate(S.trend(), dates, bars, cost_x=0)
    first_in = next(k for k, v in enumerate(run["curve"]) if abs(v - run["curve"][0]) > 1e-6)
    # rewriting every price from the day after the first fill on changes nothing up to it
    later = prices[:first_in + 1] + [p * 3 for p in prices[first_in + 1:]]
    run2 = S.simulate(S.trend(), *market(later), cost_x=0)
    assert run["curve"][:first_in + 1] == run2["curve"][:first_in + 1]
    late = S.simulate(S.trend(), dates, bars, cost_x=0, delay=1)
    assert next(k for k, v in enumerate(late["curve"]) if abs(v - late["curve"][0]) > 1e-6) == first_in + 1


def test_costs_only_ever_lower_the_result():
    prices = [100 + 10 * math.sin(k / 9) + k * 0.05 for k in range(600)]
    dates, bars = market(prices)
    for make in S.BOTS.values():
        ends = [S.simulate(make(), dates, bars, cost_x=x)["curve"][-1] for x in (0.0, 1.0, 2.0)]
        assert ends[0] >= ends[1] >= ends[2], make()["name"]


def test_trend_is_in_a_rising_market_and_out_of_a_falling_one():
    up = [100 * 1.001 ** k for k in range(400)]
    down = [100 * 0.999 ** k for k in range(400)]
    assert S.trend()["decide"](399, {"SPY": up}, None) == {"SPY": 1.0}
    assert S.trend()["decide"](399, {"SPY": down}, None) == {"SPY": 0.0}
    assert S.trend()["decide"](100, {"SPY": up}, None) == {"SPY": 0.0}        # not enough history for 200 days yet


def test_dca_puts_every_contribution_to_work():
    prices = [100.0] * 70
    dates, bars = market(prices)
    run = S.simulate(S.dca(), dates, bars, cost_x=0)
    weeks = len({dt.date.fromisoformat(d).isocalendar()[:2] for d in dates})
    assert sum(run["flows"]) == 100.0 * weeks
    assert abs(run["curve"][-1] - 100.0 * weeks) < 1.0                       # flat prices, no costs: worth what went in
    m = S.metrics(run)
    assert abs(m["net_return"]) < 0.01 and abs(m["max_drawdown"]) < 1e-9     # contributions are not counted as gains


def test_rebalancing_only_trades_when_the_drift_reaches_the_band():
    bot = S.rebalancing()
    assert bot["decide"](5, {}, {"SPY": 0.63, "IEF": 0.37}) == {"SPY": 0.63, "IEF": 0.37}
    assert bot["decide"](5, {}, {"SPY": 0.66, "IEF": 0.34}) == {"SPY": 0.6, "IEF": 0.4}
    assert bot["decide"](0, {}, None) == {"SPY": 0.6, "IEF": 0.4}


def test_mean_reversion_buys_a_sharp_drop_and_sells_the_recovery():
    bot = S.mean_reversion()
    xs = [100.0 + (k % 2) * 0.5 for k in range(30)] + [92.0]
    assert bot["decide"](30, {"SPY": xs}, {"SPY": 0.0}) == {"SPY": 1.0}
    xs2 = xs + [101.0]
    assert bot["decide"](31, {"SPY": xs2}, {"SPY": 1.0}) == {"SPY": 0.0}


def test_the_grid_holds_more_the_further_the_price_falls_and_all_of_it_below_the_range():
    bot = S.grid()
    d = {0: "2020-01-02"}
    assert bot["decide"](0, {"SPY": [100.0]}, None, d) == {"SPY": 0.5}       # mid-range: half the levels
    w95 = bot["decide"](1, {"SPY": [100.0, 95.0]}, None, {1: "2020-01-03"})["SPY"]
    w85 = bot["decide"](2, {"SPY": [100.0, 95.0, 85.0]}, None, {2: "2020-01-06"})["SPY"]
    assert 0.5 < w95 < w85 == 1.0                                             # a breakout down: fully loaded


def test_metrics_from_known_trades():
    run = {"curve": [100.0, 110.0, 99.0, 120.0], "flows": [0, 0, 0, 0], "dates": ["2020-01-01", "2020-06-01", "2021-01-01", "2022-01-01"],
           "exposure": 0.5, "trades": [{"pnl": 30.0}, {"pnl": -10.0}, {"pnl": 20.0}, {"pnl": -10.0}]}
    m = S.metrics(run)
    assert m["trades"] == 4 and m["win_rate"] == 0.5 and m["avg_win"] == 25.0 and m["avg_loss"] == 10.0
    assert m["profit_factor"] == 2.5 and m["expectancy"] == 7.5
    assert abs(m["max_drawdown"] - (99 / 110 - 1)) < 1e-12 and abs(m["net_return"] - 0.2) < 1e-12


def test_every_bot_has_nudges_that_stay_valid():
    for key in S.BOTS:
        for label, params in S.nudges(key):
            S.BOTS[key](params)                                                # constructs
            if key == "trend":
                assert {**S.trend()["params"], **params}["fast"] < {**S.trend()["params"], **params}["slow"]


def test_fragility_needs_a_real_flip_not_noise_around_zero():
    assert S.flips(0.02, [("2x", 0.01), ("late", -0.03)]) == ["late"]
    assert S.flips(0.0, [("2x", -0.0001), ("late", 0.0002)]) == []          # level with SPY: nothing to flip
    assert S.flips(-0.02, [("half", 0.0005)]) == []                        # into the level zone is not across it


def test_dividends_are_paid_in_cash_and_volume_and_commission_models_apply():
    dates, bars = market([100.0] * 30)
    extras = {"SPY": {"volume": [1_000_000] * 30, "div": [0.0] * 20 + [1.0] + [0.0] * 9},
              "IEF": {"volume": [1_000_000] * 30, "div": [0.0] * 30}}
    hold = {"name": "hold", "assets": ["SPY"], "decide": lambda i, closes, held: {"SPY": 1.0}}
    run = S.simulate(hold, dates, bars, cost_x=0, extras=extras)
    assert abs(run["dividends"] - 100.0) < 1.0                                   # 100 shares x $1, in cash
    assert abs(run["curve"][-1] - 10_100.0) < 1.0
    thin = {"SPY": {"volume": [1000] * 30, "div": [0.0] * 30}, "IEF": extras["IEF"]}
    capped = S.simulate(hold, dates, bars, cost_x=0, extras=thin, volume_share=True)
    assert capped["capped_fills"] > 0                                            # 2.5% of 1,000 shares a day
    per = S.simulate(hold, dates, bars, cost_x=1, extras=extras, per_share=True)
    assert abs(per["commissions"] - 1.0) < 1e-9                                  # 100 shares x $0.005 -> the $1 minimum


def test_the_bias_checks_pass_honest_rules_and_catch_a_cheat():
    import bias_checks as B
    prices = [100 + 5 * math.sin(k / 7) + 0.03 * k for k in range(400)]
    dates, bars = market(prices)
    closes = {a: [c for _, c in bars[a]] for a in bars}
    days, bad = B.lookahead(S.trend, closes, dates, samples=10)
    assert days and bad == []
    cheat = lambda: {"name": "cheat", "assets": ["SPY"], "params": {},
                     "decide": lambda i, c, held: {"SPY": 1.0 if i + 1 < len(c["SPY"]) and c["SPY"][i + 1] > c["SPY"][i] else 0.0}}
    _, caught = B.lookahead(cheat, closes, dates, samples=10)
    assert caught, "a rule that reads tomorrow's close must be flagged"
    warm = B.recursive(S.trend, closes, dates, warmups=(None, 300))
    assert len(set(map(str, warm.values()))) == 1
