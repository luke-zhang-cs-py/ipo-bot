"""Portfolio mode: loading and checking the file, valuing it, the sizing rules, and the tools and CLI around them.
Offline: prices come from the file. Numbers here are made up for the tests.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ipo_bot  # noqa: E402
import portfolio  # noqa: E402
import tools  # noqa: E402

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "portfolio.example.json"


def pf(**over):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update(over)
    return portfolio.validate(raw)


def view(**over):
    return portfolio.valued(pf(**over))


def test_the_example_values_add_up():
    v = view()
    # 20 x 180 + 40 x 55 + 15 x 100 = 3600 + 2200 + 1500 = 7300 invested, plus 4000 cash
    assert v["equity"] == 11300
    assert {h["symbol"]: round(h["weight_pct"], 2) for h in v["holdings"]} == {"AAA": 31.86, "BBB": 19.47, "CCC": 13.27}
    assert round(v["cash_pct"], 2) == 35.4
    assert v["holdings"][0]["gain_pct"] == 20.0


def test_breaches_are_found_with_the_shares_to_trim():
    b = view()["rule_breaches"]
    pos = [x for x in b if x["rule"] == "max_position_pct"]
    # AAA is 3600 of 11300; the 25% cap is 2825, so 775 over at 180 a share: 5 shares (rounded up)
    assert pos == [{"rule": "max_position_pct", "symbol": "AAA", "weight_pct": 31.86, "shares_over": 5}]
    assert not [x for x in b if x["rule"] == "max_sector_pct"], "Technology is 31.9%, under the 40% cap"


def test_risk_sets_the_size_when_it_is_tightest():
    s = portfolio.size_position(view(), "DDD", 50, 45, "Medium", "Energy")
    # 1% of 11300 = 113 at risk; 113 / (50 - 45) = 22.6 -> 22 shares
    assert s["shares"] == 22 and s["binding_rule"] == "risk"
    assert s["loss_at_stop"] == 110.0 and s["cost"] == 1100.0 and s["cash_after"] == 2900.0


def test_conviction_scales_the_risk_budget():
    low = portfolio.size_position(view(), "DDD", 50, 45, "Low", "Energy")
    high = portfolio.size_position(view(), "DDD", 50, 45, "High", "Energy")
    assert (low["shares"], high["shares"]) == (11, 33)      # 56.5 / 5 and 169.5 / 5, rounded down


def test_the_position_cap_counts_what_is_already_held():
    # AAA is already over its 25% cap: there is no room to add, whatever the risk budget allows
    s = portfolio.size_position(view(), "AAA", 180, 170, "High")
    assert s["shares"] == 0 and s["binding_rule"] == "position"


def test_the_sector_cap_and_the_cash_floor():
    tech = portfolio.size_position(view(), "EEE", 10, 9.9, "High", "Technology")
    # 40% of 11300 = 4520 for Technology; AAA has 3600 of it: 920 left at 10 = 92 shares
    assert tech["shares"] == 92 and tech["binding_rule"] == "sector"
    poor = portfolio.size_position(view(cash=600), "DDD", 50, 49.9, "High", "Energy")
    # equity 7900; 5% floor is 395 of cash; 205 spendable at 50 = 4 shares
    assert poor["shares"] == 4 and poor["binding_rule"] == "cash"


def test_sizing_refuses_a_stop_at_or_above_the_entry():
    for stop in (50, 55):
        with pytest.raises(portfolio.PortfolioError):
            portfolio.size_position(view(), "DDD", 50, stop, "Medium")
    with pytest.raises(portfolio.PortfolioError):
        portfolio.size_position(view(), "DDD", 50, 45, "Certain")


@pytest.mark.parametrize("bad, word", [
    ({"cash": -1}, "cash"), ({"cash": "lots"}, "cash"),
    ({"holdings": [{"symbol": "A", "shares": 1}, {"symbol": "a", "shares": 2}]}, "twice"),
    ({"holdings": [{"shares": 1}]}, "symbol"),
    ({"holdings": [{"symbol": "A", "shares": 1, "price": 0}]}, "price"),
    ({"rules": {"max_position_pct": 150}}, "at most 100"),
])
def test_a_bad_file_is_refused_with_what_is_wrong(bad, word):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update(bad)
    with pytest.raises(portfolio.PortfolioError, match=word):
        portfolio.validate(raw)


def test_an_unpriced_holding_is_named_and_left_out():
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["holdings"].append({"symbol": "ZZZ", "shares": 5, "sector": "Energy"})
    v = portfolio.valued(portfolio.validate(raw))
    assert v["unpriced"] == ["ZZZ"] and v["equity"] == 11300


def test_the_tools_need_a_loaded_portfolio_and_read_only_that_file(monkeypatch):
    monkeypatch.setitem(tools.PORTFOLIO, "path", None)
    text, err = tools.run_tool("portfolio_view", {})
    assert err and "--portfolio" in text
    monkeypatch.setitem(tools.PORTFOLIO, "path", str(EXAMPLE))
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    text, err = tools.run_tool("portfolio_view", {})
    data = json.loads(text)["data"]
    assert not err and data["equity"] == 11300 and data["holdings"][0]["price_source"].startswith("portfolio file")
    text, err = tools.run_tool("portfolio_size", {"symbol": "DDD", "entry_price": 50, "stop_price": 45,
                                                  "conviction": "Medium", "sector": "Energy"})
    assert not err and json.loads(text)["data"]["shares"] == 22
    text, err = tools.run_tool("portfolio_size", {"symbol": "DDD", "entry_price": 50, "stop_price": 60, "conviction": "Medium"})
    assert err and "below the entry" in text
    assert tools.run_tool("portfolio_view", {"path": "/etc/passwd"})[1], "the model cannot name a file"


def test_the_cli_checks_the_file_and_tells_the_model_portfolio_mode_is_on(monkeypatch, tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="Portfolio file"):
        ipo_bot.main(["--portfolio", str(bad)])
    monkeypatch.setitem(tools.PORTFOLIO, "path", str(EXAMPLE))
    from test_offline import FakeClient, GOOD, text_block
    from types import SimpleNamespace as NS
    client = FakeClient([NS(stop_reason="end_turn", content=[text_block(GOOD)])])
    ipo_bot.Bot(client=client, log=lambda s: None).ask("Should I buy DDD?")
    assert "Portfolio mode: on" in client.requests[0]["messages"][0]["content"]
    monkeypatch.setitem(tools.PORTFOLIO, "path", None)
