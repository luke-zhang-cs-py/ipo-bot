"""Regression tests for the code-scan fixes in src/: each failed before its fix. Offline."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import audit  # noqa: E402
import execution  # noqa: E402
import monitor  # noqa: E402
import paper  # noqa: E402
import portfolio  # noqa: E402
import tools  # noqa: E402
import track  # noqa: E402
import verify  # noqa: E402

EXAMPLE = json.loads((pathlib.Path(__file__).resolve().parents[1] / "portfolio.example.json").read_text(encoding="utf-8"))


def view(**extra):
    return portfolio.valued(portfolio.validate({**EXAMPLE, **extra}))


def test_a_fill_at_the_limit_stays_inside_the_risk_budget():
    # entry 100, stop 99: a fill at the 0.5% limit (100.45) costs 1.45 a share at the stop, not 1.00
    signal = {"symbol": "RRR", "entry": 100.0, "stop": 99.0, "conviction": "High", "sector": "Energy"}
    broker = execution.SimBroker({"RRR": {"bid": 100.43, "ask": 100.45, "depth": 10_000_000}}, impact=0.0)
    out = execution.execute_buy(view(), signal, broker)
    assert out["filled"] > 0 and out["loss_at_stop"] <= out["risk_budget"] + 1e-6, out
    assert out["planned"] <= out["sized_at_entry"]


def test_guardrail_state_survives_a_fill():
    v = view(equity_high=1_000_000.0)
    after = execution.after_fill(v, "XYZ", 10, 50.0, "Energy", fee=1.5)
    assert after["equity_high"] == 1_000_000.0 and after["halted"] is False
    assert after["cash"] == pytest.approx(v["cash"] - 501.5)
    halted = execution.after_fill(view(halted=True), "XYZ", 1, 50.0, "Energy")
    assert halted["halted"] is True


def test_no_attempts_is_not_an_index_error():
    signal = {"symbol": "DDD", "entry": 50.0, "stop": 45.0, "conviction": "Medium", "sector": "Energy"}
    out = execution.execute_buy(view(), signal, execution.SimBroker({"DDD": {"bid": 49.98, "ask": 50.0, "depth": 10**6}}),
                                attempts=0)
    assert out["status"] == "not submitted" and out["filled"] == 0


@pytest.mark.parametrize("expr", ["exp(1000)", "1e10**100", "((10**100)**100)**100"])
def test_the_calculator_reports_huge_results_instead_of_crashing(expr):
    text, is_error = tools.run_tool("calculate", {"expression": expr})
    assert is_error and "too large" in text


def test_a_read_that_times_out_is_a_tool_error(monkeypatch):
    class Slow:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            raise TimeoutError("timed out")

    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda *a, **k: Slow())
    with pytest.raises(tools.ToolError, match="failed mid-response"):
        tools._get("https://example.test/x")


def test_the_ledger_folder_is_created_with_its_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEDGER", tmp_path / "runs" / "stamp" / "ledger.jsonl")
    tools.record_forecast("ABC", "Overweight", 20.0, 10.0, "2026-01-02", 12, 14.0, 11.0, 8.0, 25, 50, 25)
    assert (tmp_path / "runs" / "stamp" / "ledger.jsonl").exists()


def memo(block):
    return "A memo.\n```json\n" + json.dumps({"key_numbers": block}) + "\n```\n"


def test_verify_fails_bad_shapes_instead_of_crashing():
    for block in ({"inputs": ["a"]}, {"outputs": {"multiples": ["P/E"]}}, {"segments": ["x", {"parts": ["y"]}]}):
        rows = verify.check(verify.extract(memo(block)))
        assert any(not ok for _, ok, _ in rows), block


def test_pwv_uses_the_checked_numbers():
    scen = {"bull": {"value": "130", "prob": 25}, "base": {"value": 110, "prob": 50}, "bear": {"value": "80", "prob": "25"},
            "pwv": 107.5}
    rows = {name: ok for name, ok, _ in verify.check({"rating": "Overweight", "scenarios": scen})}
    assert rows["PWV matches the probabilities and values"] is True


def test_an_invalid_snippet_does_not_hide_the_real_block():
    text = memo({"rating": "Hold"}) + "\nExample:\n```json\n{not json\n```\n"
    assert verify.extract(text) == {"rating": "Hold"}
    with pytest.raises(verify.BlockError, match="not valid JSON"):
        verify.extract("```json\n{oops\n```")


def test_sizing_a_held_symbol_with_no_price_is_refused():
    raw = {**EXAMPLE, "holdings": [*EXAMPLE["holdings"], {"symbol": "NOPR", "shares": 1000, "cost_basis": 10.0}]}
    v = portfolio.valued(portfolio.validate(raw))
    assert "NOPR" in v["unpriced"]
    with pytest.raises(portfolio.PortfolioError, match="no price"):
        portfolio.size_position(v, "NOPR", 10.0, 9.0, "Medium")


@pytest.mark.parametrize("base", ["https://evil.example/paper-api", "http://paper-api.alpaca.markets",
                                  "https://paper-api.alpaca.markets.evil.example"])
def test_alpaca_keys_go_only_to_the_paper_host(base):
    with pytest.raises(execution.BrokerError):
        paper.AlpacaPaper("k", "s", base=base, http=lambda *a: {})
    assert paper.AlpacaPaper("k", "s", http=lambda *a: {}).base == paper.ALPACA_PAPER


def test_an_unreadable_quote_time_is_stale_data():
    m = monitor.Monitor()
    reason = m.gate({"quote": {"time": "yesterday-ish", "bid": 10, "ask": 10.01, "last": 10}}, now="2026-01-02T15:00:00Z")
    assert reason and reason.startswith("stale data")


def test_an_unreadable_audit_reply_is_a_fail_verdict():
    out = audit.parse("{cut off mid", [("PWV", True, "")])
    assert out["verdict"] == "fail" and out["mechanical"][0]["pass"] is True


def test_one_unpriceable_ticker_leaves_its_row_pending(monkeypatch):
    import urllib.error

    def gone(symbol, start, end):
        raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

    monkeypatch.setattr(track, "closes", gone)
    import datetime as dt
    assert track.close_on("GONE", dt.date(2025, 1, 2)) is None
