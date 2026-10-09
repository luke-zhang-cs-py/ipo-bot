"""Orders after sizing: partial fills, thin markets, broker errors, Alpaca's paper API (faked), and the
signal-to-execution divergence report. Offline; no account."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import execution  # noqa: E402
import monitor  # noqa: E402
import paper  # noqa: E402
import portfolio  # noqa: E402

EXAMPLE = json.loads((pathlib.Path(__file__).resolve().parents[2] / "portfolio.example.json").read_text(encoding="utf-8"))


def view(**rules):
    return portfolio.valued(portfolio.validate({**EXAMPLE, "rules": {**EXAMPLE["rules"], **rules}}))


SIGNAL = {"symbol": "DDD", "entry": 50.0, "stop": 45.0, "conviction": "Medium", "sector": "Energy"}


def deep(symbol="DDD", ask=50.0, depth=1_000_000):
    return {symbol: {"bid": ask - 0.02, "ask": ask, "depth": depth}}


def test_a_deep_market_fills_the_whole_plan_and_the_stop_covers_it():
    out = execution.execute_buy(view(), SIGNAL, execution.SimBroker(deep()))
    assert out["status"] == "filled" and out["filled"] == out["planned"] > 0
    assert out["stop_order"] == {"symbol": "DDD", "shares": out["filled"], "stop": 45.0}
    assert out["loss_at_stop"] <= out["risk_budget"] + 1e-6


def test_partial_fills_are_worked_but_never_exceed_the_plan_the_rules_or_the_risk_budget():
    broker = execution.SimBroker(deep(depth=8), impact=0.0)             # eight shares on show at a time
    for q in broker.quotes.values():
        q["depth"] = 8
    out = execution.execute_buy(view(), SIGNAL, broker)
    assert len(out["orders"]) >= 2 and out["orders"][0]["status"] == "partial"
    assert 0 < out["filled"] <= out["planned"]
    assert out["loss_at_stop"] <= out["risk_budget"] + 1e-6
    assert out["stop_order"]["shares"] == out["filled"]                 # protect what was bought, not what was asked


def test_a_price_that_climbs_while_filling_shrinks_the_rest_to_stay_inside_the_risk_budget():
    broker = execution.SimBroker(deep(depth=30), impact=0.004)
    out = execution.execute_buy(view(), SIGNAL, broker)
    assert out["loss_at_stop"] <= out["risk_budget"] + 1e-6, out
    assert all(o.get("avg_price") is None or o["avg_price"] <= o["limit"] + 1e-9 for o in out["orders"])


def test_a_liquidity_spike_leaves_the_order_unfilled_rather_than_chasing_the_price():
    broker = execution.SimBroker({"DDD": {"bid": 49.9, "ask": 50.1, "depth": 5000}})
    broker.spike(spread_x=10)                                            # the ask jumps to about 51
    out = execution.execute_buy(view(), SIGNAL, broker)
    assert out["filled"] == 0 and out["status"].startswith("unfilled") and out["stop_order"] is None


def test_a_broker_error_is_recorded_not_raised():
    out = execution.execute_buy(view(), SIGNAL, execution.SimBroker(deep(), fail_rate=1.0))
    assert out["status"] == "api error" and out["filled"] == 0 and "503" in out["orders"][0]["status"]


def test_a_blocked_signal_sends_no_order():
    broker = execution.SimBroker(deep())
    v = portfolio.valued(portfolio.validate({**EXAMPLE, "day_pnl": -10_000, "rules": {**EXAMPLE["rules"], "max_daily_loss_pct": 2}}))
    out = execution.execute_buy(v, SIGNAL, broker)
    assert out["status"] == "blocked" and out["blocked_by"] == "daily_loss" and broker.orders == []


class FakeAlpaca:
    """Alpaca's order endpoints, faked: the order fills `fill` of the shares, or is rejected."""

    def __init__(self, fill=1.0, reject=False):
        self.fill, self.reject, self.calls = fill, reject, []

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        if method == "POST":
            self.qty = int(body["qty"])
            if self.reject:
                return {"id": "o1", "status": "rejected", "reject_reason": "insufficient buying power"}
            return {"id": "o1", "status": "new"}
        filled = int(self.qty * self.fill)
        return {"id": "o1", "status": "filled" if filled == self.qty else "partially_filled", "filled_qty": str(filled),
                "filled_avg_price": "50.02" if filled else None}


def test_the_alpaca_adapter_sends_a_paper_limit_order_and_reads_the_fill():
    fake = FakeAlpaca()
    broker = paper.AlpacaPaper("k", "s", http=fake, wait_s=0.0)
    assert broker.submit("DDD", 10, 50.25) == {"order_id": "o1", "filled": 10, "avg_price": 50.02, "fee": 0.0, "status": "filled"}
    method, url, body = fake.calls[0]
    assert method == "POST" and url.startswith(paper.ALPACA_PAPER) and body["type"] == "limit" and body["limit_price"] == "50.25"
    part = paper.AlpacaPaper("k", "s", http=FakeAlpaca(fill=0.5), wait_s=0.0).submit("DDD", 10, 50.25)
    assert part["status"] == "partial" and part["filled"] == 5
    with pytest.raises(execution.BrokerError, match="buying power"):
        paper.AlpacaPaper("k", "s", http=FakeAlpaca(reject=True), wait_s=0.0).submit("DDD", 10, 50.25)
    with pytest.raises(execution.BrokerError, match="paper"):
        paper.AlpacaPaper("k", "s", base="https://api.alpaca.markets")             # never the live endpoint
    with pytest.raises(execution.BrokerError, match="ALPACA_KEY_ID"):
        paper.AlpacaPaper(key_id="", secret="")


def test_the_divergence_report_explains_every_signal(tmp_path):
    signals = [dict(SIGNAL), dict(SIGNAL, symbol="EEE"), dict(SIGNAL, symbol="FFF", entry=20.0, stop=25.0), dict(SIGNAL, symbol="GGG")]
    broker = execution.SimBroker({**deep(), **deep("EEE", ask=60.0), **deep("GGG", depth=3)}, impact=0.0)
    for q in broker.quotes.values():
        if q["depth"] == 3:
            q["depth"] = 3
    lines = paper.run(signals, view(), broker, log=tmp_path / "log.jsonl")
    d = paper.divergence(lines)
    assert d["signals"] == 4 and sum(d["outcomes"].values()) == 4
    assert d["outcomes"].get("filled") == 1                       # DDD
    assert any(k.startswith("unfilled") for k in d["outcomes"])    # EEE: the ask is 20% over the entry
    assert d["outcomes"].get("rejected by the rules") == 1         # FFF: a stop above the entry
    assert d["outcomes"].get("partial") == 1                       # GGG: three shares on show
    assert 0 < d["fill_rate_of_planned_shares"] < 1
    assert len((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()) == 4


# ----------------------------------------------------------------------------- the monitor and the audit log

NOW = "2026-10-07T14:30:00Z"
QUOTE = {"bid": 49.98, "ask": 50.02, "time": "2026-10-07T14:29:50Z"}


def test_the_monitor_stops_on_each_failure_it_watches_for(tmp_path):
    m = monitor.Monitor(kill_file=tmp_path / "KILL")
    sig = dict(SIGNAL, quote=QUOTE)
    assert m.gate(sig, now=NOW) is None
    assert m.gate(dict(SIGNAL, quote={**QUOTE, "time": "2026-10-07T14:20:00Z"}), now=NOW).startswith("stale data")
    assert m.gate(dict(SIGNAL, quote={**QUOTE, "bid": 49.0, "ask": 51.0}), now=NOW).startswith("abnormal spread")
    assert m.gate(dict(SIGNAL, quote={"bid": 50.6, "ask": 50.62, "time": QUOTE["time"]}), now=NOW).startswith("price deviation")
    assert m.gate(dict(SIGNAL, quote={**QUOTE, "bid": 51.0}), now=NOW).startswith("bad quote")       # crossed
    assert m.gate(dict(SIGNAL, quote={**QUOTE, "time": None}), now=NOW).startswith("stale data")
    (tmp_path / "KILL").write_text("stop")
    assert m.gate(sig, now=NOW).startswith("kill switch")


def test_three_api_failures_in_a_row_pause_the_bot_and_a_good_order_resets_the_count(tmp_path):
    m = monitor.Monitor(kill_file=tmp_path / "KILL")
    for _ in range(2):
        m.record({"status": "api error", "orders": [{}]})
    assert m.gate(SIGNAL) is None
    m.record({"status": "filled", "orders": [{}]})
    assert m.failures == 0
    for _ in range(3):
        m.record({"status": "api error", "orders": [{}]})
    assert m.gate(SIGNAL).startswith("paused")
    m.reset()
    assert m.gate(SIGNAL) is None


def test_the_paper_log_is_an_audit_trail_with_order_ids_fees_and_monitor_stops(tmp_path):
    broker = execution.SimBroker({**deep(), **deep("EEE", ask=60.0)}, fee_rate=0.001)
    sigs = [dict(SIGNAL, quote=QUOTE), dict(SIGNAL, symbol="EEE", quote={**QUOTE, "time": "2026-10-07T13:00:00Z"})]
    lines = paper.run(sigs, view(), broker, log=tmp_path / "log.jsonl", now=NOW)
    first, second = (ln["result"] for ln in lines)
    assert first["status"] == "filled" and first["orders"][0]["order_id"].startswith("sim-")
    assert abs(first["fees"] - 0.001 * first["filled"] * first["avg_price"]) < 0.01
    assert second["status"] == "stopped by the monitor" and second["blocked_by"].startswith("stale data")
    d = paper.divergence(lines)
    assert d["fees"] > 0 and "monitor: stale data" in d["outcomes"]
    assert broker.orders == [("DDD", first["planned"], first["orders"][0]["limit"])]        # nothing sent for EEE


def test_api_failures_through_paper_trading_pause_later_signals(tmp_path):
    broker = execution.SimBroker(deep(), fail_rate=1.0)
    lines = paper.run([dict(SIGNAL)] * 5, view(), broker, log=tmp_path / "log.jsonl")
    statuses = [ln["result"]["status"] for ln in lines]
    assert statuses[:3] == ["api error"] * 3 and statuses[3:] == ["stopped by the monitor"] * 2
    assert len(broker.orders) == 3


# ----------------------------------------------------------------------------- bar fills and protections

def test_bar_fills_need_the_price_to_trade_through_the_limit_and_respect_volume():
    b = execution.BarBroker({"DDD": (50.0, 51.0, 49.0, 50.5, 1_000_000)}, price_impact=0.0)
    assert b.submit("DDD", 100, 50.2)["avg_price"] == 50.0                       # opened under the limit: the open
    gap = execution.BarBroker({"DDD": (51.0, 51.5, 50.1, 51.2, 1_000_000)}, price_impact=0.0)
    assert gap.submit("DDD", 100, 50.2)["avg_price"] == 50.2                     # traded down through it: the limit
    touch = execution.BarBroker({"DDD": (51.0, 51.5, 50.2, 51.2, 1_000_000)})
    assert touch.submit("DDD", 100, 50.2)["filled"] == 0                         # only touched it: no fill
    thin = execution.BarBroker({"DDD": (50.0, 50.5, 49.5, 50.2, 2_000)})
    first, second = thin.submit("DDD", 100, 50.2), thin.submit("DDD", 100, 50.2)
    assert first["filled"] == 50 and first["status"] == "partial" and second["filled"] == 0     # 2.5% of 2,000, once
    impact = execution.BarBroker({"DDD": (50.0, 50.5, 49.0, 50.2, 1000)}, volume_limit=1.0)
    assert 50.0 < impact.submit("DDD", 500, 60.0)["avg_price"] <= 50.0 * (1 + 0.1 * 0.25) + 1e-9


def test_cooldown_and_stoploss_guard(tmp_path):
    m = monitor.Monitor(kill_file=tmp_path / "KILL")
    m.record_exit("2026-10-05", "DDD", stopped=False)
    assert m.gate(SIGNAL, now="2026-10-07T14:00:00Z").startswith("cooldown")
    assert m.gate(SIGNAL, now="2026-10-09T14:00:00Z") is None                    # four days on: allowed
    assert m.gate(dict(SIGNAL, symbol="EEE"), now="2026-10-07T14:00:00Z") is None
    for d, s in (("2026-10-01", "A"), ("2026-10-04", "B"), ("2026-10-08", "C")):
        m.record_exit(d, s, stopped=True)
    assert m.gate(dict(SIGNAL, symbol="ZZZ"), now="2026-10-10T14:00:00Z").startswith("stoploss guard")
    assert m.gate(dict(SIGNAL, symbol="ZZZ"), now="2026-10-14T14:00:00Z") is None   # the pause is over
