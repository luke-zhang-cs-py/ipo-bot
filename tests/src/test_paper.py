"""The paper bot (src/paper.py) end to end, offline: exits found by the broker are logged and reach the monitor,
so its cooldown and stoploss guard work across runs; Alpaca's adapter over a faked HTTP layer; the CLI."""
import io
import json
import pathlib
import sys
import urllib.error

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import execution  # noqa: E402
import monitor  # noqa: E402
import paper  # noqa: E402
import portfolio  # noqa: E402

RULES = {"max_position_pct": 10, "max_sector_pct": 100, "risk_per_trade_pct": 1, "min_cash_pct": 5}


def view(cash=100_000):
    return portfolio.valued(portfolio.validate({"as_of": "2026-10-01", "cash": cash, "holdings": [], "rules": RULES}))


def sig(symbol, entry=50.0, stop=45.0):
    return {"symbol": symbol, "entry": entry, "stop": stop, "conviction": "Medium", "sector": "Energy"}


def quotes(*symbols, bid=49.98, ask=50.0):
    return {s: {"bid": bid, "ask": ask, "depth": 1_000_000} for s in symbols}


def day(d):
    return f"2026-10-{d:02d}T14:00:00Z"


# ----------------------------------------------------------------------------- exits reach the monitor

def test_three_stop_outs_in_the_window_block_the_next_buy_and_the_block_survives_a_new_run(tmp_path):
    log = tmp_path / "log.jsonl"
    first = paper.run([sig("AAA"), sig("BBB"), sig("CCC")], view(), execution.SimBroker(quotes("AAA", "BBB", "CCC")),
                      log=log, now=day(1))
    assert all(ln["result"]["status"] == "filled" for ln in first)
    assert set(paper.open_positions(paper.read_log(log))) == {"AAA", "BBB", "CCC"}

    # the next day all three trade under their stops (45): the broker closes them, and DDD's buy is refused
    crashed = execution.SimBroker({**quotes("AAA", "BBB", "CCC", bid=44.0, ask=44.1), **quotes("DDD")})
    second = paper.run([sig("DDD")], view(), crashed, log=log, now=day(2))
    assert second[0]["result"]["status"] == "stopped by the monitor"
    assert second[0]["result"]["blocked_by"].startswith("stoploss guard: 3 stop-outs between 2026-10-02 and 2026-10-02")
    assert crashed.orders == []                                        # nothing reached the broker
    exits = [ln["exit"] for ln in paper.read_log(log) if "exit" in ln]
    assert [(e["symbol"], e["day"], e["stopped"], e["price"]) for e in exits] == [
        ("AAA", "2026-10-02", True, 44.0), ("BBB", "2026-10-02", True, 44.0), ("CCC", "2026-10-02", True, 44.0)]
    assert paper.open_positions(paper.read_log(log)) == {}

    # a later run, with a new monitor and no new exit, still knows about the stop-outs from the log
    third = paper.run([sig("DDD")], view(), execution.SimBroker(quotes("DDD")), log=log, now=day(5))
    assert third[0]["result"]["blocked_by"].startswith("stoploss guard")
    # past the pause (5 days after the last stop-out) buying resumes
    fourth = paper.run([sig("DDD")], view(), execution.SimBroker(quotes("DDD")), log=log, now=day(8))
    assert fourth[0]["result"]["status"] == "filled"
    report = paper.divergence(paper.read_log(log))
    assert report["exits"] == 3 and report["stop_outs"] == 3 and report["signals"] == 6


def test_a_recent_exit_on_the_same_symbol_triggers_the_cooldown_and_only_for_that_symbol(tmp_path):
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now=day(1))
    paper.run([], view(), execution.SimBroker(quotes("AAA", bid=44.0, ask=44.1)), log=log, now=day(2))
    again = paper.run([sig("AAA"), sig("BBB")], view(), execution.SimBroker(quotes("AAA", "BBB")), log=log, now=day(4))
    assert again[0]["result"]["blocked_by"] == "cooldown: AAA was sold on 2026-10-02"
    assert again[1]["result"]["status"] == "filled"                   # one stop-out is not the stoploss guard
    later = paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now=day(6))
    assert later[0]["result"]["status"] == "filled"                   # the cooldown (3 days) has passed


def test_a_position_whose_stop_holds_stays_open_and_a_broker_without_closed_is_not_asked(tmp_path):
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now=day(1))
    paper.run([], view(), execution.SimBroker(quotes("AAA", bid=46.0, ask=46.1)), log=log, now=day(2))
    paper.run([], view(), execution.SimBroker({}), log=log, now=day(2))   # no quote: not checked
    bars = execution.BarBroker({"AAA": (40, 41, 39, 40, 1_000_000)})       # no closed(): nothing is asked
    paper.run([], view(), bars, log=log, now=day(2))
    assert "AAA" in paper.open_positions(paper.read_log(log))


def test_a_monitor_passed_in_keeps_its_own_history_and_still_hears_of_new_exits(tmp_path):
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now=day(1))
    paper.run([], view(), execution.SimBroker(quotes("AAA", bid=40.0, ask=40.1)), log=log, now=day(2))
    paper.run([sig("BBB")], view(), execution.SimBroker(quotes("BBB")), log=log, now=day(3))
    watch = monitor.Monitor(kill_file=tmp_path / "KILL")
    paper.run([], view(), execution.SimBroker(quotes("BBB", bid=40.0, ask=40.1)), log=log, monitor=watch, now=day(4))
    assert watch.exits == [("2026-10-04", "BBB", True)]            # the log's AAA exit was not added again


def test_open_positions_adds_a_second_buy_and_takes_the_newer_stop():
    lines = [{"result": {"stop_order": {"symbol": "aaa", "shares": 5, "stop": 40.0}}},
             {"result": {"stop_order": None}},
             {"result": {"stop_order": {"symbol": "AAA", "shares": 3, "stop": 42.0}}}]
    assert paper.open_positions(lines) == {"AAA": {"shares": 8, "stop": 42.0}}
    assert paper.open_positions(lines + [{"exit": {"symbol": "aaa"}}]) == {}
    assert paper.read_log(pathlib.Path("no-such-dir") / "log.jsonl") == []


def test_an_exit_day_from_the_broker_wins_over_the_run_date(tmp_path):
    class Broker(execution.SimBroker):
        def closed(self, positions):
            return [{"symbol": s, "shares": p["shares"], "price": 1.0, "stopped": False, "day": "2026-09-30"}
                    for s, p in positions.items()]
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now=day(1))
    paper.run([], view(), Broker({}), log=log)                         # no `now`: today, but the broker dated it
    (e,) = [ln["exit"] for ln in paper.read_log(log) if "exit" in ln]
    assert e["day"] == "2026-09-30" and e["stopped"] is False


# ----------------------------------------------------------------------------- Alpaca, faked

class FakeAlpaca:
    """Alpaca's positions and orders endpoints: `held` symbols are open; `sells` per symbol the closed sells."""

    def __init__(self, held=(), sells=None, polls=None):
        self.held, self.sells, self.polls, self.calls = held, sells or {}, list(polls or []), []

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        if url.endswith("/v2/positions"):
            return [{"symbol": s} for s in self.held] or None          # Alpaca's empty body reads as {} -> None
        if "/v2/orders?" in url:
            sym = url.split("symbols=")[1].split("&")[0]
            return self.sells.get(sym, [])
        if method == "POST":
            return {"id": "o1", "status": "new"}
        if method == "DELETE":
            return {}
        return self.polls.pop(0)


def test_alpaca_reports_a_logged_position_it_no_longer_holds_as_an_exit_only_with_a_filled_sell():
    fake = FakeAlpaca(held=["KEEP"], sells={
        "STOP": [{"filled_qty": "0", "type": "limit"}, {"filled_qty": "10", "type": "stop", "filled_avg_price": "44.5",
                                                         "filled_at": "2026-10-02T15:00:00Z"}],
        "UNDER": [{"filled_qty": "10", "type": "market", "filled_avg_price": "44.0"}],
        "OVER": [{"filled_qty": "10", "type": "limit", "filled_avg_price": "60.0"}]})
    broker = paper.AlpacaPaper("k", "s", http=fake, wait_s=0.0)
    pos = {s: {"shares": 10, "stop": 45.0} for s in ("KEEP", "STOP", "UNDER", "OVER", "GONE")}
    out = {e["symbol"]: e for e in broker.closed(pos)}
    assert set(out) == {"STOP", "UNDER", "OVER", "GONE"}
    assert out["STOP"] == {"symbol": "STOP", "shares": 10, "price": 44.5, "stopped": True, "day": "2026-10-02"}
    assert out["UNDER"]["stopped"] is True and out["OVER"]["stopped"] is False
    # gone with no filled sell to show for it (sold by hand, or a sim position): a note, never a priceless exit
    assert out["GONE"] == {"symbol": "GONE", "note": "Alpaca no longer holds GONE but shows no filled sell since its buy: "
                                                     "closed, unpriced (no exit for the monitor)"}
    assert all("side=sell" in url for _, url, _ in fake.calls if "/v2/orders?" in url)
    assert paper.AlpacaPaper("k", "s", http=FakeAlpaca(), wait_s=0.0).closed({"X": {"shares": 1, "stop": 1.0}})[0]["symbol"] == "X"


def test_alpaca_polls_an_open_order_until_it_fills(monkeypatch):
    monkeypatch.setattr(paper.time, "sleep", lambda s: None)
    fake = FakeAlpaca(polls=[{"id": "o1", "status": "new"}, {"id": "o1", "status": "filled", "filled_qty": "4",
                                                              "filled_avg_price": "10.01"}])
    out = paper.AlpacaPaper("k", "s", http=fake, wait_s=60.0).submit("ABC", 4, 10.05)
    assert out["status"] == "filled" and out["avg_price"] == 10.01
    assert [m for m, _, _ in fake.calls] == ["POST", "GET", "GET"]


def test_alpaca_cancels_what_is_left_at_the_deadline_and_reports_it_unfilled():
    fake = FakeAlpaca(polls=[{"id": "o1", "status": "canceled", "filled_qty": "0"}])
    out = paper.AlpacaPaper("k", "s", http=fake, wait_s=0.0).submit("ABC", 4, 10.05)
    assert out == {"order_id": "o1", "filled": 0, "avg_price": None, "fee": 0.0, "status": "unfilled: canceled"}
    assert [m for m, _, _ in fake.calls] == ["POST", "DELETE", "GET"]


class Resp:
    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_real_http_layer_sends_the_keys_and_turns_failures_into_broker_errors(monkeypatch):
    seen = []

    def urlopen(req, timeout):
        seen.append(req)
        return Resp("" if req.full_url.endswith("empty") else '{"ok": 1}')
    monkeypatch.setattr(paper.urllib.request, "urlopen", urlopen)
    broker = paper.AlpacaPaper("key", "secret")
    assert broker._http("POST", paper.ALPACA_PAPER + "/v2/orders", {"a": 1}) == {"ok": 1}
    assert broker._http("GET", paper.ALPACA_PAPER + "/empty") == {}
    assert seen[0].get_header("Apca-api-key-id") == "key" and json.loads(seen[0].data) == {"a": 1} and seen[1].data is None

    def http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"forbidden"))
    monkeypatch.setattr(paper.urllib.request, "urlopen", http_error)
    with pytest.raises(execution.BrokerError, match="Alpaca 403: forbidden"):
        broker._http("GET", paper.ALPACA_PAPER + "/v2/account")

    def url_error(req, timeout):
        raise urllib.error.URLError("no route")
    monkeypatch.setattr(paper.urllib.request, "urlopen", url_error)
    with pytest.raises(execution.BrokerError, match="could not reach Alpaca: no route"):
        broker._http("GET", paper.ALPACA_PAPER + "/v2/account")


# ----------------------------------------------------------------------------- the command line

@pytest.fixture
def cli(tmp_path, monkeypatch):
    import ipo_bot
    monkeypatch.setattr(ipo_bot, "load_env", lambda *a: None)          # never the real .env
    monkeypatch.setattr(paper, "LOG", tmp_path / "paper" / "log.jsonl")
    sigs = tmp_path / "signals.json"
    sigs.write_text(json.dumps([dict(sig("DDD"), avg_volume=500_000)]), encoding="utf-8")
    pf = tmp_path / "pf.json"
    pf.write_text(json.dumps({"as_of": "2026-10-01", "cash": 100_000, "rules": RULES}), encoding="utf-8")
    return sigs, pf


def test_the_cli_prints_its_usage_for_a_bad_command(capsys):
    assert paper.main([]) == 1 and paper.main(["run"]) == 1 and paper.main(["sell"]) == 1
    assert "python src/paper.py run" in capsys.readouterr().out


def test_the_cli_runs_on_the_simulated_broker_and_reports_from_the_log(cli, capsys):
    sigs, pf = cli
    assert paper.main(["run", str(sigs), "--portfolio", str(pf)]) == 0
    assert json.loads(capsys.readouterr().out)["outcomes"] == {"filled": 1}
    assert paper.main(["report"]) == 0
    assert json.loads(capsys.readouterr().out)["signals"] == 1
    paper.LOG.unlink()
    assert paper.main(["report"]) == 0 and json.loads(capsys.readouterr().out)["signals"] == 0


def test_the_cli_uses_the_example_portfolio_and_alpaca_when_asked(cli, monkeypatch, capsys):
    sigs, _ = cli
    made = []

    class Fake(execution.SimBroker):
        def __init__(self):
            made.append(self)
            super().__init__(quotes("DDD", bid=49.98, ask=60.0))
    monkeypatch.setattr(paper, "AlpacaPaper", Fake)
    assert paper.main(["run", str(sigs), "--alpaca"]) == 0
    assert made and json.loads(capsys.readouterr().out)["signals"] == 1
