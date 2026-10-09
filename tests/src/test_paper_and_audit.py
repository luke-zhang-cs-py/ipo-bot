"""Fix round 4: paper trading's broker reconciliation, outages, report and dates; the auditor's NOT RATED exemption;
tools' truncation; the test kit's file names."""
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "testkit"))

import audit  # noqa: E402
import execution  # noqa: E402
import kit  # noqa: E402
import monitor  # noqa: E402
import paper  # noqa: E402
import portfolio  # noqa: E402
import tools  # noqa: E402

RULES = {"max_position_pct": 10, "max_sector_pct": 100, "risk_per_trade_pct": 1, "min_cash_pct": 5}
D = "For information only, not investment advice; do your own research or consult a licensed professional."


def view():
    return portfolio.valued(portfolio.validate({"as_of": "2026-10-01", "cash": 100_000, "holdings": [], "rules": RULES}))


def sig(symbol, entry=50.0, stop=45.0):
    return {"symbol": symbol, "entry": entry, "stop": stop, "conviction": "Medium", "sector": "Energy"}


def quotes(*symbols, bid=49.98, ask=50.0):
    return {s: {"bid": bid, "ask": ask, "depth": 1_000_000} for s in symbols}


class FakeAlpaca:
    def __init__(self, held=(), sells=None, fail=False):
        self.held, self.sells, self.fail, self.urls = held, sells or {}, fail, []

    def __call__(self, method, url, body=None):
        self.urls.append(url)
        if self.fail:
            raise execution.BrokerError("Alpaca 503: service unavailable")
        if url.endswith("/v2/positions"):
            return [{"symbol": s} for s in self.held]
        return self.sells.get(url.split("symbols=")[1].split("&")[0], [])


# ----------------------------------------------------------------------------- 1. brokers and phantom exits

def test_an_alpaca_run_after_a_sim_run_writes_no_exits_for_the_sim_positions(tmp_path):
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA"), sig("BBB")], view(), execution.SimBroker(quotes("AAA", "BBB")), log=log, now="2026-10-01T14:00Z")
    assert {ln["broker"] for ln in paper.read_log(log)} == {"sim"}
    fake = FakeAlpaca()                                            # Alpaca holds none of them: they were never its
    paper.run([], view(), paper.AlpacaPaper("k", "s", http=fake, wait_s=0.0), log=log, now="2026-10-02T14:00Z")
    assert fake.urls == []                                         # not even asked: no Alpaca positions in the log
    assert not [ln for ln in paper.read_log(log) if "exit" in ln]
    assert set(paper.open_positions(paper.read_log(log), broker="sim")) == {"AAA", "BBB"}
    assert paper.open_positions(paper.read_log(log), broker="alpaca") == {}


def test_a_gone_alpaca_position_with_no_filled_sell_gets_a_note_not_an_exit(tmp_path):
    log = tmp_path / "log.jsonl"
    old = {"result": {"stop_order": {"symbol": "OLD", "shares": 5, "stop": 40.0}}}       # before brokers were recorded
    log.write_text(json.dumps(old) + "\n" + json.dumps({"broker": "alpaca", "logged": "2026-10-01T15:00:00+00:00",
                   "result": {"stop_order": {"symbol": "NEW", "shares": 4, "stop": 40.0}}}) + "\n", encoding="utf-8")
    paper.run([], view(), paper.AlpacaPaper("k", "s", http=FakeAlpaca(), wait_s=0.0), log=log, now="2026-10-02T14:00Z")
    added = paper.read_log(log)[2:]
    assert [ln.get("note", "")[:28] for ln in added] == ["Alpaca no longer holds OLD b", "Alpaca no longer holds NEW b"]
    assert all(ln["broker"] == "alpaca" and "exit" not in ln for ln in added)
    assert set(paper.open_positions(paper.read_log(log))) == {"OLD", "NEW"}    # still open, still the monitor's


# ----------------------------------------------------------------------------- 2. a broker outage

def test_a_broker_error_while_checking_positions_logs_a_warning_and_the_signals_still_run(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"broker": "alpaca", "result": {"stop_order": {"symbol": "AAA", "shares": 1, "stop": 1.0}}})
                   + "\n", encoding="utf-8")

    class Down(execution.SimBroker):
        name = "alpaca"

        def closed(self, positions):
            raise execution.BrokerError("Alpaca 503: down")
    out = paper.run([sig("BBB")], view(), Down(quotes("BBB")), log=log, now="2026-10-02T14:00Z")
    assert out[0]["result"]["status"] == "filled"
    (warn,) = [ln for ln in paper.read_log(log) if "warning" in ln]
    assert warn["warning"] == "could not check open positions with the broker: Alpaca 503: down"


# ----------------------------------------------------------------------------- 3. the run's report counts its exits

def test_main_reports_the_exits_logged_this_run(tmp_path, monkeypatch, capsys):
    import ipo_bot
    monkeypatch.setattr(ipo_bot, "load_env", lambda *a: None)
    monkeypatch.setattr(paper, "LOG", tmp_path / "paper" / "log.jsonl")
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=paper.LOG, now="2026-10-01T14:00Z")
    sigs = tmp_path / "s.json"
    sigs.write_text(json.dumps([sig("AAA", entry=40.0, stop=35.0)]), encoding="utf-8")   # AAA's bid 39.96 < its stop 45
    assert paper.main(["run", str(sigs)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["exits"] == 1 and rep["stop_outs"] == 1 and rep["signals"] == 1     # this run only, not day 1's signal


# ----------------------------------------------------------------------------- 4. partial sells, averaged

def test_the_exit_averages_every_filled_sell_since_the_last_buy():
    sells = {"AAA": [
        {"filled_qty": "6", "type": "limit", "filled_avg_price": "52.0", "filled_at": "2026-10-03T15:00:00Z"},
        {"filled_qty": "4", "type": "stop", "filled_avg_price": "44.0", "filled_at": "2026-10-02T15:00:00Z"},
        {"filled_qty": "9", "type": "limit", "filled_avg_price": "10.0", "filled_at": "2026-09-01T15:00:00Z"}],  # before the buy
        "BBB": [{"filled_qty": "2.5", "type": "limit", "filled_avg_price": "60.0"}]}       # fractional, undated
    fake = FakeAlpaca(sells=sells)
    pos = {"AAA": {"shares": 10, "stop": 45.0, "since": "2026-10-01T14:00:00+00:00"}, "BBB": {"shares": 3, "stop": 45.0}}
    out = {e["symbol"]: e for e in paper.AlpacaPaper("k", "s", http=fake, wait_s=0.0).closed(pos)}
    assert out["AAA"] == {"symbol": "AAA", "shares": 10, "price": 48.8, "stopped": True, "day": "2026-10-03"}
    assert out["BBB"] == {"symbol": "BBB", "shares": 2.5, "price": 60.0, "stopped": False}
    aaa_q = [u for u in fake.urls if "symbols=AAA" in u][0]
    assert f"limit={paper.SELL_ORDERS_LIMIT}" in aaa_q and "after=2026-10-01T14" in aaa_q
    assert "after=" not in [u for u in fake.urls if "symbols=BBB" in u][0]


def test_open_positions_keeps_when_the_last_buy_was_logged_and_skips_other_brokers():
    lines = [{"broker": "sim", "logged": "t1", "result": {"stop_order": {"symbol": "A", "shares": 1, "stop": 1.0}}},
             {"broker": "alpaca", "logged": "t2", "result": {"stop_order": {"symbol": "A", "shares": 2, "stop": 2.0}}},
             {"broker": "sim", "exit": {"symbol": "A"}}]
    assert paper.open_positions(lines, broker="alpaca") == {"A": {"shares": 2, "stop": 2.0, "since": "t2"}}
    assert paper.open_positions(lines, broker="sim") == {}
    assert paper.open_positions(lines) == {}


# ----------------------------------------------------------------------------- 5. one calendar: UTC

def test_an_evening_run_in_new_york_is_dated_by_its_utc_day():
    assert monitor.utc_day("2026-10-02T21:00:00-04:00") == dt.date(2026, 10, 3)
    assert monitor.utc_day(None) == dt.datetime.now(dt.timezone.utc).date()
    m = monitor.Monitor(kill_file=pathlib.Path("no-such-KILL"), cooldown_days=0)
    m.record_exit("2026-10-03", "AAA", False)                     # Alpaca's filled_at date for a 01:00 UTC fill
    assert m.gate(sig("AAA"), now="2026-10-02T21:00:00-04:00") == "cooldown: AAA was sold on 2026-10-03"


def test_paper_dates_its_exits_by_the_utc_day(tmp_path):
    log = tmp_path / "log.jsonl"
    paper.run([sig("AAA")], view(), execution.SimBroker(quotes("AAA")), log=log, now="2026-10-01T14:00Z")
    paper.run([], view(), execution.SimBroker(quotes("AAA", bid=40.0, ask=40.1)), log=log, now="2026-10-02T21:00:00-04:00")
    (e,) = [ln["exit"] for ln in paper.read_log(log) if "exit" in ln]
    assert e["day"] == "2026-10-03"


# ----------------------------------------------------------------------------- 6. the auditor's exemption

def test_a_rated_memo_that_mentions_not_rated_still_owes_a_block():
    _, mech = audit.build_request(f"Overweight. Peer FOO is NOT RATED.\n{D}", [])
    assert mech[0][1] is False and mech[0][0] == "KEY NUMBERS block present and valid"
    _, mech = audit.build_request(f"NOT RATED: no data. Peers are Overweight.\n{D}", [])
    assert mech == [(audit.NOT_NEEDED, True, "not needed: the answer gives no rating")]
    assert audit.own_rating("Rating: Underweight; not NOT RATED") == "Underweight" and audit.own_rating("none") is None


# ----------------------------------------------------------------------------- 7. truncation and file names

def test_one_long_string_is_cut_and_the_short_list_keeps_every_item(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 1_000)
    data = {"items": ["a", "b", "c", "x" * 5_000]}
    text = tools._result("s", data)
    out = json.loads(text)
    assert len(text) <= 1_000 and out["truncated"] is True
    assert out["data"]["items"][:3] == ["a", "b", "c"] and 500 < len(out["data"]["items"][3]) < 1_000   # kept what fits


def test_trim_drops_a_single_item_that_cannot_shrink_and_leaves_unshrinkable_leaves_alone():
    assert tools._trim([12345], 3) == []
    assert tools._trim(7, 3) == 7 and tools._trim("short", 3) == "short" and tools._trim([], 3) == []
    assert tools._trim("y" * 300, 1_000) == "y" * tools.LONG_STRING


def test_names_that_clean_to_the_same_file_get_their_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(kit, "RUNS", tmp_path)
    for name, memo in (("BRK/B", "slash"), ("BRK_B", "underscore"), ("BRK.B?", "q1"), ("BRK/B", "again")):
        kit.Runner.save("bt", name, "q", memo, [], [])
    d = tmp_path / kit.STAMP / "bt"
    assert "again" in (d / "BRK_B.md").read_text() and "underscore" in (d / "BRK_B-2.md").read_text()
    assert "q1" in (d / "BRK.B_.md").read_text() and (d / "BRK_B-2.sources.json").exists()


def test_paper_replay_uses_the_shared_console_helper():
    import common
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "evaluation"))
    import paper_replay
    assert paper_replay.common is common
    assert "reconfigure" not in pathlib.Path(paper_replay.__file__).read_text(encoding="utf-8")
