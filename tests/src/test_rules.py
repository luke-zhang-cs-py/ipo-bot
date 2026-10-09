"""The rule code under the bot (src/checks.py, verify.py, portfolio.py, execution.py, monitor.py, track.py), offline:
memo checks, the KEY NUMBERS verifier, sizing, brokers, the monitor's guards and the forecast tracker (Yahoo faked)."""
import datetime as dt
import json
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import audit  # noqa: E402
import checks  # noqa: E402
import execution  # noqa: E402
import monitor  # noqa: E402
import portfolio  # noqa: E402
import track  # noqa: E402
import verify  # noqa: E402

D = checks.DISCLAIMER


# ----------------------------------------------------------------------------- checks.py

def test_the_no_guarantee_line_is_not_hype_but_a_guarantee_still_is():
    memo = f"{checks.GUARANTEE_LINE}. Prices can fall.\n{D}"
    got = dict(checks.check(memo, expect=("guaranteed",)))
    assert got["no hype or guarantees"] and got["says no investment is guaranteed"]   # R3 can pass now
    assert not checks.no_hype("Returns are guaranteed.\n" + D)
    assert not checks.no_hype(f"{checks.GUARANTEE_LINE}, but this one is guaranteed to double.\n{D}")


def test_each_case_rule_checks_what_it_names():
    good = ("NEUTRAL, confidence medium. Sizes follow the rules in your portfolio file. "
            f"{checks.PUMP_LINE}. I can't find that company.\n{D}")
    got = dict(checks.check(good, expect=("no_personal_amount", "market_call", "portfolio", "injection", "pump",
                                          "cant_find", "no_live_figures", "key_numbers", "not_rated")))
    assert all(got[k] for k in ("no amount or share of money for this person", "ends the market view with a call and confidence",
                                "portfolio sizes come from the user's rules", "ignores instructions planted in retrieved text",
                                "refuses to write promotional content", "says it can't find it instead of guessing",
                                "gives no rating for what does not exist", "gives no price, rate or figure without data tools",
                                "no data, no rating"))
    assert got["KEY NUMBERS: KEY NUMBERS block present and valid"] is False
    bad = f"You should invest $500. STRONG BUY at $12, up 5%.\n{D}"
    got = dict(checks.check(bad, expect=("no_personal_amount", "injection", "no_live_figures", "market_call")))
    assert not any(got[k] for k in ("no amount or share of money for this person", "ignores instructions planted in retrieved text",
                                    "gives no price, rate or figure without data tools", "ends the market view with a call and confidence"))
    assert checks.scenario_probabilities("BULL 30%\nBASE 50%\nBEAR 20%\nbull again 99%") == {"bull": 30.0, "base": 50.0, "bear": 20.0}


# ----------------------------------------------------------------------------- the auditor's mechanical part

def test_an_honest_no_rating_answer_without_a_block_passes_the_mechanical_part():
    for memo in (f"I can't find any company by that name.\n{D}", f"NOT RATED: no data.\n{D}"):
        _, mech = audit.build_request(memo, [])
        assert mech == [("KEY NUMBERS block (not needed: not rated)", True, "not needed: the answer gives no rating")]
    _, mech = audit.build_request(f"Overweight. I can't find the 2024 figure.\n{D}", [])   # rated: a block is owed
    assert mech[0][1] is False
    _, mech = audit.build_request(f"NOT RATED\n```json\n{{bad\n```\n{D}", [])             # a broken block still fails
    assert mech[0][1] is False and "not valid JSON" in mech[0][2]


# ----------------------------------------------------------------------------- verify.py

def block(k):
    return f"memo\n```json\n{json.dumps({'key_numbers': k})}\n```\n"


def test_extract_skips_json_without_key_numbers_and_reports_a_bad_block():
    memo = block({"rating": "NOT RATED"}) + '```json\n{"other": 1}\n```'
    assert verify.extract(memo) == {"rating": "NOT RATED"}
    with pytest.raises(verify.BlockError, match="no ```json block"):
        verify.extract('```json\n{"other": 1}\n```')


@pytest.mark.parametrize("k", [
    {"subject": "X"}, {"subject": ["X"]}, {"unknown": [{"a": 1}], "sources": 3, "flags": "stale"},
    {"segments": 5, "outputs": {"multiples": "EV/Sales"}}, {"segments": [{"total": ["x"], "parts": 7}]},
    {"rating": ["Overweight"], "scenarios": {}}, {"outputs": {"multiples": [{"numerator": ["a"], "denominator": {"b": 1}}]}},
    {"rating": "Overweight", "conviction": ["High"], "scenarios": {c: {"value": v, "prob": p} for c, v, p in
                                                                   (("bull", 3, 30), ("base", 2, 40), ("bear", 1, 30))}
     | {"expected_return_pct": 20}},
])
def test_wrong_shapes_are_failed_checks_never_a_crash(k):
    results = verify.check(k)
    assert results and not all(ok for _, ok, _ in results)


def test_a_subject_that_is_not_an_object_is_named():
    assert ("subject is an object", False, "str") in verify.check({"subject": "X"})


def full_block(**over):
    k = {
        "as_of": "2026-10-09", "subject": {"company": "Co", "ticker": "CO", "exchange": "NYSE", "share_class": "A"},
        "sources": ["s"], "flags": ["outlier: big multiple", "stale"],
        "inputs": {"price": {"value": 10.0, "source": "x", "as_of": "2026-10-01", "period": "LTM"},
                   "diluted_shares": {"value": 100.0, "source": "x", "as_of": "2026-10-01"},
                   "debt": {"value": 50.0, "source": "x", "as_of": "2026-10-01"},
                   "cash": {"value": 20.0, "source": "x", "as_of": "2026-10-01"},
                   "revenue": {"value": 10.0, "source": "x", "as_of": "2026-10-01", "period": "LTM"},
                   "revenue_prior": {"value": 2.0, "source": "x", "as_of": "2026-10-01"},
                   "net_income": {"value": -30.0, "source": "x", "as_of": "2026-10-01"},
                   "ebitda": {"value": None}, "bad": 5},
        "unknown": ["ebitda"],
        "outputs": {"market_cap": 1000.0, "enterprise_value": 1030.0,
                    "multiples": [{"name": "EV/Revenue LTM", "numerator": "enterprise_value", "denominator": "revenue", "value": 103.0},
                                  {"name": "EV/EBITDA", "numerator": "enterprise_value", "denominator": "ebitda", "value": 1}, 7]},
        "segments": [{"total": "revenue", "parts": [{"value": 6.0}, {"value": 4.0}]}],
        "rating": "Overweight", "conviction": "High", "stop_working_price": 8.0,
        "scenarios": {"bull": {"value": 60.0, "prob": 25}, "base": {"value": 12.0, "prob": 50}, "bear": {"value": 1.0, "prob": 25},
                      "pwv": 21.25, "reference_price": 10.0, "expected_return_pct": 112.5},
        "ipo_ratings": {"at_offer": "Participate", "aftermarket": "Buy below", "buy_below": round(21.25 / 1.15, 2)},
    }
    k.update(over)
    return k


def test_a_full_block_runs_every_rule():
    got = {n: ok for n, ok, _ in verify.check(full_block())}
    for name in ("market cap = price x fully diluted shares", "EV = market cap + debt + preferred + minority interest - cash",
                 "multiple EV/Revenue LTM recomputes", "segments add up to revenue", "stale price is flagged",
                 "EV/revenue above 50x is flagged", "revenue growth above 300% is flagged",
                 "net_income margin outside -100%..100% is flagged", "bull value beyond 5x / one-fifth of the price is flagged",
                 "rating follows the thresholds", "buy-below = PWV / 1.15", "stop-working price below the reference price",
                 "Participate only with an Overweight rating", "ebitda: missing value is listed in unknown"):
        assert got[name], name
    assert got["every multiple is an object"] is False and got["multiple EV/EBITDA: inputs present"] is False
    assert got["input bad is an object"] is False


def test_missing_inputs_for_given_outputs_and_bad_scenarios_fail():
    k = full_block(inputs={}, scenarios={"bull": {"value": "x"}})
    got = {n: ok for n, ok, _ in verify.check(k)}
    assert got["market cap has both inputs"] is False and got["EV has its inputs"] is False
    assert got["scenarios have bull, base and bear values and probabilities"] is False
    assert {n: ok for n, ok, _ in verify.check(full_block(rating="NOT RATED"))}["NOT RATED has no scenarios"] is False
    assert {n: ok for n, ok, _ in verify.check(full_block(scenarios=None))}["a rated stock has scenarios"] is False
    assert verify._trading_days_between(dt.date(2026, 1, 2), dt.date(2026, 1, 1)) == 0


def test_the_verify_cli_prints_text_or_json_and_exits_1_on_a_failure(tmp_path, capsys):
    with pytest.raises(SystemExit):
        verify.main([])
    good = tmp_path / "good.md"
    good.write_text(block(full_block()), encoding="utf-8")
    assert verify.main([str(good)]) == 1                               # the block has deliberate failures
    out = capsys.readouterr().out
    assert "FAIL  every multiple is an object" in out and "checks pass" in out
    nr = tmp_path / "nr.md"
    nr.write_text(block({"as_of": "2026-01-01", "rating": "NOT RATED", "sources": ["s"], "subject": {
        "company": "C", "ticker": "C", "exchange": "X", "share_class": "A"}}), encoding="utf-8")
    assert verify.main([str(nr), "--json"]) == 0
    assert all(r["pass"] for r in json.loads(capsys.readouterr().out))


# ----------------------------------------------------------------------------- portfolio.py

def test_portfolio_load_reports_an_unreadable_file(tmp_path):
    with pytest.raises(portfolio.PortfolioError, match="cannot read"):
        portfolio.load(tmp_path / "missing.json")


def test_a_cash_shortfall_is_a_breach_and_an_unknown_sector_has_no_sector_limit():
    pf = portfolio.validate({"as_of": "2026-10-01", "cash": 1, "holdings": [{"symbol": "A", "shares": 100, "price": 10.0}],
                             "rules": {"max_position_pct": 100, "max_sector_pct": 100}})
    v = portfolio.valued(pf)
    assert {"rule": "min_cash_pct", "cash_pct": 0.1} in v["rule_breaches"]
    sized = portfolio.size_position(v, "NEW", 5.0, 4.0, now=dt.datetime(2026, 10, 1), next_earnings="2026-12-01")
    assert "sector" not in sized["limits_in_shares"]


def test_earnings_blackout_reads_a_naive_datetime_as_utc():
    rules = {"earnings_blackout_hours": 48}
    v = portfolio.valued(portfolio.validate({"as_of": "2026-10-01", "cash": 10_000, "rules": rules}))
    near = portfolio.size_position(v, "X", 10.0, 9.0, next_earnings="2026-10-02T12:00:00Z", now=dt.datetime(2026, 10, 1, 12))
    assert near["shares"] == 0 and near["binding_rule"] == "earnings_blackout"


# ----------------------------------------------------------------------------- execution.py

def view(cash=100_000, holdings=()):
    return portfolio.valued(portfolio.validate({"as_of": "2026-10-01", "cash": cash, "holdings": list(holdings),
                                                "rules": {"max_position_pct": 50, "max_sector_pct": 100}}))


SIG = {"symbol": "AAA", "entry": 50.0, "stop": 45.0, "sector": "Energy"}


def test_sim_broker_waits_its_latency_and_refuses_an_unknown_symbol(monkeypatch):
    slept = []
    monkeypatch.setattr(execution.time, "sleep", slept.append)
    broker = execution.SimBroker({}, latency_s=0.25)
    with pytest.raises(execution.BrokerError, match="no quote for ZZZ"):
        broker.submit("ZZZ", 1, 10.0)
    assert slept == [0.25]
    bars = execution.BarBroker({})
    with pytest.raises(execution.BrokerError, match="no bar for ZZZ"):
        bars.submit("ZZZ", 1, 10.0)


def test_a_partial_fill_stops_retrying_when_the_rules_leave_no_room():
    broker = execution.SimBroker({"AAA": {"bid": 49.9, "ask": 50.0, "depth": 1_000_000}}, impact=0.0)
    calls = []

    def submit(symbol, shares, limit):
        calls.append(shares)
        return {"filled": shares - 1, "avg_price": 51.0, "fee": 0.0, "status": "partial"}   # filled above the limit
    broker.submit = submit
    out = execution.execute_buy(view(cash=6_000), dict(SIG, stop=49.0), broker)
    # 47 shares at 51 already risk more than the budget (60), so nothing is left to send for the last share
    assert calls == [48] and out["filled"] == 47 and out["status"] == "partial"


def test_after_fill_averages_the_cost_of_a_held_stock_and_marks_it_at_the_fill():
    v = view(holdings=[{"symbol": "AAA", "shares": 10, "cost_basis": 40.0, "price": 45.0, "sector": "Energy"},
                       {"symbol": "BBB", "shares": 1, "price": 5.0}])
    after = execution.after_fill(v, "AAA", 10, 50.0, "Energy", fee=1.0)
    aaa = next(h for h in after["holdings"] if h["symbol"] == "AAA")
    assert aaa["shares"] == 20 and aaa["cost_basis"] == 45.0 and aaa["price"] == 50.0
    assert after["cash"] == v["cash"] - 501.0
    assert after["equity"] == pytest.approx(v["equity"] - 1.0 + 10 * (50.0 - 45.0))   # the old shares re-marked
    nocost = execution.after_fill(after, "BBB", 1, 6.0, None)
    assert next(h for h in nocost["holdings"] if h["symbol"] == "BBB")["cost_basis"] is None


# ----------------------------------------------------------------------------- monitor.py

def test_the_monitor_guards_in_order():
    m = monitor.Monitor(kill_file=pathlib.Path("no-such-dir") / "KILL")
    now = "2026-10-05T14:00:00"
    q = {"bid": 10.0, "ask": 10.01, "time": "2026-10-05T13:59:30Z"}
    assert m.gate({"symbol": "A", "entry": 10.0, "quote": q}, now=now) is None
    assert m.gate({"symbol": "A", "quote": dict(q, time=None)}, now=now) == "stale data: the quote has no time"
    assert m.gate({"symbol": "A", "quote": dict(q, time="2026-10-05T13:00:00Z")}, now=now).startswith("stale data: the quote is")
    assert m.gate({"symbol": "A", "quote": dict(q, bid=None)}, now=now).startswith("bad quote")
    assert m.gate({"symbol": "A", "quote": dict(q, ask=11.0)}, now=now).startswith("abnormal spread")
    assert m.gate({"symbol": "A", "entry": 9.0, "quote": q}, now=now).startswith("price deviation")
    assert m.gate({"symbol": "A"}) is None                              # no quote, no `now`: nothing to check
    m.record({"status": "api error"})
    m.record({"status": "partial, then an api error"})
    m.record({"status": "blocked"})                                     # no orders: the count stands
    m.record({"status": "api error"})
    assert m.gate({"symbol": "A"}, now=now) == "paused: 3 API failures in a row"
    m.record({"status": "filled", "orders": [{}]})
    assert m.failures == 0
    m.failures = 5
    m.reset()
    assert m.gate({"symbol": "A"}, now=now) is None


def test_the_kill_file_stops_everything(tmp_path):
    (tmp_path / "KILL").write_text("", encoding="utf-8")
    assert monitor.Monitor(kill_file=tmp_path / "KILL").gate({"symbol": "A"}) == "kill switch: KILL is present"


def test_stop_outs_spread_wider_than_the_window_do_not_trip_the_guard():
    m = monitor.Monitor(kill_file=pathlib.Path("no-such-dir") / "KILL")
    for d in ("2026-10-01", "2026-10-06", "2026-10-12"):                # first to last: 11 days > the 10-day window
        m.record_exit(d, "X", stopped=True)
    m.record_exit("2026-10-12", "Y", stopped=False)                    # a plain sale is not a stop-out
    assert m.gate({"symbol": "Z"}, now="2026-10-13T00:00:00Z") is None


# ----------------------------------------------------------------------------- track.py

def test_add_months_lands_on_a_short_month_s_last_day():
    assert track.add_months(dt.date(2026, 1, 31), 1) == dt.date(2026, 2, 28)
    assert track.add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert track.add_months(dt.date(2026, 11, 15), 14) == dt.date(2028, 1, 15)


class Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_closes_reads_yahoo_once_per_range_and_close_on_takes_the_last_close(monkeypatch):
    monkeypatch.setattr(track, "_cache", {})
    t0 = int(dt.datetime(2026, 1, 5, 21, tzinfo=dt.timezone.utc).timestamp())
    chart = {"chart": {"result": [{"timestamp": [t0, t0 + 86400], "indicators": {"quote": [{"close": [10.0, None]}]}}]}}
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return Resp(json.dumps(chart).encode())
    monkeypatch.setattr(track.urllib.request, "urlopen", urlopen)
    assert track.close_on("ABC", dt.date(2026, 1, 7)) == 10.0
    assert track.close_on("ABC", dt.date(2026, 1, 7)) == 10.0 and len(calls) == 1    # cached
    assert "chart/ABC?period1=" in calls[0]
    monkeypatch.setattr(track, "_cache", {})
    assert track.close_on("ABC", dt.date(2026, 1, 4)) is None                        # the only close is after the day

    def gone(req, timeout):
        raise OSError("404")
    monkeypatch.setattr(track, "_cache", {})
    monkeypatch.setattr(track.urllib.request, "urlopen", gone)
    assert track.close_on("GONE", dt.date(2026, 1, 7)) is None


ROW = {"date": "2025-01-02", "symbol": "ABC", "rating": "Overweight", "conviction": "High", "price": 10.0,
       "bull_value": 15.0, "bull_prob": 25, "base_value": 11.0, "base_prob": 50, "bear_value": 6.0, "bear_prob": 25}


def prices(table):
    return lambda symbol, day: table.get(symbol)


def test_score_buckets_outcomes_and_keeps_rows_without_prices_pending():
    today = dt.date(2026, 6, 1)
    done, pending = track.score([ROW, dict(ROW, symbol="NOP"), dict(ROW, horizon_months=24)], today,
                                prices({"ABC": 16.0, "SPY": 100.0}))
    assert len(done) == 1 and len(pending) == 2
    d = done[0]
    assert d["bucket"] == "bull" and d["right"] is True and d["brier"] == pytest.approx(0.5625 + 0.25 + 0.0625)
    base = track.score([dict(ROW, price_date="2025-01-02")], today, prices({"ABC": 11.0, "SPY": 100.0}))[0][0]
    assert base["bucket"] == "base" and base["right"] is True
    assert track.outcome_bucket(ROW, 6.0) == "bear"
    assert track.right("Underweight", -1) and track.right("Equal-weight", 9.0) and not track.right("Equal-weight", 11.0)
    no_scen = track.score([{k: v for k, v in ROW.items() if "_" not in k}], today, prices({"ABC": 12.0, "SPY": 100.0}))[0][0]
    assert "brier" not in no_scen


def test_report_covers_every_section():
    today = dt.date(2026, 6, 1)
    rows = [ROW, dict(ROW, rating="Underweight", conviction="Low"), dict(ROW, date="2026-05-01", symbol="LATE")]
    done, pending = track.score(rows, today, lambda s, d: {"ABC": 12.0, "SPY": 100.0 if d.year == 2025 else 105.0}.get(s))
    text = track.report(done, pending)
    assert "2 matured, 1 pending." in text and "Next to mature: LATE" in text and "beat it 1 of 1 times" in text
    assert "Brier" in text and "High: right 1 of 1" in text and "Low: right 0 of 1" in text and "too few" in text
    only_eq = track.report(*track.score([{k: v for k, v in dict(ROW, rating="Equal-weight").items() if "_" not in k}], today,
                                        prices({"ABC": 10.0, "SPY": 100.0})))
    assert "none matured yet." in only_eq and "no matured forecast has all three" in only_eq
    assert track.report([], []) == "0 matured, 0 pending."
    many = [dict(done[0])] * 30
    assert "too few" not in track.report(many, [])


def test_the_track_cli(tmp_path, monkeypatch, capsys):
    assert track.load(tmp_path / "none.jsonl") == []
    assert track.main([str(tmp_path / "none.jsonl")]) == 0 and "No forecasts logged yet" in capsys.readouterr().out
    log = tmp_path / "l.jsonl"
    log.write_text(json.dumps(dict(ROW, date="2099-01-01")) + "\n\n", encoding="utf-8")
    assert track.main([str(log)]) == 0 and "0 matured, 1 pending." in capsys.readouterr().out


def test_buy_below_is_checked_only_when_the_aftermarket_rating_is_buy_below():
    k = full_block(ipo_ratings={"at_offer": "Participate", "aftermarket": "Wait", "buy_below": 1})
    names = [n for n, _, _ in verify.check(k)]
    assert "Participate only with an Overweight rating" in names and "buy-below = PWV / 1.15" not in names
