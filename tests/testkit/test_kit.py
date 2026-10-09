"""testkit/kit.py end to end, offline: every suite runs the real Bot loop against a fake Anthropic client that
replays canned responses (text, tool_use, stop reasons). Nothing touches the network or the real testkit/runs."""
import datetime as dt
import io
import json
import pathlib
import shutil
import socket
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testkit"))

import audit  # noqa: E402
import checks  # noqa: E402
import ipo_bot  # noqa: E402
import kit  # noqa: E402
import tools  # noqa: E402
import track  # noqa: E402
import verify  # noqa: E402

DISC = checks.DISCLAIMER
REAL_KIT = kit.KIT


# ----------------------------------------------------------------------------- the fake client

def text(t):
    return NS(type="text", text=t)


def tool_use(i, name, inp):
    return NS(type="tool_use", id=i, name=name, input=inp)


def msg(*content, stop="end_turn"):
    return NS(stop_reason=stop, content=list(content))


class FakeStream:
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.m


class FakeClient:
    """bot(question, turn, request) answers the bot; auditor(request) answers audit.audit."""

    def __init__(self, bot, auditor=None):
        self.bot, self.auditor, self.requests, self.as_of_seen = bot, auditor, [], []
        self.beta = NS(messages=NS(stream=self._bot))
        self.messages = NS(stream=self._audit)

    def _bot(self, **kw):
        self.requests.append(kw)
        self.as_of_seen.append(tools.AS_OF["date"])
        first = kw["messages"][0]["content"]
        turn = sum(1 for m in kw["messages"] if m["role"] == "assistant")
        return FakeStream(self.bot(first, turn, kw))

    def _audit(self, **kw):
        return FakeStream(self.auditor(kw))


def block(key_numbers):
    return "Memo.\n```json\n" + json.dumps({"key_numbers": key_numbers}) + "\n```\n" + DISC


def say(t):
    """A bot that always answers with this text and no tools."""
    return lambda q, turn, kw: msg(text(t))


# ----------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No network, the kit's output in tmp_path, and the bot's globals restored afterwards."""
    def guard(*a, **k):
        raise AssertionError("a test tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(tools, "_get", guard)
    monkeypatch.setattr(track, "closes", guard)
    monkeypatch.setattr(ipo_bot, "client_or_exit", guard)
    monkeypatch.setattr(kit, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(kit, "HISTORY", tmp_path / "history.jsonl")
    monkeypatch.setattr(tools, "LEDGER", tools.LEDGER)
    monkeypatch.setattr(tools, "AS_OF", {"date": None})


@pytest.fixture
def kitdir(monkeypatch, tmp_path):
    """A copy of testkit's data files that a test may rewrite."""
    d = tmp_path / "kit"
    d.mkdir()
    for f in REAL_KIT.glob("*.json"):
        shutil.copy(f, d / f.name)
    monkeypatch.setattr(kit, "KIT", d)

    def write(name, doc):
        (d / name).write_text(json.dumps(doc), encoding="utf-8")
    return write


def runner(bot, auditor=None):
    return kit.Runner(audit=auditor is not None, client=FakeClient(bot, auditor))


def saved(tmp_path, suite, name):
    return (tmp_path / "runs" / kit.STAMP / suite / f"{name}.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- small pieces

def test_load_run_dir_and_fingerprint(tmp_path):
    assert kit.load("stale.json")["cases"][0]["id"] == "S1"
    d = kit.run_dir("x")
    assert d == tmp_path / "runs" / kit.STAMP / "x" and d.is_dir()
    assert len(kit.prompt_fingerprint()) == 12


def test_runner_points_the_ledger_into_the_run(tmp_path):
    r = runner(say("hi"))
    assert tools.LEDGER == tmp_path / "runs" / kit.STAMP / "ledger.jsonl"
    assert r.ask("q") == ("hi", [], None)


def test_runner_save_makes_a_ticker_with_a_slash_a_file_name(tmp_path):
    kit.Runner.save("bt", "BRK/B", "q", "memo", [{"tool": "x"}], [("ok", True), ("bad", False)])
    text_ = saved(tmp_path, "bt", "BRK_B")
    assert "- [x] ok" in text_ and "- [ ] bad" in text_ and "Audit" not in text_
    assert json.loads((tmp_path / "runs" / kit.STAMP / "bt" / "BRK_B.sources.json").read_text()) == [{"tool": "x"}]


def test_same_compares_numbers_and_labels():
    assert kit.same(1.0, 1.0 + 1e-12)
    assert not kit.same(1.0, 1.1)
    assert kit.same(1.0, 1.0005, rel=1e-3)
    assert kit.same("Overweight", "Overweight") and not kit.same("Overweight", "Underweight")
    assert not kit.same(None, 5) and not kit.same(True, 1) and not kit.same("5", 5)
    assert kit.same(0, 0)


def test_find_value_looks_everywhere_and_never_crashes():
    k = {"outputs": {"market_cap": 10, "nested": {"x": 1},
                     "multiples": ["junk", {"name": "Other", "value": 1}, {"name": "EV/Revenue (TTM)", "value": 5.0}]},
         "scenarios": {"pwv": 22.0, "bull": {"value": 30}}, "rating": "Overweight",
         "ipo_ratings": {"at_offer": "Participate"}}
    assert kit.find_value(k, "market_cap") == 10
    assert kit.find_value(k, "ev/revenue") == 5.0
    assert kit.find_value(k, "pwv") == 22.0
    assert kit.find_value(k, "rating") == "Overweight"
    assert kit.find_value(k, "at_offer") == "Participate"
    assert kit.find_value(k, "bull") is None and kit.find_value(k, "nested") is None
    assert kit.find_value(k, "missing") is None
    # wrong shapes from a model: a list for outputs, a string for multiples or scenarios
    assert kit.find_value({"outputs": [1], "scenarios": "x", "ipo_ratings": 3}, "pwv") is None
    assert kit.find_value({"outputs": {"multiples": "P/E 20"}}, "P/E") is None


def test_tally_counts_and_prints(capsys):
    out = kit.tally("s", [{"id": "a", "pass": True}, {"id": "b", "pass": False, "why": "broken"}, {"id": "c", "pass": False}])
    assert (out["passed"], out["total"]) == (1, 3)
    printed = capsys.readouterr().out
    assert "s: 1 of 3 passed" in printed and "FAIL  b  broken" in printed and "PASS  a" in printed


def test_with_audit_explains_a_failed_audit():
    assert kit.with_audit({"id": "a", "pass": True}, None) == {"id": "a", "pass": True}
    ok = kit.with_audit({"id": "a", "pass": True}, {"verdict": "pass", "checks": []})
    assert ok["pass"] and ok["audit"] == "pass"
    v = {"verdict": "fail", "checks": [{"id": "A1", "result": "fail", "reason": "no source"},
                                       {"id": "A2", "result": "pass", "reason": ""}], "mechanical": []}
    bad = kit.with_audit({"id": "a", "pass": True, "why": "earlier"}, v)
    assert not bad["pass"] and bad["why"] == "earlier; audit failed: A1 no source"
    mech_only = {"verdict": "fail", "checks": [], "mechanical": [{"check": "as_of date present", "pass": False},
                                                                 {"check": "x", "pass": True}]}
    assert kit.with_audit({"id": "a", "pass": True}, mech_only)["why"] == "audit failed: as_of date present"
    long = {"verdict": "fail", "checks": [{"id": "A1", "result": "fail", "reason": "r" * 1000}]}
    assert len(kit.with_audit({"id": "a", "pass": True}, long)["why"]) == kit.WHY_CHARS


# ----------------------------------------------------------------------------- calculations

def test_calc_offline_passes_the_real_cases_and_fails_a_wrong_one(kitdir):
    out = kit.calc_offline()
    assert out["passed"] == out["total"] == len(kit.load("calc_cases.json")["cases"])
    cases = kit.load("calc_cases.json")
    cases["cases"][0]["expected"]["market_cap"] = 1      # a wrong expected figure
    kitdir("calc_cases.json", cases)
    out = kit.calc_offline()
    assert out["passed"] == out["total"] - 1
    assert "market_cap" in out["items"][0]["why"] and not out["items"][0]["pass"]


def calc_bot(wrong=None, use_calc=True):
    """Answers each calc case with its expected outputs, after one real calculate tool call."""
    cases = kit.load("calc_cases.json")["cases"]

    def bot(q, turn, kw):
        if use_calc and turn == 0:
            return msg(tool_use("t1", "calculate", {"expression": "2 * 3"}), stop="tool_use")
        case = next(c for c in cases if c["ask"] in q)
        outputs = dict(case["expected"])
        if wrong and case["id"] == wrong:
            first = next(iter(outputs))
            outputs[first] = outputs[first] * 1.01 if kit._number(outputs[first]) else "Underweight"
        return msg(text(block({"outputs": outputs})))
    return bot


def test_calc_live_passes_right_numbers_from_the_calculator(tmp_path):
    r = runner(calc_bot())
    out = kit.calc_live(r)
    assert out["passed"] == out["total"] == 9
    assert "- [x] results match the expected outputs" in saved(tmp_path, "calc", "C1")
    tool_result = r.client.requests[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and json.loads(tool_result["content"])["result"] == 6


def test_calc_live_fails_a_wrong_number_a_skipped_calculator_and_a_missing_block():
    out = kit.calc_live(runner(calc_bot(wrong="C3")))
    c3 = next(i for i in out["items"] if i["id"] == "C3")
    assert out["passed"] == 8 and not c3["pass"] and "got vs expected" in c3["why"]
    c7 = kit.calc_live(runner(calc_bot(wrong="C7")))
    assert c7["passed"] == 8                              # a 1% miss on a PWV fails at 0.1% tolerance
    out = kit.calc_live(runner(calc_bot(use_calc=False)))
    assert out["passed"] == 0 and all(i["why"] == "calculate never called" for i in out["items"])
    out = kit.calc_live(runner(say("no block here\n" + DISC)))
    assert out["passed"] == 0 and "key_numbers" in out["items"][0]["why"]


def test_calc_live_accepts_a_scenario_case_without_conviction(kitdir):
    case = next(c for c in kit.load("calc_cases.json")["cases"] if "scenarios" in c)
    case = {**case, "id": "X"}
    del case["conviction"]
    case["expected"] = {"pwv": kit.compute(case)["pwv"]}
    kitdir("calc_cases.json", {"cases": [case]})
    out = kit.calc_live(runner(calc_bot()))
    assert out["passed"] == 1


# ----------------------------------------------------------------------------- golden set

GOLDEN_KN = {"subject": {"company": "Example", "ticker": "BRK-B", "exchange": "NYSE", "share_class": "B"},
             "as_of": "2026-10-01",
             "inputs": {"price": {"value": 100.0, "source": "10-Q 0001", "as_of": "2026-09-30"},
                        "revenue": {"value": 1000, "source": "10-K", "as_of": "2026-09-30"}}}


def golden_doc():
    return {"entries": [
        {"id": "G0", "type": "listed", "question": "nothing filled", "subject": {"ticker": "X"}, "as_of": "",
         "expected": {"price": None}},
        {"id": "G1", "type": "listed", "question": "Rate BRK.B", "subject": {"ticker": "BRK.B"}, "as_of": "",
         "expected": {"price": 100.5, "revenue": 1000, "flag": True}, "expected_source": {"price": "10-q"}},
        {"id": "G2", "type": "ipo", "question": "Rate the IPO", "subject": {"ticker": "BRK.B"}, "as_of": "2026-07-01",
         "tolerance_pct": 0.1, "expected": {"price": 100.5, "revenue": 1000}, "expected_source": {"revenue": "S-1"}},
        {"id": "G3", "type": "listed", "question": "Rate broken", "subject": {"ticker": "Z"}, "as_of": "",
         "expected": {"price": 1}},
        {"id": "G4", "type": "listed", "question": "Rate odd shapes", "subject": {"ticker": "Z"}, "as_of": "",
         "expected": {"price": 1}},
    ]}


def golden_bot(q, turn, kw):
    if "broken" in q:
        return msg(text("no block\n" + DISC))
    if "odd shapes" in q:
        return msg(text(block({"inputs": ["price", 1], "subject": {"ticker": 5}})))
    return msg(text(block(GOLDEN_KN)))


def test_golden_scores_extraction_maths_citation_and_identity(kitdir, tmp_path, capsys):
    kitdir("golden_set.json", golden_doc())
    r = runner(golden_bot)
    out = kit.golden(r)
    items = {i["id"]: i for i in out["items"]}
    assert set(items) == {"G1", "G2", "G3", "G4"}           # G0 has no figures; True is not a figure
    g1, g2 = items["G1"], items["G2"]
    assert g1["parts"]["extraction"] == 1.0 and g1["parts"]["citation"] == 1.0 and g1["parts"]["identity"] == 1.0
    assert g2["parts"]["extraction"] == 0.5 and g2["parts"]["citation"] == 0.0 and "price: got 100.0" in g2["why"]
    assert items["G3"]["score"] == 0.0 and not items["G3"]["pass"]
    assert items["G4"]["parts"]["extraction"] == 0.0 and items["G4"]["parts"]["identity"] == 0.0
    # only the IPO runs as of its date
    assert r.client.as_of_seen == [None, "2026-07-01", None, None]
    assert "tools" in r.client.requests[1] and all(t.get("name") != "web_search" for t in r.client.requests[1]["tools"])
    assert out["score"] == round(sum(i["score"] for i in out["items"]) / 4, 3)
    assert "average score" in capsys.readouterr().out
    assert "KEY NUMBERS block" in saved(tmp_path, "golden", "G3")


def test_golden_with_a_perfect_answer_passes(kitdir):
    kitdir("golden_set.json", {"entries": [golden_doc()["entries"][1]]})
    good = {**GOLDEN_KN, "inputs": {"price": {"value": 100.5, "source": "10-Q", "as_of": "2026-09-30"},
                                    "revenue": {"value": 1000, "source": "10-K", "as_of": "2026-09-30"}}}
    out = kit.golden(runner(say(block(good))))
    item = out["items"][0]
    assert item["parts"]["extraction"] == 1.0 and item["parts"]["identity"] == 1.0
    assert item["pass"] == (item["parts"]["math"] >= kit.FULL_MARKS)


def test_golden_with_nothing_filled_in(kitdir, capsys):
    kitdir("golden_set.json", {"entries": [golden_doc()["entries"][0]]})
    out = kit.golden(runner(say("unused")))
    assert out == {"suite": "golden", "passed": 0, "total": 0, "items": [], "score": None}
    assert "no entries filled in yet" in capsys.readouterr().out


# ----------------------------------------------------------------------------- traps, stale, rules

def test_traps_pass_only_an_answer_that_says_unknown(tmp_path):
    honest = kit.traps(runner(say("I can't find any record of that company.\n" + DISC)))
    assert honest["passed"] == honest["total"] == 6
    liar = kit.traps(runner(say("Veltrana's offer price was $18 and it popped 40%.\n" + DISC)))
    assert liar["passed"] == 0
    rated = kit.traps(runner(say("I can't find it.\n" + block({"rating": "Overweight"}))))
    assert rated["passed"] == 0 and "NOT RATED" in rated["items"][0]["why"]
    unrated = kit.traps(runner(say("I can't find it.\n" + block({"rating": "NOT RATED"}))))
    assert unrated["passed"] == 6
    assert unrated["items"][0]["id"] == "T1 fake company"


def test_traps_fill_in_the_future_date():
    r = runner(say("I can't find it.\n" + DISC))
    kit.traps(r)
    future = (dt.date.today() + dt.timedelta(days=kit.TRAP_FUTURE_DAYS)).strftime("%d %B %Y")
    asked = [req["messages"][0]["content"] for req in r.client.requests]
    assert not any("{future_date}" in a for a in asked)
    if any("{future_date}" in c["question"] for c in kit.load("traps.json")["cases"]):
        assert any(future in a for a in asked)


def test_stale_runs_with_no_tools_and_fails_a_live_price():
    r = runner(say("I have no data tools here, so I can't give a current figure.\n" + DISC))
    out = kit.stale(r)
    assert out["passed"] == out["total"] == 3
    assert all("tools" not in req for req in r.client.requests)
    assert all("Tools: none are available" in req["messages"][0]["content"] for req in r.client.requests)
    bad = kit.stale(runner(say("Apple trades at $231.10.\n" + DISC)))
    assert bad["passed"] == 0 and bad["items"][0]["id"] == "S1"


def test_rules_apply_each_case_s_checks():
    answer = (f"{checks.MNPI_LINE}. {checks.PUMP_LINE}.\n" + DISC)
    out = kit.rules(runner(say(answer)))
    items = {i["id"].split()[0]: i for i in out["items"]}
    assert all(items[r]["pass"] for r in ("R1", "R2", "R4", "R5")) and out["total"] == 5
    assert not items["R3"]["pass"] and "says no investment is guaranteed" in items["R3"]["why"]
    # with the line R3 requires, R3 passes: "No investment is guaranteed" is not read as hype (src/checks.py)
    r3 = {i["id"].split()[0]: i for i in kit.rules(runner(say(f"{checks.GUARANTEE_LINE}.\n" + DISC)))["items"]}
    assert r3["R3"]["pass"]
    out = kit.rules(runner(say("This will definitely double. You should invest $5,000.\n" + DISC)))
    assert out["passed"] == 0 and out["items"][1]["id"] == "R2 inside information"


def test_a_failed_audit_fails_the_item_and_is_saved(tmp_path):
    def auditor(kw):
        assert kw["model"] == ipo_bot.MODEL
        checks_ = [{"id": r, "result": "pass", "reason": ""} for r in audit.RULES]
        checks_[0] = {"id": "A1", "result": "fail", "reason": "invented"}
        return msg(text(json.dumps({"checks": checks_, "summary": "one fail"})))
    out = kit.stale(runner(say("I can't give a figure without tools.\n" + DISC), auditor))
    assert out["passed"] == 0 and "audit failed: A1 invented" in out["items"][0]["why"]
    assert out["items"][0]["audit"] == "fail"
    assert "Audit:\nVerdict: FAIL" in saved(tmp_path, "stale", "S1")


# ----------------------------------------------------------------------------- consistency

def kn(rating="Overweight", pwv=62.0, er=24.0, stop=35.0):
    return block({"rating": rating, "scenarios": {"pwv": pwv, "expected_return_pct": er}, "stop_working_price": stop})


def cycle_bot(answers):
    it = iter(answers)
    return lambda q, turn, kw: msg(text(next(it)))


def test_consistency_passes_steady_answers_and_fails_drift(capsys):
    out = kit.consistency(runner(cycle_bot([kn(pwv=62 + i * 0.2, er=24 + i * 0.3) for i in range(5)])))
    assert out["passed"] == out["total"] == 4
    assert "runs: Overweight PWV 62" in capsys.readouterr().out
    drift = [kn(), kn(rating="Equal-weight"), kn(pwv=80), kn(er=40), kn(stop=60)]
    out = kit.consistency(runner(cycle_bot(drift)))
    assert out["passed"] == 0


def test_consistency_counts_a_missing_block_and_odd_shapes(tmp_path):
    answers = [kn(), "no block\n" + DISC, block({"rating": "Overweight", "scenarios": "n/a"}), kn(), kn()]
    out = kit.consistency(runner(cycle_bot(answers)))
    items = {i["id"]: i for i in out["items"]}
    assert not items["same rating every time"]["pass"]
    assert not items["expected return within 3 points"]["pass"]
    assert "- [ ] KEY NUMBERS block" in saved(tmp_path, "consistency", "run2")


def test_consistency_with_no_questions_does_not_crash(kitdir):
    kitdir("consistency.json", {"question": "q", "repeats": 0, "rewordings": [],
                                "tolerance": {"pwv_pct": 5, "expected_return_pts": 3, "stop_working_pct": 5}})
    out = kit.consistency(runner(say("unused")))
    assert out["passed"] == 0 and out["total"] == 4


def test_spread_pct():
    assert kit.spread_pct([1]) is None
    assert kit.spread_pct([True, False, 5]) is None
    assert kit.spread_pct([-1, 1]) is None                 # a zero mean has no percentage spread
    assert kit.spread_pct([90, 110, "x"]) == pytest.approx(20.0)


# ----------------------------------------------------------------------------- finding IPOs

def efts_hit(name, cik, date):
    return {"_source": {"display_names": [name] if name is not None else [], "ciks": [cik] if cik else [],
                        "file_date": date}}


def fake_sec(pages, subs, calls):
    def get(url, headers=None):
        calls.append(url)
        if "efts.sec.gov" in url:
            start = int(url.rsplit("&from=", 1)[1])
            return json.dumps({"hits": {"hits": pages.get(start, [])}})
        cik = url.rsplit("CIK", 1)[1].split(".")[0]
        return json.dumps(subs[cik])
    return get


def test_find_ipos_keeps_first_time_ipos_only(monkeypatch):
    filler = [efts_hit("Alpha Acquisition Corp (AAC) (CIK 0000000009)", "0000000009", "2026-07-01")] * 99
    page0 = filler + [efts_hit("Good Co (GOOD, GOODW) (CIK 0000000001)", "0000000001", "2026-07-10")]
    page1 = [
        efts_hit("Good Co (GOOD, GOODW) (CIK 0000000001)", "0000000001", "2026-07-05"),   # earlier: kept
        efts_hit("Good Co (GOOD, GOODW) (CIK 0000000001)", "0000000001", "2026-07-20"),   # later: ignored
        efts_hit("No Ticker Inc (CIK 0000000002)", "0000000002", "2026-07-03"),
        efts_hit("Blank Check Inc (BCI) (CIK 0000000003)", "0000000003", "2026-07-04"),
        efts_hit("Old Co (OLD) (CIK 0000000004)", "0000000004", "2026-07-06"),
        efts_hit("Big Co (BIG) (CIK 0000000005)", "0000000005", "2026-07-07"),
        efts_hit("Nobody", None, "2026-07-08"),
        efts_hit(None, "0000000006", "2026-07-09"),
        efts_hit("Hudson Capital Corp. IV (HCC) (CIK 0000000007)", "0000000007", "2026-07-09"),
    ]
    subs = {"0000000001": {"sic": "7372", "filings": {"recent": {"form": ["S-1", "10-Q"],
                                                                 "filingDate": ["2026-06-01", "2026-08-01"]}}},
            "0000000002": {"sic": 2834},
            "0000000003": {"sic": "6770"},
            "0000000004": {"filings": {"recent": {"form": ["10-K"], "filingDate": ["2025-03-01"]}}},
            "0000000005": {"filings": {"recent": {}, "files": [{"name": "older.json"}]}},
            "0000000006": {"filings": {}}}
    calls = []
    monkeypatch.setattr(tools, "_get", fake_sec({0: page0, 100: page1}, subs, calls))
    monkeypatch.setattr(tools, "_sec_headers", lambda: {"User-Agent": "test"})
    out = kit.find_ipos("2026-07-01", "2026-09-30")
    assert [c for c in calls if "efts" in c][1].endswith("&from=100")
    assert [(e["ticker"], e["company"], e["cik"], e["prospectus_date"], e["as_of"]) for e in out] == [
        ("", "No Ticker Inc", "2", "2026-07-03", "2026-07-02"),
        ("GOOD", "Good Co", "1", "2026-07-05", "2026-07-04"),
        ("", "", "6", "2026-07-09", "2026-07-08"),
    ]
    assert all(e["verified"] is False and e["offer_price"] is None for e in out)


def test_find_ipos_cmd_adds_only_new_companies(monkeypatch, kitdir, capsys):
    found = [{"company": "Good Co", "ticker": "GOOD", "cik": "1", "prospectus_date": "2026-07-05", "as_of": "2026-07-04"},
             {"company": "Mystery", "ticker": "", "cik": "2", "prospectus_date": "2026-07-06", "as_of": "2026-07-05"}]
    monkeypatch.setattr(kit, "find_ipos", lambda s, e: found)
    kitdir("backtest_ipos.json", {"entries": [{"cik": "1", "ticker": "GOOD"}]})
    kit.find_ipos_cmd(["2026-07-01", "2026-07-31"])
    doc = json.loads((kit.KIT / "backtest_ipos.json").read_text())
    assert [e["cik"] for e in doc["entries"]] == ["1", "2"] and doc["_about"] == kit.BACKTEST_ABOUT
    printed = capsys.readouterr().out
    assert "2 first-time IPOs" in printed and "1 added" in printed and "?      Mystery" in printed
    (kit.KIT / "backtest_ipos.json").unlink()
    kit.find_ipos_cmd(["2026-07-01", "2026-07-31"])
    assert len(json.loads((kit.KIT / "backtest_ipos.json").read_text())["entries"]) == 2


# ----------------------------------------------------------------------------- the backtest

def test_backtest_needs_its_file(kitdir):
    (kit.KIT / "backtest_ipos.json").unlink()
    with pytest.raises(SystemExit, match="find-ipos"):
        kit.backtest(runner(say("x")))


def ipo_kn(at_offer="Participate"):
    return block({"rating": "NOT RATED", "ipo_ratings": {"at_offer": at_offer, "buy_below": 18.0}})


def test_backtest_runs_as_of_the_day_before_and_refuses_later_data(kitdir, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    today = dt.date.today()
    listed = today - dt.timedelta(days=200)          # 1 and 6 months have passed, 12 have not
    kitdir("backtest_ipos.json", {"entries": [
        {"company": "Old IPO", "ticker": "OLD", "cik": "9", "as_of": kit.MODEL_CUTOFF},
        {"company": "Win Co", "ticker": "WIN", "cik": "1", "as_of": "2026-07-01", "verified": True,
         "offer_price": 20.0, "listing_date": listed.isoformat()},
        {"company": "Lose Co", "ticker": "LOSE", "cik": "2", "as_of": "2026-07-02", "verified": True,
         "offer_price": 10.0, "listing_date": listed.isoformat()},
        {"company": "Unpriced", "ticker": "", "cik": "3", "as_of": "2026-07-03", "verified": False},
        {"company": "No Data Co", "ticker": "NODA", "cik": "4", "as_of": "2026-07-04", "verified": True,
         "offer_price": 10.0, "listing_date": listed.isoformat()},
    ]})
    m1, m6 = track.add_months(listed, 1), track.add_months(listed, 6)
    prices = {("WIN", listed): 25.0, ("WIN", m1): 30.0, ("WIN", m6): 24.0,
              ("LOSE", m1): 8.0, ("LOSE", m6): 5.0,
              ("SPY", listed - dt.timedelta(days=1)): 500.0, ("SPY", m1): 510.0, ("SPY", m6): 550.0}
    asked = []
    monkeypatch.setattr(track, "close_on", lambda s, d: (asked.append(s), prices.get((s, d)))[1])

    def bot(q, turn, kw):
        if turn == 0:      # the bot reaches for today's quote: a backtest must refuse it
            return msg(tool_use("q1", "market_data", {"endpoint": "quote", "params": {"symbol": "WIN"}}), stop="tool_use")
        if "Unpriced" in q:
            return msg(text("no block\n" + DISC))
        return msg(text(ipo_kn("Pass" if "Lose" in q or "No Data" in q else "Participate")))
    r = runner(bot)
    out = kit.backtest(r)
    printed = capsys.readouterr().out
    assert "skip OLD" in printed and "OLD" not in [row["ticker"] for row in out["rows"]]
    assert r.client.as_of_seen == ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02", "2026-07-03", "2026-07-03",
                                   "2026-07-04", "2026-07-04"]
    assert tools.AS_OF["date"] is None
    assert all(t.get("name") != "web_search" for req in r.client.requests for t in req["tools"])
    refused = r.client.requests[1]["messages"][2]["content"][0]
    assert refused["is_error"] and "not available in a backtest dated 2026-07-01" in refused["content"]
    rows = {row["ticker"] or "3": row for row in out["rows"]}
    win, lose, nodata = rows["WIN"], rows["LOSE"], rows["NODA"]
    assert (win["1m"], win["6m"], win["12m"], win["first_close"]) == (50.0, 20.0, None, 25.0)
    assert (win["1m_spy"], win["6m_spy"]) == (2.0, 10.0) and "12m_spy" not in win
    assert (lose["1m"], lose["6m"]) == (-20.0, -50.0)
    assert nodata["1m"] is None and nodata["1m_spy"] == 2.0 and nodata["first_close"] is None
    assert "1m" not in rows["3"]
    assert (out["passed"], out["total"]) == (3, 4)
    assert next(i for i in out["items"] if i["id"] == "3")["why"] == "no IPO ratings"
    assert "| WIN | 2026-07-01 | Participate | 18.0 | +50.0% | +20.0% | pending | +2.0% / +10.0% / - |" in printed
    assert "Participate calls, 1 months after listing: +50.0% from the offer price (n = 1)" in printed
    assert "Pass calls, 6 months after listing: -50.0% from the offer price (n = 1)" in printed
    assert "- [ ] KEY NUMBERS block" in saved(tmp_path, "backtest", "3")
    assert "- [x] both IPO ratings in KEY NUMBERS" in saved(tmp_path, "backtest", "WIN")


def test_backtest_allow_hindsight_runs_old_ipos(kitdir):
    kitdir("backtest_ipos.json", {"entries": [{"company": "Old IPO", "ticker": "OLD", "cik": "9",
                                               "as_of": kit.MODEL_CUTOFF}]})
    out = kit.backtest(runner(say(ipo_kn())), allow_hindsight=True)
    assert out["passed"] == out["total"] == 1


def test_after_listing_without_a_benchmark_price(monkeypatch):
    listed = dt.date.today() - dt.timedelta(days=40)
    monkeypatch.setattr(track, "close_on", lambda s, d: 12.0 if s == "X" else None)
    row = kit.after_listing({"ticker": "X", "listing_date": listed.isoformat(), "offer_price": 10.0}, dt.date.today())
    assert row["1m"] == 20.0 and row["1m_spy"] is None and row["6m"] is None and row["first_close"] == 12.0


# ----------------------------------------------------------------------------- regression

def test_regress_records_history_and_compares_with_the_last_good_line(kitdir, tmp_path, capsys):
    kitdir("golden_set.json", {"entries": [golden_doc()["entries"][1]]})
    kitdir("consistency.json", {"question": "q", "repeats": 2, "rewordings": [],
                                "tolerance": {"pwv_pct": 5, "expected_return_pts": 3, "stop_working_pct": 5}})
    r = runner(say("I can't find it.\n" + DISC))
    first = kit.regress(r)
    assert first["total"] == sum(s["total"] for s in first["suites"].values())
    assert first["suites"]["calc (formulas)"] == {"passed": 9, "total": 9}
    assert first["suites"]["traps"] == {"passed": 6, "total": 6}
    assert first["suites"]["golden"]["score"] is not None
    assert "Last run" not in capsys.readouterr().out
    # a damaged last line, and an older entry whose suites differ
    old = {"date": "then", "prompt": "abc", "passed": 1, "total": 2, "suites": {"traps": {"passed": 0, "total": 6}}}
    kit.HISTORY.write_text(json.dumps(old) + "\n[1]\n{cut off", encoding="utf-8")
    second = kit.regress(r, with_consistency=True)
    printed = capsys.readouterr().out
    assert "Last run then (prompt abc): 1 of 2." in printed and "traps: 0/6 -> 6/6" in printed
    assert "consistency" in second["suites"] and "calc (formulas):" not in printed.split("Last run")[1]
    assert json.loads(kit.HISTORY.read_text().splitlines()[-1])["passed"] == second["passed"]


def test_regress_skips_an_empty_golden_set_and_reads_an_unreadable_history_as_none(kitdir, capsys):
    kitdir("golden_set.json", {"entries": []})
    kit.HISTORY.write_text("not json\n", encoding="utf-8")
    out = kit.regress(runner(say("I can't find it.\n" + DISC)))
    assert "golden" not in out["suites"] and "Last run" not in capsys.readouterr().out
    assert kit.last_history()["passed"] == out["passed"]


# ----------------------------------------------------------------------------- the command line

@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(ipo_bot, "load_env", lambda *a: None)
    client = FakeClient(lambda q, turn, kw: msg(text("I can't find it. I have no data tools.\n" + DISC)),
                        lambda kw: msg(text(json.dumps({"checks": [{"id": r, "result": "pass", "reason": ""}
                                                                   for r in audit.RULES]}))))
    monkeypatch.setattr(ipo_bot, "client_or_exit", lambda: client)
    return client


def test_main_offline_commands(cli, monkeypatch, kitdir):
    with pytest.raises(SystemExit) as e:
        kit.main([])
    assert "test kit" in str(e.value.code)
    assert kit.main(["calc"]) == 0
    cases = kit.load("calc_cases.json")
    cases["cases"][0]["expected"]["market_cap"] = 1
    kitdir("calc_cases.json", cases)
    assert kit.main(["calc"]) == 1
    monkeypatch.setattr(verify, "main", lambda a: ("verify", a))
    assert kit.main(["verify", "m.md"]) == ("verify", ["m.md"])
    monkeypatch.setattr(track, "main", lambda a: ("track", a))
    assert kit.main(["track", "--x"]) == ("track", ["--x"])
    got = []
    monkeypatch.setattr(kit, "find_ipos_cmd", got.append)
    assert kit.main(["find-ipos", "2026-07-01", "2026-07-31"]) == 0 and got == [["2026-07-01", "2026-07-31"]]
    with pytest.raises(SystemExit, match="needs a start and an end"):
        kit.main(["find-ipos", "2026-07-01"])
    monkeypatch.setattr(audit, "main", lambda a: ("audit", a))
    assert kit.main(["audit", "m.md", "--yes"]) == ("audit", ["m.md"])
    with pytest.raises(SystemExit, match="unknown command 'nope'"):
        kit.main(["nope"])
    assert cli.requests == []


def test_main_live_commands_need_yes(cli):
    for argv in (["traps"], ["calc", "--live"], ["audit", "m.md"]):
        with pytest.raises(SystemExit, match="costs API credit"):
            kit.main(argv)
    assert cli.requests == []


def test_main_runs_live_suites_through_the_client(cli, kitdir, tmp_path, capsys):
    assert kit.main(["stale", "--yes"]) == 0
    assert "Answers saved under" in capsys.readouterr().out
    # an honest "can't find it" owes no KEY NUMBERS block, so the audited traps pass (src/audit.py)
    assert kit.main(["traps", "--yes", "--audit"]) == 0
    assert "Audit:\nVerdict: PASS" in saved(tmp_path, "traps", "T1")
    assert kit.main(["rules", "--yes"]) == 1
    kitdir("backtest_ipos.json", {"entries": [{"company": "Old", "ticker": "OLD", "cik": "9", "as_of": "2026-01-01"}]})
    assert kit.main(["backtest", "--yes"]) == 0                       # skipped: nothing to fail
    assert kit.main(["backtest", "--yes", "--allow-hindsight"]) == 1  # run: no IPO ratings
    kitdir("golden_set.json", {"entries": []})
    assert kit.main(["regress", "--yes"]) == 1
    assert "Answers saved under" in capsys.readouterr().out


def test_main_explains_a_rejected_key(cli, monkeypatch):
    import anthropic
    import types
    request = types.SimpleNamespace(method="POST", url="http://x")  # no HTTP library: the SDK switched libraries
    err = anthropic.AuthenticationError("bad key", response=types.SimpleNamespace(status_code=401, headers={}, request=request),
                                        body=None)

    def reject(*a, **k):
        raise err
    monkeypatch.setattr(kit, "stale", reject)
    with pytest.raises(SystemExit, match="rejected the API key"):
        kit.main(["stale", "--yes"])


def test_main_survives_a_stream_without_reconfigure(cli, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert kit.main(["calc"]) == 0
    assert "calc (formulas)" in sys.stdout.getvalue()
