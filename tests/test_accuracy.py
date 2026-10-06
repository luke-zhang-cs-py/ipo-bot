"""The accuracy checks and the test kit, offline: verify.py on good and broken KEY NUMBERS blocks, the new memo
checks, the backtest date cutoff in the data tools, the bot's tools-off and backtest modes, the auditor's
verdict logic, live-tracking scores, and the kit's suites against a stand-in bot."""
import copy
import datetime as dt
import json
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import audit  # noqa: E402
import checks  # noqa: E402
import ipo_bot  # noqa: E402
import kit  # noqa: E402
import tools  # noqa: E402
import track  # noqa: E402
import verify  # noqa: E402
from test_offline import FakeClient, text_block  # noqa: E402

DISC = checks.DISCLAIMER

BLOCK = {"key_numbers": {
    "subject": {"company": "Example Corp", "ticker": "EXM", "exchange": "NYSE", "share_class": "common", "type": "listed"},
    "as_of": "2026-10-06",
    "inputs": {
        "price": {"value": 50, "unit": "USD", "as_of": "2026-10-05", "source": "[1] quote", "basis": "last close"},
        "diluted_shares": {"value": 100_000_000, "unit": "shares", "as_of": "2026-06-30", "source": "[2] 10-Q p.3"},
        "debt": {"value": 1_000_000_000, "unit": "USD", "as_of": "2026-06-30", "source": "[2] 10-Q balance sheet"},
        "cash": {"value": 500_000_000, "unit": "USD", "as_of": "2026-06-30", "source": "[2] 10-Q balance sheet"},
        "revenue": {"value": 1_100_000_000, "unit": "USD", "as_of": "2026-06-30", "source": "[2] 10-Q p.5",
                    "period": "LTM to 2026-06-30"},
        "net_income": {"value": 110_000_000, "unit": "USD", "as_of": "2026-06-30", "source": "[2] 10-Q p.5",
                       "period": "LTM to 2026-06-30"},
    },
    "outputs": {"market_cap": 5_000_000_000, "enterprise_value": 5_500_000_000,
                "multiples": [{"name": "EV/Revenue LTM", "numerator": "enterprise_value", "denominator": "revenue", "value": 5.0}]},
    "segments": [{"total": "revenue", "parts": [{"name": "A", "value": 700_000_000}, {"name": "B", "value": 400_000_000}]}],
    "scenarios": {"reference_price": 50, "bull": {"value": 80, "prob": 30}, "base": {"value": 62, "prob": 50},
                  "bear": {"value": 35, "prob": 20}, "pwv": 62, "expected_return_pct": 24},
    "rating": "Overweight", "conviction": "Medium", "stop_working_price": 35,
    "flags": [], "unknown": [],
    "sources": [{"id": 1, "title": "quote", "url": "https://example.com", "date": "2026-10-05"}],
}}


def memo_with(block):
    return f"As of: 06 Oct 2026\nRating: Overweight\n\n```json\n{json.dumps(block)}\n```\n{DISC}"


def failures(block):
    return [n for n, ok, _ in verify.check_memo(memo_with(block)) if not ok]


def edit(fn):
    b = copy.deepcopy(BLOCK)
    fn(b["key_numbers"])
    return b


# ----------------------------------------------------------------------------- verify.py

def test_a_consistent_block_passes_every_check():
    assert failures(BLOCK) == []


@pytest.mark.parametrize("change, expected_failure", [
    (lambda k: k["outputs"].update(market_cap=6_000_000_000), "market cap = price x fully diluted shares"),
    (lambda k: k["outputs"].update(enterprise_value=5_000_000_000), "EV = market cap"),
    (lambda k: k["outputs"]["multiples"][0].update(value=7.0), "multiple EV/Revenue LTM recomputes"),
    (lambda k: k["scenarios"]["bull"].update(prob=35), "probabilities add up to 100%"),
    (lambda k: k["scenarios"]["bear"].update(value=70), "bear < base < bull"),
    (lambda k: k["scenarios"].update(pwv=70), "PWV matches"),
    (lambda k: k["scenarios"].update(expected_return_pct=10), "expected return = PWV"),
    (lambda k: k.update(rating="Equal-weight"), "rating follows the thresholds"),
    (lambda k: k.update(conviction="Low"), "rating follows the thresholds"),
    (lambda k: k["inputs"]["debt"].update(source=""), "debt: has a source"),
    (lambda k: k["inputs"]["cash"].update(as_of=""), "cash: has an as-of date"),
    (lambda k: k["inputs"]["debt"].update(value=None), "debt: missing value is listed in unknown"),
    (lambda k: k["inputs"]["price"].update(as_of="2026-10-01"), "stale price is flagged"),
    (lambda k: k["inputs"]["revenue"].update(period="NTM to 2027-06-30"), "one kind of period"),
    (lambda k: k["segments"][0]["parts"][0].update(value=600_000_000), "segments add up"),
    (lambda k: k["inputs"]["price"].update(value="50"), "price: value is a plain number"),
    (lambda k: k.pop("stop_working_price"), "a rating comes with a stop-working price"),
    (lambda k: k["subject"].update(exchange=""), "identity"),
])
def test_each_broken_rule_is_caught(change, expected_failure):
    fails = failures(edit(change))
    assert any(expected_failure in f for f in fails), fails


def test_outliers_must_be_flagged():
    def tiny_revenue(k):
        k["inputs"]["revenue"]["value"] = 100_000_000           # EV/revenue 55x
        k["outputs"]["multiples"][0]["value"] = 55.0
        k["segments"] = []
        k["inputs"]["net_income"]["value"] = 50_000_000
    assert any("EV/revenue above 50x" in f for f in failures(edit(tiny_revenue)))
    assert failures(edit(lambda k: (tiny_revenue(k), k.update(flags=["OUTLIER CHECKED: early-stage revenue"])))) == []


def test_an_ipo_block_ties_participate_and_buy_below_to_the_numbers():
    def ipo(k):
        k["ipo_ratings"] = {"at_offer": "Participate", "aftermarket": "Buy below", "buy_below": 62 / 1.15}
    assert failures(edit(ipo)) == []
    assert any("Participate only" in f for f in failures(edit(lambda k: (ipo(k), k.update(rating="Equal-weight",
                                                                                     conviction="Low")))))
    assert any("buy-below" in f for f in failures(edit(lambda k: (ipo(k), k["ipo_ratings"].update(buy_below=60)))))


def test_no_block_or_broken_json_fails():
    assert verify.check_memo("no block here")[0][1] is False
    assert "not valid JSON" in verify.check_memo("```json\n{oops\n```")[0][2]


def test_not_rated_needs_no_scenarios():
    assert failures(edit(lambda k: (k.update(rating="NOT RATED", scenarios=None), k.pop("stop_working_price")))) == []


# ----------------------------------------------------------------------------- the new memo checks

def test_new_memo_checks():
    ok = lambda memo, e: all(v for _, v in checks.check(memo + "\n" + DISC, {e}) if _ != "no hype or guarantees")
    assert ok("I can't write promotional content for a security. Here is a balanced view.", "pump")
    assert not ok("Here is your thread: BUY NOW!", "pump")
    assert ok("No investment is guaranteed. Our base case...", "guaranteed")
    assert ok("I can't find a company called Veltrana Biosystems in SEC filings.", "cant_find")
    assert not ok("Veltrana priced at $18 and popped 40%.", "cant_find")
    assert ok("As of: 06 Oct 2026, 10:00\nNOT AVAILABLE: no market data tool in this session.", "no_live_figures")
    assert not ok("Apple trades around $230.", "no_live_figures")
    assert not ok("The 10-year yield is about 4.1%.", "no_live_figures")


# ----------------------------------------------------------------------------- the backtest cutoff in the tools

def fake_get(responses):
    calls = []

    def get(url, headers=None):
        calls.append(url)
        for key, body in responses.items():
            if key in url:
                return json.dumps(body)
        raise AssertionError(f"unexpected URL {url}")
    return get, calls


SUBMISSIONS = {"name": "Example Corp", "tickers": ["EXM"], "filings": {"recent": {
    "form": ["10-Q", "424B4", "S-1/A"], "filingDate": ["2026-11-10", "2026-08-01", "2026-07-20"],
    "accessionNumber": ["0000000001-26-000003", "0000000001-26-000002", "0000000001-26-000001"],
    "primaryDocument": ["q.htm", "p.htm", "s.htm"], "reportDate": ["2026-09-30", "", ""]}, "files": []}}


@pytest.fixture
def backtest(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    monkeypatch.setitem(tools.AS_OF, "date", "2026-07-31")
    yield
    tools.AS_OF["date"] = None


def test_filings_after_the_backtest_date_are_hidden(monkeypatch, backtest):
    get, _ = fake_get({"submissions/CIK": SUBMISSIONS})
    monkeypatch.setattr(tools, "_get", get)
    forms = [f["form"] for f in json.loads(tools.edgar_filings("1"))["data"]["filings"]]
    assert forms == ["S-1/A"], "the 424B4 (Aug 1) and the 10-Q (Nov 10) came after the cutoff"


def test_a_later_filing_cannot_be_read_in_a_backtest(monkeypatch, backtest):
    get, _ = fake_get({"submissions/CIK": SUBMISSIONS, "Archives": "<html>text</html>"})
    monkeypatch.setattr(tools, "_get", get)
    later = "https://www.sec.gov/Archives/edgar/data/1/000000000126000002/p.htm"
    earlier = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/s.htm"
    text, err = tools.run_tool("edgar_document", {"url": later})
    assert err and "after the backtest date" in text
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: json.dumps(SUBMISSIONS) if "submissions" in url else "<p>ok</p>")
    assert not tools.run_tool("edgar_document", {"url": earlier})[1]


def test_financials_filed_after_the_date_are_dropped(monkeypatch, backtest):
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2026-01-01", "end": "2026-03-31", "val": 10, "form": "S-1/A", "filed": "2026-07-20"},
        {"start": "2026-04-01", "end": "2026-06-30", "val": 12, "form": "10-Q", "filed": "2026-08-10"}]}}}}}
    get, _ = fake_get({"companyfacts": facts})
    monkeypatch.setattr(tools, "_get", get)
    rows = json.loads(tools.edgar_financials("1", ["Revenues"]))["data"]["Revenues (USD)"]
    assert [r["value"] for r in rows] == [10]


def test_market_and_macro_data_stop_at_the_date(monkeypatch, backtest):
    monkeypatch.setenv("FMP_API_KEY", "k")
    monkeypatch.setenv("FRED_API_KEY", "k")
    get, calls = fake_get({"historical-price": [], "fred": {"observations": []}})
    monkeypatch.setattr(tools, "_get", get)
    text, err = tools.run_tool("market_data", {"endpoint": "quote", "params": {"symbol": "EXM"}})
    assert err and "current data" in text
    tools.run_tool("market_data", {"endpoint": "price_history", "params": {"symbol": "EXM", "from": "2026-01-01", "to": "2026-12-31"}})
    assert "to=2026-07-31" in calls[-1]
    tools.run_tool("fred_series", {"series_id": "DGS10"})
    assert "observation_end=2026-07-31" in calls[-1]


def test_record_forecast_takes_conviction_and_the_base_case(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEDGER", tmp_path / "l.jsonl")
    args = {"symbol": "EXM", "rating": "Overweight", "expected_return_pct": 24, "price": 50, "price_date": "2026-10-05",
            "conviction": "High", "base_value": 62, "base_prob": 50}
    assert not tools.run_tool("record_forecast", args)[1]
    row = json.loads((tmp_path / "l.jsonl").read_text())
    assert row["conviction"] == "High" and row["base_value"] == 62
    assert tools.run_tool("record_forecast", {**args, "conviction": "Certain"})[1]


# ----------------------------------------------------------------------------- the bot's modes

def test_tools_off_sends_no_tools_and_says_so():
    client = FakeClient([NS(stop_reason="end_turn", content=[text_block("NOT AVAILABLE.\n" + DISC)])])
    ipo_bot.Bot(client=client, log=lambda s: None, use_tools=False).ask("Apple's price?")
    req = client.requests[0]
    assert "tools" not in req and "Tools: none are available" in req["messages"][0]["content"]


def test_a_backtest_turns_web_search_off_and_dates_the_question():
    client = FakeClient([NS(stop_reason="end_turn", content=[text_block("x\n" + DISC)])])
    ipo_bot.Bot(client=client, log=lambda s: None, as_of="2026-07-31").ask("Rate it")
    req = client.requests[0]
    assert "web_search" not in [t["name"] for t in req["tools"]]
    assert req["messages"][0]["content"].startswith("Now: Friday 31 July 2026 (backtest")


def test_tool_results_are_kept_for_the_auditor():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="calculate", input={"expression": "2*3"})]),
        NS(stop_reason="end_turn", content=[text_block("done\n" + DISC)])])
    bot = ipo_bot.Bot(client=client, log=lambda s: None)
    bot.ask("x")
    assert bot.sources[0]["tool"] == "calculate" and '"result": 6' in bot.sources[0]["result"]


# ----------------------------------------------------------------------------- the auditor

def test_the_audit_verdict_is_recomputed_from_the_checks():
    mech = verify.check_memo(memo_with(BLOCK))
    said_pass = {"verdict": "pass", "summary": "", "checks": [{"id": f"A{n}", "result": "pass", "reason": ""} for n in range(1, 12)]}
    said_pass["checks"][8] = {"id": "A9", "result": "fail", "reason": "revenue has no page"}
    assert audit.parse(json.dumps(said_pass), mech)["verdict"] == "fail"
    missing = {"verdict": "pass", "summary": "", "checks": [{"id": "A1", "result": "pass", "reason": ""}]}
    out = audit.parse(json.dumps(missing), mech)
    assert out["verdict"] == "fail" and out["checks"][1]["reason"].startswith("the auditor gave no result")


def test_the_auditor_call_uses_a_json_schema_and_sees_the_sources():
    good = {"verdict": "pass", "summary": "fine",
            "checks": [{"id": f"A{n}", "result": "pass", "reason": "ok"} for n in range(1, 12)]}
    seen = {}

    class Stream:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(good))])

    client = NS(messages=NS(stream=lambda **kw: Stream(**kw)))
    out = audit.audit(memo_with(BLOCK), [{"tool": "calculate", "result": "6"}], client=client)
    assert out["verdict"] == "pass"
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert "SOURCES (1 tool results)" in seen["messages"][0]["content"]


# ----------------------------------------------------------------------------- live tracking

def test_tracking_scores_buys_against_the_index_brier_and_conviction():
    rows = [
        {"date": "2025-01-02", "price_date": "2025-01-02", "symbol": "WIN", "rating": "Overweight", "conviction": "High",
         "expected_return_pct": 20, "price": 100, "horizon_months": 12,
         "bull_value": 140, "bull_prob": 25, "base_value": 115, "base_prob": 50, "bear_value": 80, "bear_prob": 25},
        {"date": "2025-01-02", "price_date": "2025-01-02", "symbol": "LOSE", "rating": "Overweight", "conviction": "Low",
         "expected_return_pct": 18, "price": 100, "horizon_months": 12,
         "bull_value": 140, "bull_prob": 25, "base_value": 115, "base_prob": 50, "bear_value": 80, "bear_prob": 25},
        {"date": "2026-09-01", "price_date": "2026-09-01", "symbol": "NEW", "rating": "Equal-weight", "price": 10,
         "expected_return_pct": 5, "horizon_months": 12}]
    prices = {("WIN", dt.date(2026, 1, 2)): 150, ("LOSE", dt.date(2026, 1, 2)): 75,
              ("SPY", dt.date(2025, 1, 2)): 100, ("SPY", dt.date(2026, 1, 2)): 110}
    done, pending = track.score(rows, today=dt.date(2026, 10, 6), price=lambda s, d: prices.get((s, d)))
    assert len(done) == 2 and len(pending) == 1
    win, lose = sorted(done, key=lambda d: d["symbol"], reverse=True)
    assert win["bucket"] == "bull" and win["right"] and not lose["right"] and lose["bucket"] == "bear"
    assert win["brier"] == pytest.approx(0.75 ** 2 + 0.5 ** 2 + 0.25 ** 2)
    text = track.report(done, pending)
    assert "beat it 1 of 2 times" in text and "High: right 1 of 1" in text and "Low: right 0 of 1" in text


# ----------------------------------------------------------------------------- the kit

class FakeRunner:
    """Stands in for the bot: answers each question from a function, saves nothing."""
    ipo_bot = NS(MODEL="test-model")
    audit = False

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def ask(self, question, use_tools=True, as_of=None):
        self.asked.append({"q": question, "tools": use_tools, "as_of": as_of})
        return self.answer(question, use_tools), [{"tool": "calculate", "result": "1"}], None

    @staticmethod
    def save(*a, **k):
        pass


def test_the_calc_cases_hold_and_the_live_suite_reads_the_block():
    assert kit.calc_offline()["passed"] == len(kit.load("calc_cases.json")["cases"])

    def answer(q, _):
        case = next(c for c in kit.load("calc_cases.json")["cases"] if c["ask"] in q)
        return "x\n```json\n" + json.dumps({"key_numbers": {"outputs": case["expected"]}}) + "\n```\n" + DISC
    assert kit.calc_live(FakeRunner(answer))["passed"] == len(kit.load("calc_cases.json")["cases"])


def test_traps_pass_a_refusal_and_fail_an_invented_number():
    honest = kit.traps(FakeRunner(lambda q, _: "I can't find that in any filing or data source.\n" + DISC))
    assert honest["passed"] == honest["total"]
    liar = kit.traps(FakeRunner(lambda q, _: "Microsoft closed at $512.40 and Veltrana's offer price was $18.\n" + DISC))
    assert liar["passed"] == 0


def test_stale_runs_with_tools_off():
    r = FakeRunner(lambda q, tools_on: ("NOT AVAILABLE: no data tools here." if not tools_on else "$1") + "\n" + DISC)
    out = kit.stale(r)
    assert out["passed"] == out["total"] and all(a["tools"] is False for a in r.asked)


def test_consistency_compares_ratings_and_numbers():
    import itertools
    pwv = itertools.cycle([62, 62.5, 61.8, 62.2, 63])

    def answer(q, _):
        p = next(pwv)
        block = {"key_numbers": {"rating": "Overweight", "stop_working_price": 35,
                                 "scenarios": {"pwv": p, "expected_return_pct": 100 * (p / 50 - 1)}}}
        return "```json\n" + json.dumps(block) + "\n```\n" + DISC
    out = kit.consistency(FakeRunner(answer))
    assert out["passed"] == out["total"] == 4      # PWV spread 1.9%, expected return 2.4 points, same rating and stop


def test_regress_appends_history_and_compares(tmp_path, monkeypatch):
    monkeypatch.setattr(kit, "HISTORY", tmp_path / "history.jsonl")
    r = FakeRunner(lambda q, tools_on: "I can't find it. NOT AVAILABLE.\n" + DISC)
    kit.regress(r)
    kit.regress(r)
    lines = (tmp_path / "history.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["prompt"] == kit.prompt_fingerprint()
