"""Offline tests: the calculator, tool dispatch and validation, the checks, and the bot's loop against a
stand-in client. No network and no API key needed.

    python -m pytest -q tests
"""
import json
import pathlib
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import ipo_bot  # noqa: E402
import tools  # noqa: E402


# --------------------------------------------------------------------------- calculator

def calc(expr, **v):
    text, err = tools.run_tool("calculate", {"expression": expr, "variables": v})
    return text if err else json.loads(text)["result"]


def test_calculator_does_valuation_arithmetic():
    assert calc("price * shares", price=21.5, shares=410_000_000) == 8_815_000_000.0
    assert calc("cap + debt - (cash + proceeds)", cap=100, debt=20, cash=15, proceeds=30) == 75
    assert round(calc("0.25*30 + 0.5*22 + 0.25*12"), 6) == 21.5
    assert calc("max(1, 2, 3) - min(4, 5) + abs(-2) + round(2.5, 0)") == 3.0


def test_calculator_refuses_anything_but_arithmetic():
    for bad in ("__import__('os').system('x')", "open('f')", "[1,2]", "a.b", "lambda: 1", "2 ** 1000"):
        text, err = tools.run_tool("calculate", {"expression": bad})
        assert err and text.startswith("Error"), bad


def test_calculator_reports_bad_inputs_as_errors():
    for args in ({"expression": "1/0"}, {"expression": "x + 1"}, {"expression": "1 +"},
                 {"expression": "x", "variables": {"x": "5"}}, {"expression": "x", "variables": {"x": True}}):
        text, err = tools.run_tool("calculate", args)
        assert err, args


# --------------------------------------------------------------------------- dispatch, validation, configuration

def test_tool_inputs_are_validated_before_running():
    assert tools.run_tool("calculate", {})[1]
    assert tools.run_tool("calculate", {"expression": "1", "shell": "x"})[1]
    assert tools.run_tool("no_such_tool", {})[1]
    assert tools.run_tool("calculate", "not a dict")[1]


def test_unconfigured_sources_say_so(monkeypatch):
    for var in ("SEC_USER_AGENT", "FRED_API_KEY", "FMP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for name, args, word in (("edgar_lookup", {"query": "ABC"}, "SEC_USER_AGENT"),
                             ("fred_series", {"series_id": "DGS10"}, "FRED_API_KEY"),
                             ("market_data", {"endpoint": "quote", "params": {"symbol": "ABC"}}, "FMP_API_KEY")):
        text, err = tools.run_tool(name, args)
        assert err and word in text, name


def test_edgar_document_reads_sec_gov_only(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    for url in ("https://evil.example/x.htm", "http://www.sec.gov/x.htm", "file:///etc/passwd"):
        text, err = tools.run_tool("edgar_document", {"url": url})
        assert err and "sec.gov" in text, url


def test_market_data_only_named_endpoints(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    text, err = tools.run_tool("market_data", {"endpoint": "../admin"})
    assert err and "unknown endpoint" in text


def test_keys_never_appear_in_results(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: json.dumps({"observations": [{"date": "2026-10-01", "value": "4.1"}]}))
    text, err = tools.run_tool("fred_series", {"series_id": "DGS10"})
    assert not err and "SECRETKEY123" not in text and "retrieved_at" in text


def test_edgar_filings_parses_submissions(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    sub = {"name": "Example Corp", "tickers": ["EXM"], "filings": {"recent": {
        "form": ["10-Q", "S-1", "424B4"], "filingDate": ["2026-09-01", "2026-06-01", "2026-07-01"],
        "reportDate": ["2026-06-30", "", ""], "accessionNumber": ["0001-26-1", "0001-26-2", "0001-26-3"],
        "primaryDocument": ["q.htm", "s1.htm", "p.htm"]}}}
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: json.dumps(sub))
    text, err = tools.run_tool("edgar_filings", {"cik": "123", "forms": ["S-1", "424B4"]})
    data = json.loads(text)["data"]
    assert not err and [f["form"] for f in data["filings"]] == ["S-1", "424B4"]
    assert data["filings"][0]["url"].startswith("https://www.sec.gov/Archives/edgar/data/123/")


def test_tool_definitions_are_valid_and_match_the_handlers():
    assert {t["name"] for t in tools.TOOL_DEFS} == set(tools.HANDLERS)
    for t in tools.TOOL_DEFS:
        s = t["input_schema"]
        assert s["type"] == "object" and s["additionalProperties"] is False and set(s["required"]) <= set(s["properties"])


def test_the_prompt_names_every_tool_the_bot_has():
    prompt = (pathlib.Path(ipo_bot.HERE) / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    for t in tools.TOOL_DEFS + [ipo_bot.WEB_SEARCH]:
        assert t["name"] in prompt, t["name"]


# --------------------------------------------------------------------------- checks

GOOD = """As of: Tuesday 06 October 2026, 10:00 EDT
Rating: Equal-weight, 12 months, conviction Medium | Stop-working price: $X (bear-case value)
| Case | Value | Probability |
| BULL | $A | 25% |
| BASE | $B | 50% |
| BEAR | $C | 25% |
For information only, not investment advice; do your own research or consult a licensed professional."""


def test_checks_pass_a_complete_memo():
    assert all(ok for _, ok in checks.check(GOOD))


def test_checks_catch_the_usual_failures():
    bad_probs = GOOD.replace("| 25% |\n| BASE", "| 30% |\n| BASE")
    assert not checks.probabilities_add_up(bad_probs)
    assert not checks.last_line_is_disclaimer(GOOD + "\nThanks!")
    assert not checks.no_hype("This IPO will double and returns are guaranteed.")
    assert not dict(checks.check(GOOD.replace("Stop-working price", "Exit")))["a rating comes with a stop-working price"]
    assert not dict(checks.check(GOOD, {"ipo"}))["an IPO gets both ratings (offer, aftermarket)"]
    assert not dict(checks.check("You should put $500 in. " + checks.DISCLAIMER, {"no_personal_amount"}))["no amount or share of money for this person"]
    assert dict(checks.check(checks.MNPI_LINE + ". " + checks.DISCLAIMER, {"mnpi"}))["declines to use possible inside information"]


# --------------------------------------------------------------------------- the loop, against a stand-in client

class FakeStream:
    def __init__(self, msg): self.msg = msg
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self): return self.msg


class FakeClient:
    """Replays scripted responses and records each request's keyword arguments."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return FakeStream(self.responses.pop(0))


def text_block(t): return NS(type="text", text=t)
def tool_use(i, name, inp): return NS(type="tool_use", id=i, name=name, input=inp)


def test_loop_runs_tools_and_returns_the_memo():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "calculate", {"expression": "2*3"}),
                                            tool_use("t2", "calculate", {"expression": "1/0"})]),
        NS(stop_reason="end_turn", content=[text_block(GOOD)]),
    ])
    memo = ipo_bot.Bot(client=client, log=lambda s: None).ask("Rate X")
    assert memo == GOOD
    second = client.requests[1]["messages"]
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"], "both results in one user message"
    assert json.loads(results[0]["content"])["result"] == 6 and results[1]["is_error"] is True


def test_requests_carry_the_model_fallback_cache_and_date():
    client = FakeClient([NS(stop_reason="end_turn", content=[text_block(GOOD)])])
    ipo_bot.Bot(client=client, log=lambda s: None).ask("Rate X")
    r = client.requests[0]
    assert r["model"] == "claude-opus-5-5" and r["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in r["betas"]
    # The system prompt is exactly the file, with no date in it, so it is the same bytes on every request.
    assert r["system"][0]["cache_control"] == {"type": "ephemeral"} and r["system"][0]["text"] == ipo_bot.SYSTEM
    assert r["messages"][0]["content"].startswith("Now: ")
    assert r["tools"][0]["type"] == "web_search_20260209"
    assert all(t.get("eager_input_streaming") for t in r["tools"][1:])


def test_pause_turn_continues_and_refusal_stops():
    client = FakeClient([NS(stop_reason="pause_turn", content=[text_block("searching")]),
                         NS(stop_reason="end_turn", content=[text_block(GOOD)])])
    assert ipo_bot.Bot(client=client, log=lambda s: None).ask("Rate X") == GOOD
    assert client.requests[1]["messages"][-1]["role"] == "assistant", "the paused turn is sent back to continue"
    refused = FakeClient([NS(stop_reason="refusal", content=[])])
    out = ipo_bot.Bot(client=refused, log=lambda s: None).ask("x")
    assert out.endswith(checks.DISCLAIMER)


def test_filing_text_drops_the_hidden_xbrl_header():
    raw = ('<html><body><div style="display:none"><ix:header><ix:hidden>dei:DocumentType 10-K</ix:hidden>'
           '</ix:header></div><p>UNITED STATES</p><p>SECURITIES AND EXCHANGE COMMISSION</p></body></html>')
    text = tools._strip_html(raw)
    assert text.startswith("UNITED STATES") and "DocumentType" not in text


def test_financials_say_how_long_each_period_is(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2025-09-28", "end": "2026-06-27", "val": 300, "form": "10-Q", "filed": "2026-07-31", "fy": 2026, "fp": "Q3"},
        {"start": "2026-03-29", "end": "2026-06-27", "val": 90, "form": "10-Q", "filed": "2026-07-31", "fy": 2026, "fp": "Q3"},
        {"end": "2026-06-27", "val": 5, "form": "8-K", "filed": "2026-07-31"}]}}}}}
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: json.dumps(facts))
    text, err = tools.run_tool("edgar_financials", {"cik": "1", "concepts": ["Revenues"]})
    rows = json.loads(text)["data"]["Revenues (USD)"]
    assert not err and sorted(r["period_days"] for r in rows) == [91, 273], "a year-to-date and a quarter, told apart"
    assert all(r["form"] == "10-Q" for r in rows), "only periodic reports and registration statements"


def test_env_file_fills_unset_variables_only(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nSEC_USER_AGENT="A Person a@example.com"\nFRED_API_KEY=abc\nEMPTY=\nnot a line\n', encoding="utf-8")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "already-set")
    monkeypatch.delenv("EMPTY", raising=False)
    ipo_bot.load_env(f)
    import os
    assert os.environ["SEC_USER_AGENT"] == "A Person a@example.com"
    assert os.environ["FRED_API_KEY"] == "already-set", "the environment wins over the file"
    assert "EMPTY" not in os.environ
