"""The research bot's loop, CLI and auditor (src/ipo_bot.py, src/audit.py, src/common.py), offline: the Anthropic
client is faked, so no request leaves the machine and no API credit is spent."""
import builtins
import importlib.util
import io
import json
import pathlib
import sys
from types import SimpleNamespace as NS

import anthropic
import types
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import audit  # noqa: E402
import checks  # noqa: E402
import common  # noqa: E402
import ipo_bot  # noqa: E402
import tools  # noqa: E402


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    """Replays scripted replies on both client.beta.messages.stream and client.messages.stream."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.beta = NS(messages=NS(stream=self._stream))
        self.messages = NS(stream=self._stream)

    def _stream(self, **kw):
        self.requests.append(kw)
        return FakeStream(self.replies.pop(0))


def text(t):
    return NS(type="text", text=t)


def reply(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks))


# anthropic's errors read only these attributes, so no HTTP library is imported (the SDK switched libraries
# between versions)
REQ = types.SimpleNamespace(method="POST", url="https://api.anthropic.com/v1/messages")


def fake_response(code):
    return types.SimpleNamespace(status_code=code, headers={}, request=REQ)


def status_error(cls, code):
    return cls("boom", response=fake_response(code), body=None)


# ----------------------------------------------------------------------------- common

def test_utf8_console_reconfigures_what_it_can_and_skips_what_it_cannot():
    good = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    common.utf8_console([good, io.StringIO(), NS()])                 # the last two have no reconfigure()
    assert good.encoding == "utf-8"

    class Broken:
        def reconfigure(self, **kw):
            raise ValueError("detached")
    common.utf8_console([Broken()])
    common.utf8_console()                                            # the real streams: no error either way


def test_every_src_script_finds_the_repo_root_the_same_way():
    import monitor
    import paper
    import track
    assert ipo_bot.HERE == audit.HERE == track.HERE == paper.HERE == common.HERE == SRC.parent
    assert monitor.KILL_FILE.parent.parent.parent == common.HERE and tools.LEDGER.parent.parent == common.HERE


def load_without_truststore(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "truststore", None)              # `import truststore` raises ImportError
    spec = importlib.util.spec_from_file_location(f"_{name}_no_truststore", SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_bot_and_tracker_import_without_truststore(monkeypatch):
    monkeypatch.setattr(ipo_bot, "load_env", lambda *a: None)
    assert load_without_truststore(monkeypatch, "ipo_bot").MODEL == ipo_bot.MODEL
    assert load_without_truststore(monkeypatch, "track").BENCHMARK == "SPY"


# ----------------------------------------------------------------------------- the bot

def test_with_a_key_the_client_is_built_without_any_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert isinstance(ipo_bot.client_or_exit(), anthropic.Anthropic)


def test_load_env_ignores_a_missing_file(tmp_path):
    assert ipo_bot.load_env(tmp_path / "none.env") is None


def test_a_backtest_dates_the_turn_and_turns_off_web_search():
    client = FakeClient([reply("end_turn", text("memo"))])
    assert ipo_bot.Bot(client=client, log=lambda s: None, as_of="2026-01-05").ask("Rate X") == "memo"
    r = client.requests[0]
    assert r["messages"][0]["content"].startswith("Now: Monday 05 January 2026 (backtest")
    assert all(t.get("name") != "web_search" for t in r["tools"])


def test_no_tools_mode_sends_no_tools_and_says_so_and_portfolio_mode_is_announced(monkeypatch):
    monkeypatch.setitem(tools.PORTFOLIO, "path", "pf.json")
    client = FakeClient([reply("end_turn", text("memo"))])
    ipo_bot.Bot(client=client, log=lambda s: None, use_tools=False).ask("q")
    r = client.requests[0]
    assert "tools" not in r
    assert "Portfolio mode: on" in r["messages"][0]["content"] and "Tools: none are available" in r["messages"][0]["content"]


def test_a_reply_cut_at_max_tokens_says_so_and_a_loop_that_never_answers_stops(monkeypatch):
    cut = ipo_bot.Bot(client=FakeClient([reply("max_tokens", text("half a memo"))]), log=lambda s: None).ask("q")
    assert cut.startswith("half a memo\n\n[cut off at the length limit]") and cut.endswith(checks.DISCLAIMER)
    monkeypatch.setattr(ipo_bot, "MAX_TURNS", 2)
    call = NS(type="tool_use", id="t", name="calculate", input="not a dict")
    bot = ipo_bot.Bot(client=FakeClient([reply("tool_use", call), reply("tool_use", call)]), log=lambda s: None)
    assert bot.ask("q").startswith("Stopped after too many tool rounds")
    assert bot.sources[0]["input"] == {} and bot.sources[0]["error"] is True     # a non-dict input runs as {}


def test_web_search_results_are_kept_for_the_auditor():
    hits = NS(type="web_search_tool_result", content=[NS(title="T", url="https://x", page_age="1 day")])
    failed = NS(type="web_search_tool_result", content=NS(error_code="unavailable"))
    bot = ipo_bot.Bot(client=FakeClient([reply("end_turn", hits, failed, text("memo"))]), log=lambda s: None)
    assert bot.ask("q") == "memo"
    assert bot.sources == [{"tool": "web_search", "input": {},
                            "result": json.dumps([{"title": "T", "url": "https://x", "page_age": "1 day"}]), "error": False}]


def test_save_writes_the_memo_and_its_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(ipo_bot, "HERE", tmp_path)
    path = ipo_bot.save("Rate X", "the memo", [{"tool": "calculate"}])
    assert path.read_text(encoding="utf-8") == "> Rate X\n\nthe memo\n"
    assert json.loads((path.parent / f"{path.stem}.sources.json").read_text(encoding="utf-8")) == [{"tool": "calculate"}]
    alone = ipo_bot.save("Q", "m")                                   # within the same second: a new file, not an overwrite
    assert alone != path and path.read_text(encoding="utf-8").startswith("> Rate X")
    assert not (alone.parent / f"{alone.stem}.sources.json").exists()
    third = ipo_bot.save("Q3", "m3")
    assert len({path, alone, third}) == 3


# ----------------------------------------------------------------------------- the bot's command line

class ScriptedBot:
    """Bot() for main(): answers (or raises) from a script; records questions."""
    script, asked = [], []

    def __init__(self):
        self.sources = []

    def ask(self, q):
        ScriptedBot.asked.append(q)
        nxt = ScriptedBot.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def bot_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(ipo_bot, "HERE", tmp_path)
    monkeypatch.setattr(ipo_bot, "Bot", ScriptedBot)
    monkeypatch.setitem(tools.PORTFOLIO, "path", None)
    ScriptedBot.script, ScriptedBot.asked = [], []
    return tmp_path


def test_one_question_prints_and_saves_the_memo_then_audits_it(bot_cli, monkeypatch, capsys):
    ScriptedBot.script = ["MEMO TEXT"]
    monkeypatch.setattr(audit, "audit", lambda memo, sources: {"verdict": "pass", "checks": [], "summary": "ok", "mechanical": []})
    ipo_bot.main(["--audit", "Rate", "X"])
    out = capsys.readouterr().out
    assert ScriptedBot.asked == ["Rate X"] and "MEMO TEXT" in out and "Verdict: PASS" in out
    assert len(list((bot_cli / "memos").glob("*.md"))) == 1


def test_an_audit_that_fails_at_the_api_is_reported_not_raised(bot_cli, monkeypatch, capsys):
    ScriptedBot.script = ["MEMO"]

    def broken(memo, sources):
        raise anthropic.APIConnectionError(request=REQ)
    monkeypatch.setattr(audit, "audit", broken)
    ipo_bot.main(["--audit", "q"])
    assert "Audit failed: Connection error." in capsys.readouterr().out


def test_the_portfolio_flag_needs_a_valid_file_and_alone_asks_for_a_review(bot_cli, capsys):
    with pytest.raises(SystemExit, match="--portfolio needs a file"):
        ipo_bot.main(["--portfolio"])
    bad = bot_cli / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="Portfolio file: the portfolio file must be a JSON object"):
        ipo_bot.main(["--portfolio", str(bad)])
    good = bot_cli / "pf.json"
    good.write_text(json.dumps({"cash": 100}), encoding="utf-8")
    ScriptedBot.script = ["REVIEW"]
    ipo_bot.main(["--portfolio", str(good)])
    assert ScriptedBot.asked == [ipo_bot.REVIEW] and tools.PORTFOLIO["path"] == str(good.resolve())


def test_interactive_mode_rides_out_api_errors_and_quits_on_a_blank_line(bot_cli, monkeypatch, capsys):
    ScriptedBot.script = [status_error(anthropic.RateLimitError, 429), status_error(anthropic.InternalServerError, 500),
                          anthropic.APIConnectionError(request=REQ), "ANSWER"]
    answers = iter(["a", "b", "c", "d", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(answers))
    ipo_bot.main([])
    out = capsys.readouterr().out
    assert "Rate limited" in out and "Claude API error 500: boom" in out and "Could not reach the Claude API." in out
    assert "ANSWER" in out and ScriptedBot.asked == ["a", "b", "c", "d"]


def test_a_rejected_key_or_a_client_that_cannot_start_exits_with_a_plain_message(bot_cli, monkeypatch):
    ScriptedBot.script = [status_error(anthropic.AuthenticationError, 401)]
    with pytest.raises(SystemExit, match="rejected the API key"):
        ipo_bot.main(["q"])

    def no_start():
        raise anthropic.AnthropicError("no credentials")
    monkeypatch.setattr(ipo_bot, "Bot", no_start)
    with pytest.raises(SystemExit, match="Could not start: no credentials"):
        ipo_bot.main(["q"])


# ----------------------------------------------------------------------------- the auditor

MEMO = "As of 2026-10-01\nA memo.\n" + checks.DISCLAIMER
ALL_PASS = {"checks": [{"id": r, "result": "pass", "reason": "fine"} for r in audit.RULES], "summary": "good", "verdict": "pass"}


def test_long_sources_are_truncated_in_the_request(monkeypatch):
    monkeypatch.setattr(audit, "SOURCES_CHARS", 50)
    content, mech = audit.build_request(MEMO, [{"result": "x" * 500}])
    assert "...[sources truncated]" in content and "SOURCES (1 tool results)" in content
    assert "- FAIL KEY NUMBERS block present and valid (no ```json block" in content


def test_the_verdict_is_recomputed_from_the_checks_and_the_mechanical_results():
    mech_ok, mech_bad = [("block", True, "")], [("block", False, "missing")]
    assert audit.parse(json.dumps(ALL_PASS), mech_ok)["verdict"] == "pass"
    assert audit.parse(json.dumps(ALL_PASS), mech_bad)["verdict"] == "fail"       # a failed mechanical check fails it
    partial = audit.parse(json.dumps({"checks": [{"id": "A1", "result": "pass", "reason": ""}, {"id": "Z9"}]}), mech_ok)
    assert partial["verdict"] == "fail" and partial["checks"][1]["reason"] == "the auditor gave no result for this rule"
    assert partial["summary"] == ""
    assert audit.parse("[1, 2]", mech_ok)["summary"] == "the auditor's reply was not readable JSON"


def test_audit_sends_one_structured_request_and_handles_a_refusal():
    client = FakeClient([reply("end_turn", text(json.dumps(ALL_PASS)), NS(type="thinking"))])
    memo = MEMO.replace("A memo.", "A memo.\n```json\n" + json.dumps({"key_numbers": {}}) + "\n```")
    result = audit.audit(memo, [{"tool": "t"}], client=client)
    r = client.requests[0]
    assert r["max_tokens"] == audit.MAX_TOKENS and r["output_config"]["format"]["schema"] is audit.SCHEMA
    assert result["checks"][0]["result"] == "pass" and result["mechanical"][0]["pass"] is True
    refused = audit.audit(MEMO, client=FakeClient([reply("refusal")]))
    assert refused["verdict"] == "fail" and refused["summary"] == "the auditor declined to review this memo"


def test_the_report_lists_checks_and_failed_mechanical_checks():
    out = audit.report({"verdict": "fail", "summary": "s", "checks": [{"id": "A1", "result": "fail", "reason": "why"}],
                        "mechanical": [{"check": "c1", "pass": True, "detail": ""}, {"check": "c2", "pass": False, "detail": "d"}]})
    assert out.splitlines() == ["Verdict: FAIL", "s", "  A1  fail           why", "  mechanical: 1 of 2 pass", "    FAIL c2 (d)"]


def test_the_audit_cli(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        audit.main([])
    memo = tmp_path / "m.md"
    memo.write_text(MEMO, encoding="utf-8")
    src = tmp_path / "s.json"
    src.write_text('[{"tool": "x"}]', encoding="utf-8")
    seen = []

    def fake(m, sources):
        seen.append(sources)
        return {"verdict": "pass" if sources else "fail", "summary": "", "checks": [], "mechanical": []}
    monkeypatch.setattr(audit, "audit", fake)
    assert audit.main([str(memo), "--sources", str(src)]) == 0 and seen[-1] == [{"tool": "x"}]
    assert audit.main([str(memo)]) == 1 and seen[-1] is None

    def rejected(m, sources):
        raise status_error(anthropic.AuthenticationError, 401)
    monkeypatch.setattr(audit, "audit", rejected)
    with pytest.raises(SystemExit, match="rejected the API key"):
        audit.main([str(memo)])


def test_a_portfolio_with_a_question_asks_that_question(bot_cli):
    good = bot_cli / "pf.json"
    good.write_text(json.dumps({"cash": 100}), encoding="utf-8")
    ScriptedBot.script = ["ANSWER"]
    ipo_bot.main(["--portfolio", str(good), "Buy", "ABC?"])
    assert ScriptedBot.asked == ["Buy ABC?"]


@pytest.mark.parametrize("script, argv, code", [
    ("audit", [], None),                       # usage: sys.exit(__doc__)
    ("verify", [], None),
    ("paper", [], 1),
    ("track", ["no-such-ledger.jsonl"], 0),
    ("ipo_bot", ["--portfolio"], None),        # exits before any client is made
])
def test_each_script_runs_as_a_program_without_the_network(script, argv, code, monkeypatch, capsys):
    import runpy
    monkeypatch.setattr(sys, "argv", [script + ".py", *argv])
    monkeypatch.setattr(ipo_bot, "load_env", lambda *a: None)
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(SRC / f"{script}.py"), run_name="__main__")
    assert (e.value.code == code) if code is not None else isinstance(e.value.code, str)
