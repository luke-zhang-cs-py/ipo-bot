"""stress_test.py offline, with a fake Bot (no API key, no credit): the --yes guard, question numbers, the saved
memos and checks, and that a run leaves the real forecast ledger and portfolio settings as it found them."""
import runpy
import sys

import pytest

from eval_helpers import ROOT, plain_streams

import stress_test as st  # noqa: E402
import tools  # noqa: E402


class FakeBot:
    asked = []

    def __init__(self, log=print, **kw):
        self.log = log

    def ask(self, question):
        FakeBot.asked.append((question, tools.LEDGER, tools.PORTFOLIO["path"]))
        return "GOOD memo" if "ZQXJ" in question or "portfolio" in question else "BAD memo"


def fake_check(memo, expect=()):
    return [("first rule", True), ("second rule", memo.startswith("GOOD"))]


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeBot.asked = []
    monkeypatch.setattr(st, "Bot", FakeBot)
    monkeypatch.setattr(st, "check", fake_check)
    monkeypatch.setattr(st, "HERE", tmp_path)
    return tmp_path


def test_nothing_runs_without_yes_or_with_a_question_number_out_of_range(fake):
    with pytest.raises(SystemExit) as e:
        st.main(["3"])
    assert "costs API credit" in str(e.value.code)
    for bad in ("0", "12"):
        with pytest.raises(SystemExit) as e:
            st.main(["--yes", bad])
        assert str(e.value.code) == f"Questions are numbered 1 to {len(st.CASES)}."
    assert FakeBot.asked == [] and not (fake / "stress_runs").exists()


def test_picked_questions_are_saved_with_their_checks_and_the_globals_are_restored(fake, monkeypatch):
    before = tools.LEDGER, tools.PORTFOLIO["path"]
    out = plain_streams(monkeypatch)
    assert st.main(["--yes", "3", "11"]) == 0                               # both memos pass every check
    (run,) = (fake / "stress_runs").iterdir()
    (q3, l3, p3), (q11, l11, p11) = FakeBot.asked
    assert l3 == l11 == run / "ledger.jsonl"                                # forecasts kept out of the real log
    assert p3 is None and p11 == str(fake / "portfolio.example.json")       # only case 11 loads the portfolio
    saved = (run / "q11.md").read_text(encoding="utf-8")
    assert saved.startswith(f"> {st.CASES[10][0]}\n\nGOOD memo\n\n---\nChecks:\n- [x] first rule\n- [x] second rule")
    assert "2 of 2 passed every check" in out.getvalue() and "all checks pass" in out.getvalue()
    assert (tools.LEDGER, tools.PORTFOLIO["path"]) == before


def test_a_failed_check_is_reported_and_counted(fake, monkeypatch):
    out = plain_streams(monkeypatch)
    assert st.main(["--yes", "1"]) == 1
    assert "FAILED: second rule" in out.getvalue() and "0 of 1 passed" in out.getvalue()
    (run,) = (fake / "stress_runs").iterdir()
    assert "- [ ] second rule" in (run / "q1.md").read_text(encoding="utf-8")


def test_every_case_runs_when_none_is_picked(fake, monkeypatch):
    plain_streams(monkeypatch)
    st.main(["--yes"])
    assert [q for q, _, _ in FakeBot.asked] == [q for q, _ in st.CASES]


def test_a_crash_mid_run_still_restores_the_ledger_and_portfolio(fake, monkeypatch):
    before = tools.LEDGER, tools.PORTFOLIO["path"]

    class Broken(FakeBot):
        def ask(self, question):
            raise RuntimeError("API down")
    monkeypatch.setattr(st, "Bot", Broken)
    plain_streams(monkeypatch)
    with pytest.raises(RuntimeError):
        st.main(["--yes", "11"])
    assert (tools.LEDGER, tools.PORTFOLIO["path"]) == before


def test_the_script_entry_point_refuses_without_yes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stress_test.py"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(ROOT / "evaluation" / "stress_test.py"), run_name="__main__")
    assert "Run with --yes" in str(e.value.code)
