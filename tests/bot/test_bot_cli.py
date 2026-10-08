"""The command line, run offline: the synthetic world is recorded once, then the CLI replays those files."""

import datetime as dt
import json

import pytest
from conftest import UA, make_cfg
from world import World, at

from bot import __main__ as cli
from bot import update
from bot.http import Http
from bot.store import lock

NOW = at(dt.date(2025, 10, 8))


@pytest.fixture
def replay(tmp_path):
    rec = tmp_path / "recorded"
    cfg = make_cfg(tmp_path / "rec-run", record_dir=rec, history_start="2025-06-02", symbols=("AAA", "BBB"))
    w = World(now=NOW)
    update.run("all", cfg, NOW, Http(cfg, opener=w.opener, sleep=lambda s: None))
    return rec


def env(monkeypatch, tmp_path, rec):
    for k, v in {"BOT_DATA_DIR": str(tmp_path / "cli"), "BOT_REPLAY": str(rec), "SEC_USER_AGENT": UA,
                 "BOT_SYMBOLS": "AAA,BBB", "BOT_HISTORY_START": "2025-06-02"}.items():
        monkeypatch.setenv(k, v)


def test_update_backtest_retrain_health(tmp_path, monkeypatch, replay, capsys) -> None:
    env(monkeypatch, tmp_path, replay)
    assert cli.main(["health"]) == 1  # nothing yet
    assert cli.main(["update", "--now", "2025-10-08T22:00:00Z"]) == 0
    assert "all: " in capsys.readouterr().out
    assert cli.main(["health"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["coverage"]["universe"] == 2
    assert cli.main(["backtest"]) == 0 and cli.main(["retrain"]) == 0
    with lock(tmp_path / "cli" / "update.lock", 6, dt.datetime.now(dt.UTC)):
        assert cli.main(["update", "--job", "daily", "--now", "2025-10-08T22:00:00Z"]) == 3
    report = json.loads((tmp_path / "cli" / "health" / "latest.json").read_text())
    report["status"] = "failing"
    (tmp_path / "cli" / "health" / "latest.json").write_text(json.dumps(report))
    assert cli.main(["health"]) == 2


def test_module_entry_point(tmp_path, monkeypatch) -> None:
    import runpy
    import sys

    monkeypatch.setenv("BOT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["bot", "health"])
    monkeypatch.delitem(sys.modules, "bot.__main__", raising=False)  # run it fresh, as `python -m bot` does
    with pytest.raises(SystemExit) as e:
        runpy.run_module("bot", run_name="__main__")
    assert e.value.code == 1
