"""bias_checks.py offline: short histories, the paper-replay signal's checks, and the report on made-up prices."""
import datetime as dt
import math
import runpy

import pytest

from eval_helpers import ROOT, plain_streams

import bias_checks as B  # noqa: E402
import paper_replay as R  # noqa: E402
import strategies as S  # noqa: E402


def days(n, start=dt.date(2015, 1, 1)):
    return [(start + dt.timedelta(days=k)).isoformat() for k in range(n)]


def test_a_history_too_short_to_sample_checks_no_days_instead_of_crashing():
    closes = {"SPY": [100.0 + k for k in range(120)]}
    assert B.lookahead(S.trend, closes, days(120)) == ([], [])
    assert B.replay_signal_checks([("d", 1, 1, 1, 1, 1)] * 150)[:2] == ([], [])


def rows(n=320):
    """(date, open, high, low, close, volume): a rising, wavy stock that breaks out now and then."""
    out = []
    for k, d in enumerate(days(n)):
        c = 50 * math.exp(0.002 * k + 0.05 * math.sin(k / 9))
        out.append((d, c, c * 1.01, c * 0.99, c, 1000))
    return out


def test_the_replay_signal_has_no_look_ahead_and_its_warm_up_answers():
    rs = rows()
    checked, bad, warm = B.replay_signal_checks(rs, samples=8)
    assert len(checked) == 8 and all(200 <= i < len(rs) - 1 for i in checked) and bad == []
    assert list(warm) == ["all", 1000, 500, 300]
    assert warm["all"] == warm[1000] == warm[500] == R.signal_on(rs, len(rs) - 1)
    assert warm[300] == R.signal_on(rs[-300:], 299)


def synthetic_aligned(*symbols, mode="adjusted"):
    ds = days(300)
    bars = {"SPY": [(100 * math.exp(0.001 * k + 0.04 * math.sin(k / 11)),) * 2 for k in range(300)],
            "IEF": [(100 + 0.01 * k,) * 2 for k in range(300)]}
    return ds, bars, {}


def test_the_report_and_main_on_made_up_prices(monkeypatch):
    monkeypatch.setattr(S, "aligned", synthetic_aligned)
    monkeypatch.setattr(R, "bars", lambda symbol, days=420: rows())
    out = plain_streams(monkeypatch)
    B.main()
    text = out.getvalue()
    assert text.startswith("# Bias checks")
    for make in S.BOTS.values():
        line = next(l for l in text.splitlines() if l.startswith(f"| {make()['name']} |"))
        assert line.split(" | ")[1:3] == ["30", "0"]                          # 30 days checked, no look-ahead
    trend = next(l for l in text.splitlines() if l.startswith("| Trend 50/200 |"))
    assert trend.endswith("| no |")                                           # a moving average forgets its start
    replay = next(l for l in text.splitlines() if l.startswith("| Paper replay breakout (AAPL) |"))
    assert replay.split(" | ")[1:3] == ["30", "0"]


def test_a_cheat_is_listed_with_its_first_days(monkeypatch):
    cheat = lambda: {"name": "Cheat", "assets": ["SPY"], "params": {},
                     "decide": lambda i, c, held: {"SPY": 1.0 if i + 1 < len(c["SPY"]) and c["SPY"][i + 1] > c["SPY"][i] else 0.0}}
    monkeypatch.setattr(S, "BOTS", {"cheat": cheat})
    monkeypatch.setattr(S, "aligned", synthetic_aligned)
    monkeypatch.setattr(R, "bars", lambda symbol, days=420: rows())
    line = next(l for l in B.report().splitlines() if l.startswith("| Cheat |"))
    found = int(line.split(" | ")[2].split(" ")[0])
    assert found > 0 and line.split(" | ")[2].count(", ") == min(found, 3) - 1


def test_the_script_entry_point_runs_main(monkeypatch):
    class Stop(Exception):
        pass

    def aligned(*a, **k):
        raise Stop
    monkeypatch.setattr(S, "aligned", aligned)
    plain_streams(monkeypatch)
    with pytest.raises(Stop):
        runpy.run_path(str(ROOT / "evaluation" / "bias_checks.py"), run_name="__main__")
