"""The paper-trading replay on synthetic bars: the signal reads only the past, stops sell at the stop (or the
open below it), guardrails carry from one day to the next, and every signal is accounted for. Offline."""
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import paper  # noqa: E402
import paper_replay as R  # noqa: E402


def bars(closes, gap=0.0, lows=None):
    """[(date, open, high, low, close, volume)] on weekdays; each day opens at yesterday's close (+gap)."""
    out, day = [], dt.date(2025, 1, 6)
    for k, c in enumerate(closes):
        while day.weekday() > 4:
            day += dt.timedelta(days=1)
        o = (closes[k - 1] if k else c) * (1 + gap)
        lo = lows[k] if lows and lows[k] is not None else min(o, c) * 0.995
        out.append((day.isoformat(), o, max(o, c) * 1.005, lo, c, 5_000_000))
        day += dt.timedelta(days=1)
    return out


def rising(n=260, step=0.002):
    return [100 * (1 + step) ** k for k in range(n)]


def test_the_signal_reads_only_closes_up_to_its_day():
    rows = bars(rising())
    assert R.signal_on(rows, 150) is None                       # under 200 days of history
    stop = R.signal_on(rows, 259)
    assert stop is not None and stop < rows[259][4]
    later = rows[:260] + [(d, o * 0.5, h * 0.5, lo * 0.5, c * 0.5, v) for d, o, h, lo, c, v in bars(rising(), gap=0)[:5]]
    assert R.signal_on(later, 259) == stop                      # what comes after the day changes nothing


def test_a_replay_buys_breakouts_and_accounts_for_every_signal(tmp_path):
    data = {"AAPL": bars(rising()), "XOM": bars([100.0] * 260)}  # one breaking out, one flat
    lines, account, exits = R.replay(data, days=5, log=tmp_path / "log.jsonl")
    assert len(account) == 5 and all(ln["result"]["status"] for ln in lines)
    assert {ln["signal"]["symbol"] for ln in lines} == {"AAPL"}
    first = next(ln for ln in lines if ln["result"].get("filled"))
    assert first["result"]["orders"][0]["order_id"].startswith("bar-")
    assert sum(paper.divergence(lines)["outcomes"].values()) == len(lines)


def test_a_stop_sells_at_the_stop_or_at_the_open_when_it_gaps_below(tmp_path):
    closes = rising(255) + [rising(255)[-1] * 1.003] * 2
    crash = closes + [closes[-1] * 0.80] * 3                     # a 20% gap down after the buy
    data = {"AAPL": bars(crash)}
    lines, account, exits = R.replay(data, days=4, log=tmp_path / "log.jsonl")
    assert exits, "the stop should have been hit"
    e = exits[0]
    bought = next(ln for ln in lines if ln["result"].get("filled"))
    assert e["price"] <= bought["signal"]["stop"] + 1e-9         # never better than the stop when it gaps through
    assert e["pnl"] < 0


def test_a_position_bought_at_the_open_is_stopped_by_the_same_days_low(tmp_path):
    lows = [None] * 257 + [1.0, None, None]                      # bar 257: bought at its open, then a plunge
    data = {"AAPL": bars(rising(), lows=lows)}
    lines, account, exits = R.replay(data, days=3, log=tmp_path / "log.jsonl")
    bought = next(ln for ln in lines if ln["result"].get("filled"))
    assert bought["day"] == data["AAPL"][257][0]
    assert exits and exits[0]["day"] == bought["day"], "the same day's low should have taken out the stop"
    assert abs(exits[0]["price"] - bought["signal"]["stop"]) < 0.01   # sold at the stop, the open was above it
    assert account[0]["positions"] == 0


def test_the_daily_loss_limit_set_by_yesterday_blocks_todays_buys(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RULES", {**R.RULES, "max_daily_loss_pct": 0.05})
    aapl = bars(rising(262))                                # bought on the first replay day (bar 257)
    d, o, h, lo, c, v = aapl[259]
    aapl[259] = (d, o * 0.97, h, min(lo, o * 0.97), c, v)   # then opens 3% lower on bar 259: a loss at the open
    googl = bars([100.0] * 258 + [110.0] * 4)               # breaks out at the close of bar 258: a buy on bar 259
    lines, _, _ = R.replay({"AAPL": aapl, "GOOGL": googl}, days=5, log=tmp_path / "log.jsonl")
    g = [ln for ln in lines if ln["signal"]["symbol"] == "GOOGL"]
    assert g and g[0]["result"]["status"] == "blocked" and g[0]["result"]["blocked_by"] == "daily_loss", [
        (ln["day"], ln["signal"]["symbol"], ln["result"]["status"], ln["result"].get("blocked_by")) for ln in lines]


def test_tracker_accordance_measures_against_spy():
    import tracker as T
    days = [(dt.date(2026, 1, 5) + dt.timedelta(days=k)).isoformat() for k in range(260)]
    spy = {d: 100 * 1.001 ** k for k, d in enumerate(days)}
    account = [{"day": d, "equity": 1000 * (1 + 0.5 * (spy[d] / spy[days[0]] - 1)), "cash": 500.0} for d in days[220:]]
    a = T.accordance(account, spy, spy)
    assert abs(a["spy_return"] - (spy[days[-1]] / spy[days[220]] - 1)) < 1e-12
    assert a["days"] == 40 and a["days_above_200d"] == 40 and a["days_below_200d"] == 0
    assert a["invested_when_spy_above_200d"] < 0.6
    assert T.drawdown([100, 120, 90, 130]) == 90 / 120 - 1


def test_tracker_bootstrap_gives_nan_when_no_resample_can_be_measured():
    import math
    import statistics
    import tracker as T
    lo, hi = T.boot([0.0] * 20, [0.01 * k for k in range(20)], statistics.correlation, n=50)   # an all-cash account
    assert math.isnan(lo) and math.isnan(hi)


def test_the_tracker_refuses_to_run_on_an_incomplete_universe(tmp_path, monkeypatch):
    import pytest
    import tracker as T
    full = bars(rising(300))
    monkeypatch.setattr(R, "bars", lambda s, days=420: [] if s in ("AAPL", "XOM") else full)
    monkeypatch.setattr(T, "OUT", tmp_path / "tracking")
    with pytest.raises(RuntimeError, match="AAPL, XOM"):
        T.run()
    assert not (tmp_path / "tracking").exists()                 # nothing recorded for a different universe
