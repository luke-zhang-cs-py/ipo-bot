"""Two bias detectors for any signal rule, after freqtrade's lookahead-analysis and recursive-analysis
(re-implemented from their documented method; freqtrade is GPL-3.0 and no code is copied).

    python evaluation/bias_checks.py

Look-ahead: a rule decides at day i from the bars up to i. Run it on the full history and on the history
cut at i, at many sampled days; any difference means the rule read a bar from after its own day.
Recursive (warm-up): run the rule at the last day with less and less history before it (all of it, then 1,000,
500, 300 bars). A rule whose answer moves with how far back the data starts will trade differently live, where
only a limited history is loaded; freqtrade found this with recursive indicators such as EMAs.

Checked: the five bots in strategies.py and the breakout signal in paper_replay.py.
"""
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

import strategies as S  # noqa: E402

WARMUPS = (None, 1000, 500, 300)


def _decide(make, closes, i, dates):
    """A fresh bot's decision at day i, replayed from day 0 so stateful bots (mean reversion, the grid) build
    their state the way a live run would. held is taken from the bot's own previous decision."""
    bot = make()
    held = None
    out = None
    for k in range(i + 1):
        args = (k, closes, held, {k: dates[k]}) if bot.get("needs_dates") else (k, closes, held)
        out = bot["decide"](*args)
        held = out
    return out


def lookahead(make, closes, dates, samples=30, seed=5):
    """Days where the decision on the full history differs from the decision on history cut at that day."""
    rng = random.Random(seed)
    n = len(dates)
    days = sorted(rng.sample(range(250, n - 1), min(samples, n - 251)))
    bad = []
    for i in days:
        full = _decide(make, closes, i, dates)
        cut = {a: xs[:i + 1] for a, xs in closes.items()}
        if _decide(make, cut, i, dates[:i + 1]) != full:
            bad.append(dates[i])
    return days, bad


def recursive(make, closes, dates, warmups=WARMUPS):
    """The decision at the last day given each amount of history before it: {warm-up: decision}."""
    out = {}
    n = len(dates)
    for w in warmups:
        start = 0 if w is None else max(0, n - w)
        sub = {a: xs[start:] for a, xs in closes.items()}
        out["all" if w is None else w] = _decide(make, sub, n - 1 - start, dates[start:])
    return out


def replay_signal_checks(rows, samples=30, seed=5):
    """The paper-replay breakout signal: look-ahead days and warm-up answers."""
    import paper_replay as R
    rng = random.Random(seed)
    days = sorted(rng.sample(range(200, len(rows) - 1), min(samples, len(rows) - 201)))
    bad = [rows[i][0] for i in days if R.signal_on(rows, i) != R.signal_on(rows[:i + 1], i)]
    n = len(rows)
    warm = {("all" if w is None else w): R.signal_on(rows[(0 if w is None else max(0, n - w)):], (n - 1) - (0 if w is None else max(0, n - w)))
            for w in WARMUPS}
    return days, bad, warm


def report():
    dates, bars, _ = S.aligned("SPY", "IEF")
    closes = {a: [c for _, c in bars[a]] for a in bars}
    out = ["# Bias checks", "", "| Rule | days checked | look-ahead found | answer by warm-up (all / 1000 / 500 / 300 bars) | warm-up dependent |",
           "|---|---|---|---|---|"]
    for key, make in S.BOTS.items():
        days, bad = lookahead(make, closes, dates)
        warm = recursive(make, closes, dates)
        vals = [str(v) for v in warm.values()]
        out.append(f"| {make()['name']} | {len(days)} | {len(bad)}{' (' + ', '.join(bad[:3]) + ')' if bad else ''} | "
                   f"{' / '.join(vals)} | {'yes' if len(set(vals)) > 1 else 'no'} |")
    import paper_replay as R
    rows = R.bars("AAPL", days=1500)
    days, bad, warm = replay_signal_checks(rows)
    vals = [str(v) for v in warm.values()]
    out.append(f"| Paper replay breakout (AAPL) | {len(days)} | {len(bad)} | {' / '.join(vals)} | {'yes' if len(set(vals)) > 1 else 'no'} |")
    out += ["", "Warm-up dependence is expected for the stateful bots (the grid's range is reset each January from the "
            "first close it sees; mean reversion remembers when it entered): it means a live run must load enough history "
            "first, not that the backtest cheats. Look-ahead found should always be 0.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    print(report())
