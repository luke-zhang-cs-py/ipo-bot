"""The running check: how the bot's paper account lines up with the market and its trends, re-measured each run.

    python evaluation/tracker.py               replay from TRACK_START to the latest close; write tracking/

Every run replays the paper bot (paper_replay.py: the same signal, rules, monitor and bar fills) over the whole
window from TRACK_START, so a run is reproducible from the prices alone, then measures:
  - the account against SPY: return, daily correlation, beta, tracking error, worst drawdown of each
  - trend accordance: how invested the bot was on days SPY was above its 200-day average against below it,
    and in each regime (bull, drawdown, sideways, ordinary, labelled from SPY's past only)
  - execution: the divergence report (fills, blocks by rule and monitor, slippage) for the window
  - integrity: the look-ahead check on the replay's signal
tracking/track.json keeps one line per run (so the history of runs builds up when committed), and
tracking/README.md is the latest report.
"""
import datetime as dt
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import benchmark as bm  # noqa: E402
import bias_checks  # noqa: E402
import paper  # noqa: E402
import paper_replay as R  # noqa: E402
from ipo_eval import spy_regime  # noqa: E402

TRACK_START = "2026-01-02"
OUT = ROOT / "tracking"


def returns(xs):
    return [b / a - 1 for a, b in zip(xs, xs[1:])]


drawdown = bm.max_drawdown           # the worst fall from a running high, as the benchmark measures it


def boot(ra, rs, stat, n=1000, seed=3):
    """A 95% interval for a statistic of the paired daily returns, resampling the days. (nan, nan) when the
    statistic cannot be computed on any resample (an account that stayed in cash has no correlation)."""
    rng = random.Random(seed)
    k = len(ra)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        try:
            vals.append(stat([ra[i] for i in idx], [rs[i] for i in idx]))
        except (statistics.StatisticsError, ZeroDivisionError):
            continue
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def _beta(ra, rs):
    ma, ms = statistics.fmean(ra), statistics.fmean(rs)
    return statistics.fmean((x - ma) * (y - ms) for x, y in zip(ra, rs)) / statistics.pvariance(rs)


def accordance(account, spy_close, spy_hist):
    """Return, correlation, beta and tracking error against SPY, and exposure by trend and regime."""
    days = [a["day"] for a in account]
    eq = [a["equity"] for a in account]
    sp = [spy_close[d] for d in days]
    ra, rs = returns(eq), returns(sp)
    var = statistics.pvariance(rs) if len(rs) > 1 else 0.0
    beta = _beta(ra, rs) if var else float("nan")
    corr = statistics.correlation(ra, rs) if len(ra) > 2 and statistics.pstdev(ra) and statistics.pstdev(rs) else float("nan")
    te = statistics.pstdev([x - y for x, y in zip(ra, rs)]) * (252 ** 0.5) if len(ra) > 1 else float("nan")
    hist_days = sorted(spy_hist)
    above, below, by_regime = [], [], {}
    for a in account:
        prior = [spy_hist[d] for d in hist_days if d < a["day"]][-200:]
        invested = 1 - a["cash"] / a["equity"] if a["equity"] else 0.0
        if len(prior) == 200:
            (above if prior[-1] > statistics.fmean(prior) else below).append(invested)
        reg = spy_regime(spy_hist, a["day"])
        by_regime.setdefault(reg or "unlabelled", []).append(invested)
    mean = lambda xs: statistics.fmean(xs) if xs else None
    ci = {}
    if len(ra) > 10:
        ci = {"beta_ci": boot(ra, rs, _beta), "correlation_ci": boot(ra, rs, statistics.correlation),
              "excess_ci": boot(ra, rs, lambda a, b: 252 * (statistics.fmean(a) - statistics.fmean(b)))}
    return {"days": len(days), "from": days[0], "to": days[-1], **ci,
            "excess_per_year": 252 * (statistics.fmean(ra) - statistics.fmean(rs)) if ra else float("nan"),
            "account_return": eq[-1] / eq[0] - 1, "spy_return": sp[-1] / sp[0] - 1,
            "account_drawdown": drawdown(eq), "spy_drawdown": drawdown(sp), "correlation": corr, "beta": beta, "tracking_error": te,
            "invested_when_spy_above_200d": mean(above), "invested_when_spy_below_200d": mean(below),
            "days_above_200d": len(above), "days_below_200d": len(below),
            "invested_by_regime": {k: (mean(v), len(v)) for k, v in by_regime.items()}}


def run():
    symbols = bm.UNIVERSES["US large caps"]
    span = (dt.date.today() - dt.date.fromisoformat(TRACK_START)).days + 420
    data = {s: R.bars(s, days=span) for s in symbols}
    data = {s: rows for s, rows in data.items() if len(rows) > 230}
    spy_rows = R.bars("SPY", days=span)
    # A symbol Yahoo didn't return would quietly turn this into a run on a different universe, and track.json
    # would compare it with runs on the full one. Stop instead, and say which are missing.
    missing = [s for s in symbols if s not in data] + ([] if spy_rows else ["SPY"])
    if missing:
        raise RuntimeError(f"no usable price history for {', '.join(missing)} (Yahoo failed or the history is "
                           "too short); nothing written. Run again later.")
    spy_hist = {r[0]: r[4] for r in spy_rows}
    calendar = sorted(set.intersection(*(set(r[0] for r in rows) for rows in data.values())))
    days = len([d for d in calendar if d >= TRACK_START])
    lines, account, exits = R.replay(data, days=days, log=ROOT / "forecasts" / "paper" / "tracker_log.jsonl")
    acc = accordance(account, spy_hist, spy_hist)
    div = paper.divergence(lines)
    look_days, look_bad = bias_checks.replay_signal_checks(data["AAPL"])[:2]
    entry = {"run": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **acc,
             "signals": div["signals"], "outcomes": div["outcomes"], "fill_rate": div["fill_rate_of_planned_shares"],
             "mean_slippage": div["mean_slippage_vs_signal"], "stops_hit": len(exits),
             "lookahead_checked": len(look_days), "lookahead_found": len(look_bad),
             "universe": len(data), "missing": missing}
    OUT.mkdir(exist_ok=True)
    path = OUT / "track.json"
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    # a market holiday, or a second run the same day, measures the same last day again: replace, don't duplicate
    if lines and json.loads(lines[-1]).get("to") == entry["to"]:
        lines.pop()
    lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(render(entry), encoding="utf-8")
    return entry


def pct(x, d=2):
    return "n/a" if x is None or x != x else f"{100 * x:+.{d}f}%"


def level(x):
    """A share (how invested, a fill rate) as a percentage; n/a when there were no days or no orders to measure."""
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def rng_text(pair, fmt):
    return f" (95% CI {fmt(pair[0])} to {fmt(pair[1])})" if pair else ""


def render(e):
    reg = " | ".join(f"{k}: {level(v[0])} ({v[1]} d)" for k, v in sorted(e["invested_by_regime"].items()))
    return "\n".join([
        f"# Tracking: the paper bot against the market, {e['from']} to {e['to']}", "",
        f"Updated {e['run'][:16].replace('T', ' ')} UTC by `python evaluation/tracker.py`; every run is a line in `track.json`. "
        "The bot is the paper replay (a breakout signal on 30 large caps, every guardrail on, fills from daily bars); this "
        "measures how its account moves with the market, not whether it has an edge.", "",
        "| | Account | SPY |", "|---|---|---|",
        f"| Return | {pct(e['account_return'])} | {pct(e['spy_return'])} |",
        f"| Worst drawdown | {pct(e['account_drawdown'])} | {pct(e['spy_drawdown'])} |", "",
        f"Over {e['days']} trading days: daily correlation with SPY {e['correlation']:.3f}{rng_text(e.get('correlation_ci'), lambda v: f'{v:.3f}')}, "
        f"beta {e['beta']:.3f}{rng_text(e.get('beta_ci'), lambda v: f'{v:.3f}')}, tracking error {pct(e['tracking_error'])} a year, "
        f"average daily return against SPY's {pct(e['excess_per_year'])} a year{rng_text(e.get('excess_ci'), pct)}. "
        "Intervals resample the days 1,000 times; an interval that spans zero is no evidence either way.", "",
        "## Trend accordance", "",
        f"Invested on days SPY was above its 200-day average: {level(e['invested_when_spy_above_200d'])} on average ({e['days_above_200d']} days); "
        f"below it: {level(e['invested_when_spy_below_200d'])} ({e['days_below_200d']} days). By regime: {reg}.", "",
        "## Execution", "",
        f"{e['signals']} signals: " + ", ".join(f"{k} {v} ({100 * v / e['signals']:.1f}%)" for k, v in e["outcomes"].items()) +
        f". Fill rate of planned shares {level(e['fill_rate'])}; mean slippage from the signal's price {pct(e['mean_slippage'], 3)}; "
        f"{e['stops_hit']} stops hit.", "",
        f"Look-ahead check on the signal: {e['lookahead_found']} found in {e['lookahead_checked']} sampled days.", ""])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    e = run()
    print(render(e))
    return e


if __name__ == "__main__":  # pragma: no cover - the command line entry; main() is tested
    main()
