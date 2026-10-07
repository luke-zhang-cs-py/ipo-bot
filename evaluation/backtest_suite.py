"""The benchmark's algorithms through four more backtest checks.

    python evaluation/backtest_suite.py            all four; the out-of-sample test is run once and recorded
    python evaluation/backtest_suite.py --again    re-run the out-of-sample test anyway (logged as another look)

1. Out-of-sample, once. Algorithms are chosen on 2018-2023 (outcomes inside that window only) and tested on
   2024 on. The first test result is written to bench_runs/oos_log.jsonl; later runs show that result instead
   of testing again, because a holdout looked at twice is a holdout tuned to.
2. Regimes, read off the market as it was: each month-end is a bull market (SPY up 15%+ over 12 months, less
   than 10% off its high), a drawdown (10%+ below its 12-month high), sideways (SPY within 5% of a year ago,
   with below-median volatility), or ordinary. Only SPY's past is used to label a month.
3. Costs: each algorithm's top third, rebalanced monthly, after 0 to 200 basis points a side, and the cost at
   which it stops beating simply holding everything (its break-even).
4. Walk-forward re-optimisation: every January, pick the algorithm with the best rank IC over the past 36
   months (outcomes known by then) and use it for the year; against using every algorithm, and against the
   best one in hindsight (which cheats, and is labelled so).
"""
import datetime as dt
import json
import statistics
import sys

import benchmark as bm
import robustness as rb

DESIGN = ("2018-01", "2023-12")
TEST_FROM = "2024-01"
COSTS = (0, 10, 25, 50, 100, 200)
WINDOW = 36
OOS_LOG = bm.ROOT / "bench_runs" / "oos_log.jsonl"
ALGOS = list(bm.ALGORITHMS) + ["Blend (mom+trend+lowvol)"]


def month_index(keep):
    return {m: i for i, m in enumerate(keep["months"])}


# ----------------------------------------------------------------------------- 1. out of sample, once

def out_of_sample(data, again=False, perms=300):
    previous = [json.loads(x) for x in OOS_LOG.read_text(encoding="utf-8").splitlines() if x.strip()] if OOS_LOG.exists() else []
    if previous and not again:
        return previous[0], len(previous), False
    rows = []
    for uni, keep in data.items():
        for h in (1, 12):
            months = sorted(keep[h])
            design = [i for i in months if DESIGN[0] <= keep["months"][i] and i + h < len(keep["months"]) and keep["months"][i + h] <= DESIGN[1]]
            test = [i for i in months if keep["months"][i] >= TEST_FROM]
            if not design or not test:
                continue
            for a in ALGOS:
                ic1, p1 = rb.perm_p(keep, a, h, design, perms=perms)
                if p1 >= 0.05:
                    continue
                ic2, p2 = rb.perm_p(keep, a, h, test, perms=perms)
                rows.append({"universe": uni, "h": h, "algo": a, "design_ic": round(ic1, 4), "design_p": round(p1, 4),
                             "test_ic": round(ic2, 4), "test_p": round(p2, 4), "test_months": len(test),
                             "held": (ic1 > 0) == (ic2 > 0) and p2 < 0.10})
    entry = {"run": dt.datetime.now().isoformat(timespec="seconds"), "design": list(DESIGN), "test_from": TEST_FROM, "results": rows}
    OOS_LOG.parent.mkdir(exist_ok=True)
    with OOS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry, len(previous) + 1, True


# ----------------------------------------------------------------------------- 2. regimes

def spy_regimes(months):
    """{month: regime} from SPY's month-end closes, each month labelled from that month and the ones before it."""
    spy = bm.fetch("SPY")
    out, vols = {}, {}
    keys = sorted(spy)
    for j, m in enumerate(keys):
        if j < 12:
            continue
        rets = [spy[keys[k]] / spy[keys[k - 1]] - 1 for k in range(j - 5, j + 1)]
        vols[m] = statistics.pstdev(rets)
    med = statistics.median(vols.values()) if vols else 0.0
    for j, m in enumerate(keys):
        if m not in vols:
            continue
        r12 = spy[m] / spy[keys[j - 12]] - 1
        dd = spy[m] / max(spy[keys[k]] for k in range(j - 12, j + 1)) - 1
        if dd <= -0.10:
            out[m] = "drawdown"
        elif r12 >= 0.15:
            out[m] = "bull"
        elif abs(r12) < 0.05 and vols[m] < med:
            out[m] = "sideways"
        else:
            out[m] = "ordinary"
    return {m: out.get(m) for m in months}


def by_regime(keep, regimes, h=1):
    """{regime: {algo: (mean IC, months)}} over the months with that label."""
    out = {}
    months = sorted(keep[h])
    for reg in ("bull", "drawdown", "sideways", "ordinary"):
        ms = [i for i in months if regimes.get(keep["months"][i]) == reg]
        if len(ms) < 6:
            continue
        out[reg] = {a: (statistics.fmean(rb.ic_series(keep, a, h, ms)), len(ms)) for a in ALGOS + ["Random (no skill)"]}
    return out


# ----------------------------------------------------------------------------- 3. costs and break-even

def break_even(keep, algo, months, hi=2000.0):
    """The cost in basis points a side at which the top third's net return equals holding everything; 0 if it
    never beats it, `hi` if it still does at that cost."""
    ew = rb.equal_weight(keep, months)
    gap = lambda b: rb.top_third_net(keep, algo, months, b)[0] - ew
    if gap(0) <= 0:
        return 0.0
    if gap(hi) > 0:
        return hi
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


# ----------------------------------------------------------------------------- 4. walk-forward re-optimisation

def reoptimised(keep, h=1, window=WINDOW):
    """Every January, the algorithm with the best mean IC over the last `window` months whose outcomes were
    known by then; its IC over that year. Returns (picked ICs, all-algorithm ICs, best-in-hindsight ICs, picks)."""
    months = sorted(keep[h])
    years = sorted({keep["months"][i][:4] for i in months})
    picked, everyone, picks = [], [], []
    for y in years:
        start = f"{y}-01"
        known = [i for i in months if i + h < len(keep["months"]) and keep["months"][i + h] < start][-window:]
        test = [i for i in months if keep["months"][i][:4] == y]
        if len(known) < window or not test:
            continue
        best = max(ALGOS, key=lambda a: statistics.fmean(rb.ic_series(keep, a, h, known)))
        picks.append((y, best))
        picked += rb.ic_series(keep, best, h, test)
        everyone += [statistics.fmean(ics) for ics in zip(*(rb.ic_series(keep, a, h, test) for a in ALGOS))]
    test_months = [i for i in months if any(keep["months"][i][:4] == y for y, _ in picks)]
    hindsight = max(ALGOS, key=lambda a: statistics.fmean(rb.ic_series(keep, a, h, test_months))) if test_months else None
    cheat = rb.ic_series(keep, hindsight, h, test_months) if hindsight else []
    return picked, everyone, (hindsight, cheat), picks


# ----------------------------------------------------------------------------- the report

def report(again=False):
    data = {}
    for uni in bm.UNIVERSES:
        keep = {}
        bm.run(uni, keep=keep)
        data[uni] = keep
    out = [f"# Backtest suite, {dt.date.today():%d %b %Y}", ""]

    entry, looks, fresh = out_of_sample(data, again)
    out += [f"## 1. Out of sample: chosen on {entry['design'][0]} to {entry['design'][1]}, tested from {entry['test_from']}", "",
            (f"First run, {entry['run'][:10]}: this is the result of record." if fresh and looks == 1 else
             f"The result of record, from the first run on {entry['run'][:10]}; the test has been looked at {looks} time(s). "
             + ("" if fresh else "Not re-run: `--again` re-tests, and is logged as another look.")), "",
            "| Universe | Horizon | Algorithm | Design IC | p | Test IC | p | Held up? |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in entry["results"]:
        out.append(f"| {r['universe']} | {r['h']} mo | {r['algo']} | {r['design_ic']:+.3f} | {r['design_p']:.3f} | "
                   f"{r['test_ic']:+.3f} | {r['test_p']:.3f} | {'yes' if r['held'] else 'no'} |")
    held = sum(r["held"] for r in entry["results"])
    out += ["", f"{len(entry['results'])} chosen on the design years; {held} held up out of sample (same sign, p < 0.10).", ""]

    out += ["## 2. Regimes (1-month rank IC by market condition)", ""]
    for uni, keep in data.items():
        regimes = spy_regimes(keep["months"])
        reg = by_regime(keep, regimes)
        if not reg:
            continue
        names = list(reg)
        counts = {r: sum(1 for i in keep[1] if regimes.get(keep["months"][i]) == r) for r in ("bull", "drawdown", "sideways", "ordinary")}
        thin = [f"{r} ({c} months)" for r, c in counts.items() if r not in reg]
        out += [f"### {uni}", "", "| Algorithm | " + " | ".join(f"{n} ({reg[n][ALGOS[0]][1]} mo)" for n in names) + " |",
                "|---|" + "---:|" * len(names)]
        for a in ALGOS + ["Random (no skill)"]:
            out.append(f"| {a} | " + " | ".join(f"{reg[n][a][0]:+.3f}" for n in names) + " |")
        out += ([f"Too few months to score: {', '.join(thin)}."] if thin else []) + [""]

    out += ["## 3. Costs: top third, rebalanced monthly, against holding everything", ""]
    for uni, keep in data.items():
        months = sorted(keep[1])
        ew = rb.equal_weight(keep, months)
        out += [f"### {uni}: holding everything returned {bm.pct(ew)} a year", "",
                "| Algorithm | " + " | ".join(f"{c} bp" for c in COSTS) + " | break-even |", "|---|" + "---:|" * (len(COSTS) + 1)]
        for a in ALGOS + ["Random (no skill)"]:
            nets = [rb.top_third_net(keep, a, months, c)[0] - ew for c in COSTS]
            be = break_even(keep, a, months)
            out.append(f"| {a} | " + " | ".join(bm.pct(x) for x in nets) + f" | {'never ahead' if be == 0 else f'{be:.0f} bp'} |")
        out.append("")
    out += ["Each cell: the annual return after costs minus holding everything. Break-even: the cost a side at which "
            "the edge is gone.", ""]

    out += [f"## 4. Walk-forward re-optimisation (each January, the best of the last {WINDOW} months)", "",
            "| Universe | Picked each year | Every algorithm | Best in hindsight (cheats) | Picks |", "|---|---:|---:|---:|---|"]
    for uni, keep in data.items():
        picked, everyone, (best, cheat), picks = reoptimised(keep)
        if not picked:
            continue
        changes = sum(1 for (_, a), (_, b) in zip(picks, picks[1:]) if a != b)
        out.append(f"| {uni} | {statistics.fmean(picked):+.3f} | {statistics.fmean(everyone):+.3f} | "
                   f"{statistics.fmean(cheat):+.3f} ({best}) | {len(picks)} years, {changes} switches |")
    out += ["", "Mean monthly rank IC over the years tested. Re-optimising only helps if last period's winner tends to "
            "win the next one; the hindsight column is the ceiling no real strategy can know in advance.", ""]
    return "\n".join(out)


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    text = report(again="--again" in argv)
    runs = bm.ROOT / "bench_runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"backtest_suite_{dt.datetime.now():%Y-%m-%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
