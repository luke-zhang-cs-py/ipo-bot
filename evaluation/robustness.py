"""Does anything survive? Three harder tests on the benchmark's algorithms.

    python evaluation/robustness.py

1. Multiple testing. Six algorithms x three universes x two horizons is 36 tests: at a 5% bar, about two
   pass by luck alone. Each gets a permutation p-value (its own scores handed to the wrong stocks, 1,000
   times), and the Benjamini-Hochberg procedure keeps the false-discovery rate across all 36 at 10%.
2. Holdout. Pick what worked in the first half (2009-2017) as if that were all you knew, then check it in
   the second half (2018 on), which played no part in the choice.
3. Costs. Each algorithm's top third, rebalanced monthly, after paying 0, 10, 25 and 50 basis points per
   side on the shares that change, against simply holding the whole universe.

The random control is reported but kept out of the 36: it is the check that the machinery is fair.
"""
import datetime as dt
import random
import statistics
import sys

import benchmark as bm

PERMS = 1000
FDR = 0.10
SPLIT = "2018-01"
COSTS_BPS = (0, 10, 25, 50)


def ic_series(keep, algo, h, months, perm=None):
    return [bm.spearman([keep["raw"][i][algo][k] for k in perm] if perm else keep["raw"][i][algo], keep[h][i])
            for i in months]


def perm_p(keep, algo, h, months, perms=PERMS, seed=1):
    """Two-sided p-value of the mean rank IC against the algorithm's own scores handed to the wrong stocks."""
    obs = statistics.fmean(ic_series(keep, algo, h, months))
    n = len(keep[h][months[0]])
    rng = random.Random(seed)
    null = []
    for _ in range(perms):
        p = list(range(n))
        rng.shuffle(p)
        null.append(statistics.fmean(ic_series(keep, algo, h, months, p)))
    centre = statistics.median(null)
    extreme = sum(abs(x - centre) >= abs(obs - centre) for x in null)
    return obs, (extreme + 1) / (perms + 1)


def benjamini_hochberg(pvals, q=FDR):
    """Which tests are discoveries at false-discovery rate q."""
    order = sorted(range(len(pvals)), key=lambda k: pvals[k])
    m, cut = len(pvals), -1
    for rank, k in enumerate(order, 1):
        if pvals[k] <= q * rank / m:
            cut = rank
    keep = set(order[:cut]) if cut > 0 else set()
    return [k in keep for k in range(len(pvals))]


def top_third_net(keep, algo, months, bps):
    """Annual return of holding the algorithm's top third for a month at a time, after costs, and turnover."""
    n = len(keep[1][months[0]])
    third = n // 3
    held, growth, turns = set(), 1.0, []
    for i in months:
        sc = keep["raw"][i][algo]
        top = set(sorted(range(n), key=lambda k: sc[k])[-third:])
        turn = len(top - held) / third if held else 1.0         # share of the portfolio replaced
        turns.append(turn)
        cost = (2 * turn if held else 1.0) * bps / 10_000      # sell the old, buy the new (the first month only buys)
        growth *= (1 + statistics.fmean(keep[1][i][k] for k in top)) * (1 - cost)
        held = top
    years = len(months) / 12
    return growth ** (1 / years) - 1, statistics.fmean(turns[1:]) if len(turns) > 1 else float("nan")


def equal_weight(keep, months):
    growth = 1.0
    for i in months:
        growth *= 1 + statistics.fmean(keep[1][i])
    return growth ** (1 / (len(months) / 12)) - 1


pct = bm.pct


def report():
    out = [f"# Robustness tests, {dt.date.today():%d %b %Y}", ""]
    algos = list(bm.ALGORITHMS) + ["Blend (mom+trend+lowvol)"]
    tests, controls, data = [], [], {}
    for uni in bm.UNIVERSES:
        keep = {}
        bm.run(uni, keep=keep)
        data[uni] = keep
        for h in (12, 1):
            months = sorted(keep[h])
            for a in algos + ["Random (no skill)"]:
                ic, p = perm_p(keep, a, h, months)
                row = {"uni": uni, "h": h, "algo": a, "ic": ic, "p": p}
                (controls if a.startswith("Random") else tests).append(row)
    found = benjamini_hochberg([t["p"] for t in tests])
    out += [f"## 1. Multiple testing: {len(tests)} tests, false-discovery rate {int(FDR * 100)}%", "",
            f"Permutation p-values ({PERMS:,} runs each). A naive 5% bar would pass "
            f"{sum(t['p'] < 0.05 for t in tests)} of {len(tests)}; luck alone passes about {0.05 * len(tests):.0f}.", "",
            "| Universe | Horizon | Algorithm | Rank IC | p-value | Passes 5% alone | Survives the correction |",
            "|---|---:|---|---:|---:|---|---|"]
    for t, ok in sorted(zip(tests, found), key=lambda x: x[0]["p"]):
        if t["p"] < 0.2 or ok:
            out.append(f"| {t['uni']} | {t['h']} mo | {t['algo']} | {t['ic']:+.3f} | {t['p']:.3f} | "
                       f"{'yes' if t['p'] < 0.05 else 'no'} | {'**yes**' if ok else 'no'} |")
    out += ["", f"(Tests with p >= 0.20 are left out of the table.) Survivors: {sum(found)}.", "",
            "Random control, which should not pass: " + "; ".join(
                f"{c['uni']} {c['h']} mo p = {c['p']:.3f}" for c in controls), ""]

    out += [f"## 2. Holdout: chosen on data before {SPLIT}, checked on data from {SPLIT}", "",
            "| Universe | Horizon | Algorithm | First half IC | p | Second half IC | p | Held up? |",
            "|---|---:|---|---:|---:|---:|---:|---|"]
    chosen = held = 0
    for uni, keep in data.items():
        for h in (12, 1):
            months = sorted(keep[h])
            first = [i for i in months if month_of(keep, i + h) < SPLIT]       # outcome known before the split
            second = [i for i in months if month_of(keep, i) >= SPLIT]
            for a in algos:
                ic1, p1 = perm_p(keep, a, h, first, perms=300)
                if p1 >= 0.05:
                    continue
                ic2, p2 = perm_p(keep, a, h, second, perms=300)
                chosen += 1
                ok = (ic1 > 0) == (ic2 > 0) and p2 < 0.10
                held += ok
                out.append(f"| {uni} | {h} mo | {a} | {ic1:+.3f} | {p1:.3f} | {ic2:+.3f} | {p2:.3f} | "
                           f"{'yes' if ok else 'same sign, not significant' if (ic1 > 0) == (ic2 > 0) else 'no, reversed'} |")
    out += ["", f"{chosen} picked on the first half; {held} held up in the second (same direction, p < 0.10). "
            "A first-half outcome window never reaches into the second half.", ""]

    out += ["## 3. After trading costs: the top third, rebalanced monthly", ""]
    for uni, keep in data.items():
        months = sorted(keep[1])
        ew = equal_weight(keep, months)
        out += [f"### {uni}: holding everything returned {pct(ew)} a year", "",
                "| Algorithm | Turnover a month | " + " | ".join(f"{b} bp" for b in COSTS_BPS) + " |",
                "|---|---:|" + "---:|" * len(COSTS_BPS)]
        for a in algos + ["Random (no skill)"]:
            nets = [top_third_net(keep, a, months, b) for b in COSTS_BPS]
            out.append(f"| {a} | {100 * nets[0][1]:.0f}% | " +
                       " | ".join(f"{pct(r)} ({pct(r - ew)})" for r, _ in nets) + " |")
        out.append("")
    out += ["Each cell: annual return after costs (in brackets: against holding everything). Basis points are per "
            "side, on the part of the portfolio replaced each month. Same survivor-universe caveat as the benchmark.", ""]
    return "\n".join(out)


def month_of(keep, i):
    return keep["months"][i]


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    text = report()
    runs = bm.ROOT / "bench_runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"robustness_{dt.datetime.now():%Y-%m-%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
