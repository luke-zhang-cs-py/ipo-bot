"""Calibration: do the forecasts mean what they say?

    python evaluation/calibration.py              all universes, 12-month and 1-month horizons
    python evaluation/calibration.py --ledger     also calibrate the bot's own matured forecasts (forecasts/ledger.jsonl)

benchmark.py asks whether an algorithm ranks stocks in the right order. This asks whether its numbers can be
taken at face value, which is what the bot's memos ask of a reader. Each algorithm is turned into three kinds of
forecast, every month, fitted only on outcomes already known on that date (a 12-month outcome is known 12
months later, never sooner):

- an expected return:  realised return = a + b x (the stock's rank on the signal), refitted every month.
  Calibrated if, regressing what happened on what was forecast, the slope is 1 and the intercept 0.
  "Out-of-sample R2" compares it with the plain running average of every past return; above 0 means it helped.
- a probability that the stock beats the median: the share of past stocks in the same fifth of the ranking
  that did. Scored with the Brier score against the base rate, and the expected calibration error (ECE).
- ranges: 50%, 80% and 90% ranges from the forecast errors seen so far, and an 80% range from a normal
  curve on each stock's own volatility (what a bull/bear spread drawn from volatility assumes).
  A calibrated 80% range holds the outcome 80% of the time.

Then the bot's own rules are applied to each algorithm's 12-month forecast (Overweight at +15% or more,
Underweight at -10% or less) to show what those ratings delivered. With --ledger, the same checks run on the
bot's logged forecasts once they mature, including how often its bull and bear cases came true against the
probabilities it gave them.
"""
import bisect
import datetime as dt
import math
import random
import statistics
import sys

import benchmark as bm

HORIZONS = (12, 1)
BINS = 5                    # the ranking is cut into fifths for the probability forecast
MIN_TRAIN_DATES = 36        # months of matured outcomes before the first forecast
RANGES = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
Z80 = 1.2816
OVERWEIGHT, UNDERWEIGHT = 0.15, -0.10      # the bot's rating thresholds (prompts/system_prompt.md)
BLEND, RANDOM = "Blend (mom+trend+lowvol)", "Random (no skill)"


def positions(scores):
    """Each score's place in the ranking, from -0.5 (worst) to +0.5 (best)."""
    n = len(scores)
    return [(r - 0.5) / n - 0.5 for r in bm.ranks(scores)]


def quantile(sorted_xs, q):
    if not sorted_xs:
        return 0.0
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


class Line:
    """y = a + b x by least squares, updated one point at a time."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.sxy = 0.0

    def add(self, x, y):
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.sxy += x * y

    def coef(self):
        if self.n < 2:
            return (self.sy / self.n if self.n else 0.0), 0.0
        var = self.sxx - self.sx * self.sx / self.n
        b = (self.sxy - self.sx * self.sy / self.n) / var if var > 1e-12 else 0.0
        return self.sy / self.n - b * self.sx / self.n, b


def fit_line(xs, ys):
    line = Line()
    for x, y in zip(xs, ys):
        line.add(x, y)
    return line.coef()


# ----------------------------------------------------------------------------- the walk-forward forecasts

def prepare(months, px, symbols, h, seed=7):
    """Every algorithm's ranking, each stock's volatility, and what happened, at every month-end."""
    rng = random.Random(seed)
    idx = list(range(bm.LOOKBACK, len(months) - h))
    pos, vol, real, beat = {}, {}, {}, {}
    for i in idx:
        raw = {a: [f(px[s], i) for s in symbols] for a, f in bm.ALGORITHMS.items()}
        raw[BLEND] = bm.blend(raw)
        raw[RANDOM] = [rng.random() for _ in symbols]
        pos[i] = {a: positions(v) for a, v in raw.items()}
        vol[i] = [statistics.pstdev([px[s][k] / px[s][k - 1] - 1 for k in range(i - 35, i + 1)]) for s in symbols]
        real[i] = [px[s][i + h] / px[s][i] - 1 for s in symbols]
        med = statistics.median(real[i])
        beat[i] = [1 if y > med else 0 for y in real[i]]
    return {"months": months, "h": h, "idx": idx, "pos": pos, "vol": vol, "real": real, "beat": beat}


def walk(prep, a, perm=None, ranges=True):
    """One algorithm's forecasts, month by month, each fitted only on outcomes known by then.
    perm: give the algorithm's scores to other stocks (the same swap every month), for the permutation test.
    ranges=False skips the ranges, which the permutation test does not score."""
    months, h, idx = prep["months"], prep["h"], prep["idx"]
    real, beat, vol = prep["real"], prep["beat"], prep["vol"]
    pos = {i: ([prep["pos"][i][a][k] for k in perm] if perm else prep["pos"][i][a]) for i in idx}
    line, naive = Line(), Line()
    bins = [[0, 0] for _ in range(BINS)]
    base = [0, 0]
    resid = []
    recs, trained = [], 0
    nxt = 0                                   # the next month-end whose outcome is not yet in the training set
    for i in idx:
        while nxt < len(idx) and idx[nxt] + h <= i:     # outcomes known by month i, and only those
            j = idx[nxt]
            ca, cb = line.coef()
            for x, y, w in zip(pos[j], real[j], beat[j]):
                if ranges:
                    bisect.insort(resid, y - (ca + cb * x))
                line.add(x, y)
                naive.add(0.0, y)
                b = min(BINS - 1, int((x + 0.5) * BINS))
                bins[b][0] += w
                bins[b][1] += 1
                base[0] += w
                base[1] += 1
            nxt += 1
            trained += 1
        if trained < MIN_TRAIN_DATES:
            continue
        ca, cb = line.coef()
        mean_all = naive.sy / naive.n
        clim = (base[0] + 1) / (base[1] + 2)
        for k, x in enumerate(pos[i]):
            f = ca + cb * x
            b = min(BINS - 1, int((x + 0.5) * BINS))
            rec = {"month": months[i], "f": f, "naive": mean_all, "y": real[i][k],
                   "p": (bins[b][0] + 1) / (bins[b][1] + 2), "clim": clim, "beat": beat[i][k]}
            if ranges:
                sd = vol[i][k] * math.sqrt(h)
                rec["ranges"] = {c: (f + quantile(resid, lo), f + quantile(resid, hi)) for c, (lo, hi) in RANGES.items()}
                rec["normal80"] = (f - Z80 * sd, f + Z80 * sd)
            recs.append(rec)
    demean(recs)
    return recs


def forecasts(months, px, symbols, h, seed=7):
    """For every algorithm: one record per (month-end, stock) with the forecasts made then and what happened."""
    prep = prepare(months, px, symbols, h, seed)
    return {a: walk(prep, a) for a in prep["pos"][prep["idx"][0]]}


def demean(recs):
    """Add each forecast and outcome relative to that month's average: the stock-against-stock part.
    A forecast mixes a market call (the same for every stock that month) with a tilt between stocks; the
    market part moves with the years and would swamp the tilt if the two were scored together."""
    by_month = {}
    for r in recs:
        by_month.setdefault(r["month"], []).append(r)
    for rows in by_month.values():
        mf = statistics.fmean(r["f"] for r in rows)
        my = statistics.fmean(r["y"] for r in rows)
        for r in rows:
            r["fx"], r["yx"] = r["f"] - mf, r["y"] - my


NULL_SEEDS = 40
NULL_KEYS = ("slope", "r2_os", "bss", "ece")


def core_scores(recs):
    """The scores the permutation test compares (no ranges): slope, out-of-sample R2, Brier and its skill, ECE."""
    _, slope = fit_line([r["fx"] for r in recs], [r["yx"] for r in recs])
    sse = sum((r["y"] - r["f"]) ** 2 for r in recs)
    sse0 = sum((r["y"] - r["naive"]) ** 2 for r in recs)
    brier = statistics.fmean((r["p"] - r["beat"]) ** 2 for r in recs)
    brier0 = statistics.fmean((r["clim"] - r["beat"]) ** 2 for r in recs)
    return {"slope": slope, "r2_os": 1 - sse / sse0 if sse0 else float("nan"), "brier": brier,
            "bss": 1 - brier / brier0 if brier0 else float("nan"),
            "ece": ece([r["p"] for r in recs], [r["beat"] for r in recs])}


def null_bands(prep, seeds=NULL_SEEDS):
    """Per algorithm, what luck alone gives: its own scores handed to the wrong stocks (one fixed swap per run),
    which keeps how slowly the signal changes and breaks only its link to the returns. The middle 95%."""
    n = len(prep["real"][prep["idx"][0]])
    out = {}
    for a in prep["pos"][prep["idx"][0]]:
        rng = random.Random(500)
        runs = []
        for _ in range(seeds):
            perm = list(range(n))
            rng.shuffle(perm)
            runs.append(core_scores(walk(prep, a, perm=perm, ranges=False)))
        out[a] = {k: (quantile(sorted(r[k] for r in runs), 0.025), quantile(sorted(r[k] for r in runs), 0.975))
                  for k in NULL_KEYS}
    return out


def outside(value, band, better_high=True):
    """'*' when a score is better than luck would give 97.5% of the time, '!' when worse."""
    lo, hi = band
    if value > hi:
        return "*" if better_high else "!"
    if value < lo:
        return "!" if better_high else "*"
    return ""


# ----------------------------------------------------------------------------- scoring

def ece(ps, ws, width=0.05):
    groups = {}
    for p, w in zip(ps, ws):
        groups.setdefault(min(int(p / width), int(1 / width) - 1), []).append((p, w))
    n = len(ps)
    return sum(len(g) / n * abs(statistics.fmean(p for p, _ in g) - statistics.fmean(w for _, w in g))
               for g in groups.values())


def score(recs):
    """core_scores, plus how often each range held the outcome and the average forecast and outcome."""
    cover = {c: statistics.fmean(lo <= r["y"] <= hi for r in recs for lo, hi in [r["ranges"][c]]) for c in RANGES}
    return {
        "n": len(recs), **core_scores(recs), "cover": cover,
        "normal80": statistics.fmean(r["normal80"][0] <= r["y"] <= r["normal80"][1] for r in recs),
        "mean_f": statistics.fmean(r["f"] for r in recs), "mean_y": statistics.fmean(r["y"] for r in recs),
    }


def reliability(recs, groups=5):
    """Forecasts against the month's average, cut into fifths: the average forecast and outcome in each."""
    rs = sorted(recs, key=lambda r: r["fx"])
    size = len(rs) / groups
    return [(statistics.fmean(r["fx"] for r in part), statistics.fmean(r["yx"] for r in part))
            for g in range(groups) for part in [rs[round(g * size):round((g + 1) * size)]]]


def ratings(recs):
    out = {}
    for name, keep in (("Overweight", lambda f: f >= OVERWEIGHT), ("Equal-weight", lambda f: UNDERWEIGHT < f < OVERWEIGHT),
                       ("Underweight", lambda f: f <= UNDERWEIGHT)):
        part = [r for r in recs if keep(r["f"])]
        out[name] = None if not part else {
            "n": len(part), "share": len(part) / len(recs), "f": statistics.fmean(r["f"] for r in part),
            "y": statistics.fmean(r["y"] for r in part), "up": statistics.fmean(r["y"] > 0 for r in part),
            "beat": statistics.fmean(r["beat"] for r in part)}
    return out


def block_slopes(recs):
    years = sorted({int(r["month"][:4]) for r in recs})
    out = []
    for y0 in range(years[0], years[-1] + 1, bm.BLOCK_YEARS):
        part = [r for r in recs if y0 <= int(r["month"][:4]) < y0 + bm.BLOCK_YEARS]
        label = f"{y0}-{str(min(y0 + bm.BLOCK_YEARS - 1, years[-1]))[2:]}"
        out.append((label, fit_line([r["fx"] for r in part], [r["yx"] for r in part])[1] if len(part) > 10 else float("nan")))
    return out


def market_level(recs):
    """The market call every algorithm shares (the running average of past returns), month by month,
    against the average outcome that month."""
    by_month = {}
    for r in recs:
        by_month.setdefault(r["month"], []).append(r)
    f = [statistics.fmean(r["naive"] for r in rows) for rows in by_month.values()]
    y = [statistics.fmean(r["y"] for r in rows) for rows in by_month.values()]
    _, slope = fit_line(f, y)
    hits = statistics.fmean((a > 0) == (b > 0) for a, b in zip(f, y))
    return {"f": statistics.fmean(f), "y": statistics.fmean(y), "slope": slope, "sign": hits, "n": len(f)}


# ----------------------------------------------------------------------------- the bot's own forecasts

def ledger_calibration(refresh=False):
    if not bm.LEDGER.exists():
        return ("No bot forecasts logged yet. When there are, this section shows the same checks for the bot: "
                "expected return against what happened, and how often its bull and bear cases came true "
                "against the probabilities it gave them.")
    rows = bm.ledger_rows()
    done = []
    for r in rows:
        _, due, end = bm.maturity(r)
        if not bm.is_due(due):
            continue
        prices = bm.fetch(r["symbol"], refresh)
        if end not in prices:
            continue
        # The logged price and scenario values are as traded; the closes are adjusted. Put the forecast in the
        # closes' terms (bm.entry_factor) so a split during the horizon is not a crash past the bear case.
        factor = bm.entry_factor(r["symbol"], r.get("price_date") or r["date"])
        if not factor:
            continue
        scaled = {k: r[k] * factor for k in ("price", "bull_value", "bear_value") if r.get(k) is not None}
        done.append({**r, **scaled, "realised": prices[end] / scaled["price"] - 1, "end_price": prices[end]})
    lines = [f"{len(rows)} logged, {len(done)} matured and priced."]
    if len(done) >= 2:
        f = [d["expected_return_pct"] / 100 for d in done]
        y = [d["realised"] for d in done]
        a, b = fit_line(f, y)
        lines.append(f"Expected return {100 * statistics.fmean(f):+.1f}% on average, realised {100 * statistics.fmean(y):+.1f}%; "
                     f"slope {b:.2f}, intercept {100 * a:+.1f}% (calibrated: slope 1, intercept 0). n = {len(done)}"
                     + ("; too few to read much into." if len(done) < 30 else "."))
    for case, worse in (("bear", lambda d: d["end_price"] <= d["bear_value"]), ("bull", lambda d: d["end_price"] >= d["bull_value"])):
        part = [d for d in done if d.get(f"{case}_value") and d.get(f"{case}_prob") is not None]
        if part:
            lines.append(f"{case.title()} case or beyond: happened {100 * statistics.fmean(worse(d) for d in part):.0f}% "
                         f"of the time; the bot gave it {statistics.fmean(d[f'{case}_prob'] for d in part):.0f}% on average "
                         f"(n = {len(part)}).")
    for name in ("Overweight", "Equal-weight", "Underweight"):
        part = [d for d in done if d["rating"] == name]
        if part:
            lines.append(f"{name}: {len(part)} calls, expected {statistics.fmean(d['expected_return_pct'] for d in part):+.1f}%, "
                         f"realised {100 * statistics.fmean(d['realised'] for d in part):+.1f}%, "
                         f"up {100 * statistics.fmean(d['realised'] > 0 for d in part):.0f}% of the time.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- report

pct = bm.pct


def report(refresh=False, ledger=False):
    out = [f"# Calibration test, {dt.date.today():%d %b %Y}", "",
           "Each forecast is fitted only on outcomes known at the time it is made. Ideal values: slope 1.00, "
           "intercept 0, out-of-sample R2 above 0, Brier skill above 0, ECE near 0, each range holding the outcome "
           "as often as its name says.", ""]
    for uni, symbols in bm.UNIVERSES.items():
        months, px = bm.panel(symbols, refresh)
        out += [f"## {uni} ({len(symbols)} names)", ""]
        for h in HORIZONS:
            prep = prepare(months, px, symbols, h)
            # no month-ends at all (LOOKBACK + h months or fewer), or too few to fit even the first forecast
            per = {a: walk(prep, a) for a in prep["pos"][prep["idx"][0]]} if prep["idx"] else {}
            if not per.get(RANDOM):
                out += [f"### {h}-month forecasts: too few months ({MIN_TRAIN_DATES} matured month-ends come first)", ""]
                continue
            scores = {a: score(r) for a, r in per.items()}
            bands = null_bands(prep)
            span = f"{per[RANDOM][0]['month']} to {per[RANDOM][-1]['month']}"
            n_tests = len({r["month"] for r in per[RANDOM]})
            out += [f"### {h}-month forecasts ({n_tests} consecutive month-ends, {span}; "
                    f"{scores[RANDOM]['n']:,} stock forecasts per algorithm)", "",
                    "| Algorithm | Slope, stock vs stock | OOS R2 | OOS R2 from luck | Brier skill | ECE | 50% range | 80% range | 90% range | 80% normal-vol |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for a, s in sorted(scores.items(), key=lambda kv: -kv[1]["r2_os"]):
                c, band = s["cover"], bands[a]
                out.append(f"| {a} | {s['slope']:.2f}{outside(s['slope'], band['slope'])} | "
                           f"{pct(s['r2_os'], 2)}{outside(s['r2_os'], band['r2_os'])} | "
                           f"{pct(band['r2_os'][0], 2)} to {pct(band['r2_os'][1], 2)} | "
                           f"{pct(s['bss'], 2)}{outside(s['bss'], band['bss'])} | "
                           f"{100 * s['ece']:.1f} pts{outside(s['ece'], band['ece'], better_high=False)} | "
                           f"{100 * c[50]:.1f}% | {100 * c[80]:.1f}% | {100 * c[90]:.1f}% | {100 * s['normal80']:.1f}% |")
            m = market_level(per[RANDOM])
            out += ["", f"Market level (shared by every algorithm: the running average of past returns): forecast "
                    f"{pct(m['f'])} on average, realised {pct(m['y'])}; slope {m['slope']:.2f}; direction right in "
                    f"{100 * m['sign']:.0f}% of {m['n']} months.", "",
                    "Stock-against-stock forecast and outcome (each relative to that month's average), "
                    "forecasts cut into fifths, lowest to highest:", "",
                    "| Algorithm | " + " | ".join(f"Fifth {g + 1}" for g in range(5)) + " |", "|---|" + "---:|" * 5]
            for a in sorted(per, key=lambda a: -scores[a]["r2_os"]):
                out.append(f"| {a} | " + " | ".join(f"{pct(f, 1)} -> {pct(y, 1)}" for f, y in reliability(per[a])) + " |")
            out += ["", "Stock-against-stock calibration slope in consecutive blocks (1.00 = the gaps it forecast between stocks came true in full):", ""]
            labels = [l for l, _ in block_slopes(per[RANDOM])]
            out += ["| Algorithm | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
            for a in sorted(per, key=lambda a: -scores[a]["r2_os"]):
                out.append(f"| {a} | " + " | ".join("n/a" if math.isnan(s) else f"{s:.2f}" for _, s in block_slopes(per[a])) + " |")
            if h == 12:
                out += ["", "The bot's rating rules applied to each algorithm's 12-month forecast "
                        "(Overweight >= +15%, Underweight <= -10%):", "",
                        "| Algorithm | Rating | Share of calls | Forecast | Realised | Up | Beat median |",
                        "|---|---|---:|---:|---:|---:|---:|"]
                for a in sorted(per, key=lambda a: -scores[a]["r2_os"]):
                    for name, r in ratings(per[a]).items():
                        if r:
                            out.append(f"| {a} | {name} | {100 * r['share']:.0f}% | {pct(r['f'])} | {pct(r['y'])} | "
                                       f"{100 * r['up']:.0f}% | {100 * r['beat']:.0f}% |")
            out.append("")
    out += ["## How to read this", "",
            "- **Slope, stock vs stock**: regress each stock's outcome against the month's average on its forecast "
            "against the month's average. 1 = the gaps forecast between stocks came true in full; between 0 and 1 = "
            "right direction, too confident; 0 or below = the forecast gaps say nothing or point the wrong way. "
            "Random should sit near 0, with noise.",
            "- **Marks**: `*` = better than 97.5% of permutation runs; `!` = worse than 97.5% of them. Each "
            "algorithm gets its own luck baseline: its scores handed to the wrong stocks (one fixed swap per run, "
            "40 runs), which keeps how slowly the signal changes. Anything "
            "unmarked is within what luck produces on this universe and period.",
            "- **Market level**: the part of every forecast that is the same for all stocks that month. A slope "
            "below 0 means good past years were followed by weaker ones.",
            "- **OOS R2** against the running average of all past returns. Stock returns are mostly noise: "
            "a fraction of a percent above 0 is good; below 0 means the signal made forecasts worse than ignoring it.",
            "- **Brier skill** of the beat-the-median probability against the base rate; **ECE** is the average gap, "
            "in percentage points, between the probability given and how often it happened.",
            "- **Ranges**: a 12-month range from past errors overlaps with the outcomes it is tested on, so expect "
            "some drift in a regime change; the normal-vol range shows whether volatility alone sizes bull and bear cases well.",
            "- **Ratings**: what the bot's thresholds would have delivered on top of each algorithm. An Overweight "
            "that averages +20% forecast and +12% realised is over-confident by 8 points.", ""]
    if ledger:
        out += ["## The bot's own forecasts", "", ledger_calibration(refresh), ""]
    return "\n".join(out)


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    text = report(refresh="--refresh" in argv, ledger="--ledger" in argv)
    runs = bm.ROOT / "bench_runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"calibration_{dt.datetime.now():%Y-%m-%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
