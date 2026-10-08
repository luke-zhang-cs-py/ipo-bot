"""IPO evaluation: a pre-listing model of first-day "pops", put through the checks a real IPO strategy needs.

    python evaluation/ipo_eval.py            develop: walk-forward 2018-2023, every check; the holdout stays sealed
    python evaluation/ipo_eval.py --final    score the sealed holdout (IPOs listed from 2024 on): once, then recorded
    python evaluation/ipo_eval.py --final --again    score it again anyway (logged as another look)

The bot itself can't be scored on these IPOs: Claude has read how 2015-2024 listings went, so any backtest
of it here would be leakage. It is scored forward (track.py, and testkit/kit.py backtest after its training
cutoff). This model is the bar it has to clear, the way benchmark.py's simple algorithms are for stocks:
a logistic regression on what was public before each listing, refitted every year on earlier years only.

The checks, in the order of the brief:
  1. walk-forward by year, with and without re-choosing the settings each year; a sealed holdout scored once;
     regimes by calendar (bull 2020-21, bear and high rates 2022-23) and by SPY on each listing day
  2. a leakage audit (dates, a perturbation test, and a canary feature it must catch); feature sensitivity
     (permutation and drop-one, beside a pure-noise feature); class imbalance (PR-AUC, F1, not accuracy)
  3. slippage of 0.5-2% on buying at the open, and the friction that wipes out the edge; the winner's curse
     when buying at the offer; and a live-feed
     stress test of the parsers and the model at market open
Data: ipo_data.py (SEC prospectuses; Yahoo prices, cached locally). Results go to bench_runs/.
"""
import datetime as dt
import json
import math
import pathlib
import random
import statistics
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import ipo_data  # noqa: E402

POP = 0.20                      # a "pop": first-day close 20% or more above the offer
POP_TOL = 1e-9                  # 12 / 10 - 1 is 0.19999999999999996 in floating point: still a pop (see popped)
TEST_YEARS = range(2018, 2024)  # walk-forward test years; each is predicted by a model fitted on the years before
HOLDOUT_FROM = "2024-01-01"     # IPOs listed from here on are sealed until --final, and then scored once
REGIMES = {"normal 2018-19": (2018, 2019), "bull 2020-21": (2020, 2021), "bear, high rates 2022-23": (2022, 2023)}
L2 = 1.0                        # ridge penalty for the fixed model, set in advance: nothing is tuned on test years
L2_GRID = (0.1, 1.0, 10.0, 100.0)   # what the re-optimised walk-forward chooses from, on training years only
WINDOWS = (None, 3)             # all earlier years, or only the last three
FRICTION = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03)   # slippage, spread and fees together, each way
HOLDOUT_LOG = ROOT / "bench_runs" / "ipo_holdout_log.jsonl"
LIVE_BYTES = 150_000            # what the live scorer reads of a filing: the cover page, where the offer and range are
COVERED = 0.6                   # a year counts as well covered when this share of its IPOs still has prices
BOOTSTRAP = 1000                # resamples for each confidence interval
CURSE_K = (2, 5, 10)            # how steeply a retail allocation shrinks as a deal's demand grows
OFFER_BAND = (0.5, 2.0)         # an offer below half the filed range's low or above twice its high is a misread
FEATURES = ["revision", "above_range", "below_range", "no_range", "log_proceeds", "log_price", "bank_share",
            "tech", "biotech", "finance", "foreign", "spy_20d", "heat_30d", "deals_30d"]


# ----------------------------------------------------------------------------- the data, as the model sees it

class Sealed(Exception):
    """Raised when development code asks for the holdout."""


def priced(r):
    """An IPO with an offer price and a first day's open and close."""
    return bool(r.get("offer_price") and r.get("prices") and r["prices"].get("open") and r["prices"].get("close"))


def offer_off_range(r):
    """True when the offer is far outside the price range filed before it (OFFER_BAND): a parser's misread, caught
    from pre-listing facts alone, unlike ipo_data's suspect flag, which reads the first day's open."""
    rng, offer = r.get("range"), r.get("offer_price")
    return bool(rng and offer and not OFFER_BAND[0] * rng[0] <= offer <= OFFER_BAND[1] * rng[1])


def excluded(rows):
    """The priced IPOs usable() leaves out, by reason. Suspect prices (an open outside 0.5-4x the offer) are judged
    on the listing day itself, so the report states how many there are and how they traded: the selection is
    visible rather than silent."""
    p = [r for r in rows if priced(r)]
    return {"suspect prices": [r for r in p if r["prices"].get("suspect")],
            "offer outside the filed range": [r for r in p if not r["prices"].get("suspect") and offer_off_range(r)]}


def usable(rows):
    """IPOs with an offer price and trustworthy first-day prices, whose offer agrees with the filed range."""
    return [r for r in rows if priced(r) and not r["prices"].get("suspect") and not offer_off_range(r)]


def split(rows, final=False):
    """(development rows, holdout rows). The holdout is empty unless final=True: nothing built while
    developing — features, the penalty, thresholds — can have looked at it."""
    dev = [r for r in rows if r["prices"]["listing_date"] < HOLDOUT_FROM]
    hold = [r for r in rows if r["prices"]["listing_date"] >= HOLDOUT_FROM]
    return dev, (hold if final else [])


def first_day(r):
    return r["prices"]["close"] / r["offer_price"] - 1


def popped(first_day_return):
    """True when a first-day return is a pop: POP or more, with a hair of tolerance so that a close of exactly
    20% over the offer counts despite floating point. The one definition; the harness uses it too."""
    return first_day_return >= POP - POP_TOL


def sic_group(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return ""
    if 2833 <= s <= 2836 or s in (8731, 8071) or 3841 <= s <= 3845:
        return "biotech"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379 or s == 4899:
        return "tech"
    if 6000 <= s <= 6799:
        return "finance"
    return ""


def spy_returns(spy):
    """SPY's 20-trading-day return ending the day before each date: {date: return}, from {date: close}."""
    days = sorted(spy)
    out = {}
    for i in range(21, len(days)):
        out[days[i]] = spy[days[i - 1]] / spy[days[i - 21]] - 1
    return out


def spy_regime(spy, day):
    """The market on the day before a listing, from SPY's closes up to then: "bull" (up 15%+ over a year, less
    than 10% off its high), "drawdown" (10%+ below its 52-week high), "sideways" (within 5% of a year ago, with
    a quiet last quarter), else "ordinary"."""
    days = [d for d in sorted(spy) if d < day][-253:]
    if len(days) < 253:
        return None
    closes = [spy[d] for d in days]
    r = closes[-1] / closes[0] - 1
    dd = closes[-1] / max(closes) - 1
    rets = [b / a - 1 for a, b in zip(closes[-64:], closes[-63:])]
    if dd <= -0.10:
        return "drawdown"
    if r >= 0.15:
        return "bull"
    if abs(r) < 0.05 and statistics.pstdev(rets) * math.sqrt(252) < 0.15:
        return "sideways"
    return "ordinary"


def market_features(rows, spy20):
    """Per IPO, what the market had shown before its listing day: SPY's last 20 days, and the IPOs that listed
    in the 30 days before (their average first day, and how many). Only listings strictly earlier count."""
    by_day = sorted(rows, key=lambda r: r["prices"]["listing_date"])
    out = {}
    for r in by_day:
        day = r["prices"]["listing_date"]
        start = (dt.date.fromisoformat(day) - dt.timedelta(days=30)).isoformat()
        prior = [first_day(x) for x in by_day if start <= x["prices"]["listing_date"] < day]
        out[r["adsh"]] = {"spy_20d": spy20.get(day, 0.0), "heat_30d": statistics.fmean(prior) if prior else 0.0,
                          "deals_30d": math.log1p(len(prior))}
    return out


def bank_shares(train):
    """Each lead bank's share of the training IPOs: a league table built only from the years trained on."""
    counts = {}
    for r in train:
        if r.get("lead_bank"):
            counts[r["lead_bank"]] = counts.get(r["lead_bank"], 0) + 1
    n = max(1, len(train))
    return {b: c / n for b, c in counts.items()}


def features(r, market, banks):
    """The model's inputs for one IPO, from its prospectus, the market before it, and the training league table."""
    offer, rng = r["offer_price"], r.get("range")
    listed = (r.get("prices") or {}).get("listing_date")
    if rng and listed and r.get("range_date") and r["range_date"] >= listed:
        rng = None                                 # a filing from the listing day or later was not public before it
    mid = (rng[0] + rng[1]) / 2 if rng else None
    proceeds = offer * r["shares"] if r.get("shares") else None
    g = sic_group(r.get("sic"))
    m = market.get(r["adsh"], {"spy_20d": 0.0, "heat_30d": 0.0, "deals_30d": 0.0})
    return {
        "revision": (offer / mid - 1) if mid else 0.0,
        "above_range": float(bool(rng) and offer > rng[1] + 1e-9),
        "below_range": float(bool(rng) and offer < rng[0] - 1e-9),
        "no_range": float(not rng),
        "log_proceeds": math.log(proceeds) if proceeds else math.log(50e6),
        "log_price": math.log(offer),
        "bank_share": banks.get(r.get("lead_bank"), 0.0),
        "tech": float(g == "tech"), "biotech": float(g == "biotech"), "finance": float(g == "finance"),
        "foreign": float(bool(r.get("foreign"))),
        **m,
    }


# ----------------------------------------------------------------------------- a small logistic regression

def _solve(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting (a is small and positive definite here)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(m[i][c]))
        m[c], m[p] = m[p], m[c]
        for i in range(c + 1, n):
            f = m[i][c] / m[c][c]
            for j in range(c, n + 1):
                m[i][j] -= f * m[c][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


class Logit:
    """L2-penalised logistic regression fitted by Newton's method on standardised inputs."""

    def __init__(self, names, l2=L2):
        self.names, self.l2 = names, l2

    def _x(self, f):
        return [1.0] + [(f[k] - self.mu[k]) / self.sd[k] for k in self.names]

    def fit(self, feats, ys, iters=25):
        self.mu = {k: statistics.fmean(f[k] for f in feats) for k in self.names}
        self.sd = {k: (statistics.pstdev([f[k] for f in feats]) or 1.0) for k in self.names}
        xs = [self._x(f) for f in feats]
        d = len(self.names) + 1
        w = [0.0] * d
        for _ in range(iters):
            g = [0.0] * d
            h = [[0.0] * d for _ in range(d)]
            for x, y in zip(xs, ys):
                p = 1 / (1 + math.exp(-max(-35.0, min(35.0, sum(a * b for a, b in zip(w, x))))))
                for i in range(d):
                    g[i] += (p - y) * x[i]
                    for j in range(i, d):
                        h[i][j] += p * (1 - p) * x[i] * x[j]
            for i in range(d):
                for j in range(i):
                    h[i][j] = h[j][i]
                if i:                                  # the intercept is not penalised
                    g[i] += self.l2 * w[i]
                    h[i][i] += self.l2
            step = _solve(h, g)
            w = [a - b for a, b in zip(w, step)]
            if max(abs(s) for s in step) < 1e-8:
                break
        self.w = w
        return self

    def prob(self, f):
        z = sum(a * b for a, b in zip(self.w, self._x(f)))
        return 1 / (1 + math.exp(-max(-35.0, min(35.0, z))))


# ----------------------------------------------------------------------------- metrics that respect imbalance

def average_precision(scores, ys):
    """PR-AUC as average precision, stepping through the distinct scores from the top, so tied scores count as
    one block (their order in the input changes nothing). A constant or random ranking scores about the base rate."""
    pos = sum(ys)
    if not pos:
        return float("nan")
    groups = {}
    for s, y in zip(scores, ys):
        g = groups.setdefault(s, [0, 0])
        g[0] += 1
        g[1] += y
    seen = hits = 0
    total = 0.0
    for s in sorted(groups, reverse=True):
        n, k = groups[s]
        seen += n
        hits += k
        total += (k / pos) * (hits / seen)
    return total


def roc_auc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(scores)
    rank = {}
    i = 0
    while i < len(ranked):                       # average ranks for ties
        j = i
        while j < len(ranked) and ranked[j] == ranked[i]:
            j += 1
        rank[ranked[i]] = (i + j + 1) / 2
        i = j
    return (sum(rank[s] for s in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def f1_at(scores, ys, t):
    tp = sum(1 for s, y in zip(scores, ys) if s >= t and y)
    fp = sum(1 for s, y in zip(scores, ys) if s >= t and not y)
    fn = sum(1 for s, y in zip(scores, ys) if s < t and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def best_threshold(scores, ys):
    """The threshold with the best F1 on the data given (the training years only, when evaluating)."""
    return max(sorted(set(scores)), key=lambda t: f1_at(scores, ys, t)[0], default=0.5)


def brier(scores, ys):
    return statistics.fmean((s - y) ** 2 for s, y in zip(scores, ys))


def scorecard(scores, ys, threshold, base_rate):
    """Every imbalance-aware number for one set of predictions, plus accuracy and the always-no accuracy
    beside it (to show why accuracy alone misleads)."""
    f1, p, r = f1_at(scores, ys, threshold)
    acc = statistics.fmean(1.0 if (s >= threshold) == bool(y) else 0.0 for s, y in zip(scores, ys))
    b = brier(scores, ys)
    b0 = brier([base_rate] * len(ys), ys)
    return {"n": len(ys), "pops": sum(ys), "base_rate": sum(ys) / len(ys), "pr_auc": average_precision(scores, ys),
            "roc_auc": roc_auc(scores, ys), "f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "accuracy_always_no": 1 - sum(ys) / len(ys), "brier": b, "brier_skill": 1 - b / b0 if b0 else float("nan")}


def permutation_p(scores, ys, metric=average_precision, n=1000, seed=7):
    """How often shuffled scores do as well as these: the chance a useless model scores this high."""
    rng, real = random.Random(seed), metric(scores, ys)
    s = list(scores)
    worse = 0
    for _ in range(n):
        rng.shuffle(s)
        worse += metric(s, ys) >= real
    return (worse + 1) / (n + 1)


def bootstrap_ci(scores, ys, metric=average_precision, n=BOOTSTRAP, seed=13, level=0.95):
    """A 95% interval for a metric: the IPOs resampled with replacement `n` times, the middle 95% of the results.
    The precision a figure deserves is the width of this interval, not its number of decimals."""
    rng = random.Random(seed)
    k = len(ys)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        sy = [ys[i] for i in idx]
        if 0 < sum(sy) < k:
            vals.append(metric([scores[i] for i in idx], sy))
    if not vals:                                   # every resample was all pops or none: no interval to give
        return float("nan"), float("nan")
    vals.sort()
    lo, hi = vals[int((1 - level) / 2 * (len(vals) - 1))], vals[int((1 + level) / 2 * (len(vals) - 1))]
    return lo, hi


def p_text(p, n=1000):
    """A permutation p-value stated exactly: the smallest possible one says so instead of rounding to 0.001."""
    return f"p < {1 / n:.3f} (none of {n:,} shuffled runs did as well)" if p <= 1 / (n + 1) + 1e-12 else f"p = {p:.3f}"


def share(k, n, d=1):
    return f"{100 * k / n:.{d}f}% ({k} of {n})" if n else "n/a"


# ----------------------------------------------------------------------------- 1. walk-forward and holdout

def fit_year(train, market, names=FEATURES, l2=L2):
    banks = bank_shares(train)
    feats = [features(r, market, banks) for r in train]
    ys = [int(popped(first_day(r))) for r in train]
    model = Logit(names, l2).fit(feats, ys)
    train_scores = [model.prob(f) for f in feats]
    return model, banks, best_threshold(train_scores, ys), sum(ys) / len(ys)


def walk_forward(rows, market, years=TEST_YEARS, names=FEATURES, l2=L2, window=None, settings=None):
    """Each test year predicted by a model fitted on earlier years only (all of them, or the last `window`):
    [{year, rows, scores, ys, ...}]. settings(year, train) -> (l2, window) re-chooses them each year."""
    folds = []
    for y in years:
        cut = f"{y}-01-01"
        earlier = [r for r in rows if r["prices"]["listing_date"] < cut]
        l2_y, window_y = settings(y, earlier) if settings else (l2, window)
        train = [r for r in earlier if window_y is None or r["prices"]["listing_date"] >= f"{y - window_y}-01-01"]
        test = [r for r in rows if cut <= r["prices"]["listing_date"] < f"{y + 1}-01-01"]
        if len(train) < 50 or not test:
            continue
        model, banks, t, base = fit_year(train, market, names, l2_y)
        feats = [features(r, market, banks) for r in test]
        folds.append({"year": y, "rows": test, "feats": feats, "model": model, "threshold": t, "base_rate": base,
                      "scores": [model.prob(f) for f in feats], "ys": [int(popped(first_day(r))) for r in test],
                      "train_last": max(r["prices"]["listing_date"] for r in train), "l2": l2_y, "window": window_y})
    return folds


def choose_settings(market, inner_years=2):
    """settings() for walk_forward: the penalty and window with the best PR-AUC when the last `inner_years` of
    the training years are each predicted from the years before them. Only training IPOs are ever looked at."""
    def settings(year, earlier):
        best, best_score = (L2, None), -1.0
        for l2 in L2_GRID:
            for window in WINDOWS:
                folds = walk_forward(earlier, market, years=range(year - inner_years, year), l2=l2, window=window)
                s, y, _ = pooled(folds)
                score = average_precision(s, y) if y and sum(y) else -1.0
                if score > best_score + 1e-12:
                    best, best_score = (l2, window), score
        return best
    return settings


def pooled(folds, years=None):
    fs = [f for f in folds if years is None or years[0] <= f["year"] <= years[1]]
    return ([s for f in fs for s in f["scores"]], [y for f in fs for y in f["ys"]], fs)


# ----------------------------------------------------------------------------- 2. leakage, sensitivity

def leakage_audit(rows, build=None):
    """Three checks that the features only use what was public before each listing:
    - dates: the range is from a filing before the listing day, and the final prospectus within the
      Rule 424(b) window (its terms are set at pricing, the evening before the first trade);
    - perturbation: rewriting an IPO's own trading (every price after the offer) changes neither its features
      nor any earlier or same-day IPO's;
    - a canary: a feature that does use the first day is planted, and the perturbation test must catch it.
    build(rows) -> {adsh: features} defaults to the real feature builder."""
    build = build or (lambda rs: _all_features(rs))
    # ranges filed on or after the listing day: counted, and kept out of the features (features() drops them)
    late = [r["adsh"] for r in rows if r.get("range_date") and r["range_date"] >= r["prices"]["listing_date"]]
    late_used = [a for a in late if not _all_features([r for r in rows if r["adsh"] == a])[a]["no_range"]]
    slow = [r["adsh"] for r in rows
            if (dt.date.fromisoformat(r["prospectus_date"]) - dt.date.fromisoformat(r["prices"]["listing_date"])).days > 5]
    leaks = perturbation_leaks(rows, build)
    canary = perturbation_leaks(rows, lambda rs: {a: {**f, "canary": first_day(next(r for r in rs if r["adsh"] == a))}
                                                  for a, f in build(rs).items()})
    return {"range_after_listing": late, "range_after_listing_used": late_used, "prospectus_late": slow, "leaks": leaks,
            "canary_caught": bool(canary)}


def _all_features(rows, spy20=None):
    market = market_features(rows, spy20 or {})
    banks = bank_shares(rows)
    return {r["adsh"]: features(r, market, banks) for r in rows}


def perturbation_leaks(rows, build, samples=25, seed=3):
    """IPOs whose features moved when a same-day-or-later IPO's trading was rewritten: [(changed, rewritten)]."""
    rng = random.Random(seed)
    base = build(rows)
    found = []
    for victim in rng.sample(rows, min(samples, len(rows))):
        day = victim["prices"]["listing_date"]
        changed = [dict(r, prices={**r["prices"], "close": r["prices"]["close"] * 3.0, "open": r["prices"]["open"] * 2.0,
                                   "close_21": None, "close_252": None}) if r is victim else r for r in rows]
        after = build(changed)
        for r in rows:
            if r["prices"]["listing_date"] <= day and after[r["adsh"]] != base[r["adsh"]]:
                found.append((r["adsh"], victim["adsh"]))
    return found


def sensitivity(folds, repeats=20, seed=11):
    """Per feature: the PR-AUC lost when its values are shuffled across the test IPOs, each year's fitted model
    kept. A feature the model leans on loses a lot; one it ignores loses nothing (drop_one adds pure noise
    as a refitted floor)."""
    rng = random.Random(seed)
    scores, ys, _ = pooled(folds)
    base = average_precision(scores, ys)
    out = {}
    for k in FEATURES:
        losses = []
        for _ in range(repeats):
            new = []
            for f in folds:
                col = [ft[k] for ft in f["feats"]]
                rng.shuffle(col)
                new += [f["model"].prob({**ft, k: v}) for ft, v in zip(f["feats"], col)]
            losses.append(base - average_precision(new, ys))
        out[k] = statistics.fmean(losses)
    return base, out


def drop_one(rows, market, folds_base):
    """PR-AUC with each feature left out and every year refitted, against the full model's."""
    s0, y0, _ = pooled(folds_base)
    base = average_precision(s0, y0)
    out = {}
    for k in FEATURES:
        names = [n for n in FEATURES if n != k]
        s, y, _ = pooled(walk_forward(rows, market, names=names))
        out[k] = base - average_precision(s, y)
    noise_rng = random.Random(5)
    noisy = {a: {**m, "noise": noise_rng.random()} for a, m in market.items()}
    s, y, _ = pooled(walk_forward(rows, noisy, names=FEATURES + ["noise"]))
    out["noise (added)"] = base - average_precision(s, y)
    return base, out


# ----------------------------------------------------------------------------- 3. trading: slippage, winner's curse

def top_picks(fold, top=0.2):
    """A fold's IPOs with the highest predicted pop: the top `top` share, at least one."""
    ranked = sorted(zip(fold["scores"], fold["rows"]), key=lambda t: -t[0])
    return [r for _, r in ranked[:max(1, int(len(fold["rows"]) * top))]]


def at_open(folds, slippage, top=0.2):
    """Buying at the first day's open and selling at that day's close, or 21 trading days on, with `slippage`
    lost on the way in and again on the way out. Every IPO, and the model's top fifth by predicted pop."""
    def ret(r, exit_key):
        exit_px = r["prices"].get(exit_key)
        if not exit_px:
            return None
        return (exit_px * (1 - slippage)) / (r["prices"]["open"] * (1 + slippage)) - 1
    out = {}
    for name, pick in (("every IPO", lambda f: f["rows"]),
                       ("model's top fifth", lambda f: top_picks(f, top))):
        picked = [r for f in folds for r in pick(f)]
        for exit_key, label in (("close", "day 1"), ("close_21", "21 days")):
            rs = [x for x in (ret(r, exit_key) for r in picked) if x is not None]
            out[(name, label)] = {"n": len(rs), "mean": statistics.fmean(rs) if rs else float("nan"),
                                  "median": statistics.median(rs) if rs else float("nan"),
                                  "win_rate": statistics.fmean(1.0 if x > 0 else 0.0 for x in rs) if rs else float("nan")}
    return out


def break_even_friction(folds, exit_key="close", hi=0.5):
    """The friction each way (slippage, spread and fees together) at which buying every IPO at the open and
    selling at `exit_key` stops making money on average; 0 if it never does."""
    mean = lambda f: at_open(folds, f)[("every IPO", "day 1" if exit_key == "close" else "21 days")]["mean"]
    if not mean(0.0) > 0:
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def allocation(pop, k):
    """A retail investor's fill on a request at the offer: all of it in a deal that doesn't pop (nobody else
    wanted it), and less the hotter the deal, since oversubscribed books are rationed. The first-day return
    stands in for demand, which the investor can't see in advance: that is the winner's curse."""
    return 1.0 if pop <= 0 else max(0.02, 1 / (1 + k * pop))


def winners_curse(folds, top=0.2):
    """Buying at the offer, selling at the first close: the return on what was asked for (as if filled in full)
    against the return on what would actually have been allocated, for every IPO and for the model's picks."""
    out = {}
    for name, pick in (("every IPO", lambda f: f["rows"]),
                       ("model's top fifth", lambda f: top_picks(f, top))):
        picked = [r for f in folds for r in pick(f)]
        rets = [first_day(r) for r in picked]
        row = {"n": len(rets), "as asked": statistics.fmean(rets)}
        for k in CURSE_K:
            w = [allocation(x, k) for x in rets]
            row[f"allocated k={k}"] = sum(a * x for a, x in zip(w, rets)) / sum(w)
            row[f"filled k={k}"] = statistics.fmean(w)
        out[name] = row
    return out


# ----------------------------------------------------------------------------- 3. live-feed stress

def score_live(event, model, banks, market_now):
    """Score one IPO from a live event at the open: {prospectus: text, registration: text, sic, foreign,
    opening: {indication: price} or anything}. Whatever is missing or malformed gives "unknown", never a crash."""
    def as_text(x):                               # a feed field that isn't text is treated as missing; only the cover
        return x[:LIVE_BYTES] if isinstance(x, (str, bytes)) else ""             # page is read (it holds the offer and range)
    try:
        if not isinstance(event, dict):
            return {"status": "unknown", "why": "not an event"}
        text = ipo_data.text_of(as_text(event.get("prospectus")))
        offer = ipo_data.offer_price(text)
        if not offer:
            return {"status": "unknown", "why": "no offer price in the prospectus"}
        rng = ipo_data.price_range(ipo_data.text_of(as_text(event.get("registration"))))
        row = {"adsh": "live", "offer_price": offer, "shares": ipo_data.shares_offered(text), "lead_bank": ipo_data.lead_bank(text),
               "range": list(rng) if rng else None, "sic": event.get("sic") if isinstance(event.get("sic"), str) else None,
               "foreign": event.get("foreign") is True}
        p = model.prob(features(row, {"live": market_now}, banks))
        out = {"status": "ok", "p_pop": p, "offer": offer}
        ind = event.get("opening")
        if isinstance(ind, dict) and isinstance(ind.get("indication"), (int, float)) and ind["indication"] > 0:
            out["indicated_open_vs_offer"] = ind["indication"] / offer - 1
        return out
    except Exception as e:                       # the stress test counts these; there should be none
        return {"status": "error", "why": f"{type(e).__name__}: {e}"}


def synthetic_events(n, seed=17):
    """Live events at the open, many of them broken the ways real feeds break: truncated or empty text,
    junk markup, no dollar sign, a 2 MB filing, wrong types, garbled or missing opening indications."""
    rng = random.Random(seed)
    good = ("<html><body><p>PROSPECTUS</p><p>{shares:,} Shares</p><p>Class A Common Stock</p><p>This is an initial public "
            "offering. The initial public offering price is ${price:.2f} per share.</p><p>Goldman Sachs &amp; Co. LLC "
            "&nbsp; J.P. Morgan</p></body></html>")
    reg = "<p>We expect the initial public offering price to be between ${lo:.2f} and ${hi:.2f} per share.</p>"
    breaks = [lambda t: t[: rng.randint(0, len(t))], lambda t: "", lambda t: t.replace("$", ""), lambda t: t * 2500,
              lambda t: "<<<>>>" + t + "</div" * 50, lambda t: None, lambda t: 12345, lambda t: t.encode("utf-8"),
              lambda t: t.replace("per share", "per shäre �"), lambda t: t]
    out = []
    for i in range(n):
        price = round(rng.uniform(4, 60), 2)
        e = {"prospectus": good.format(shares=rng.randint(1, 40) * 1_000_000, price=price),
             "registration": reg.format(lo=price * 0.9, hi=price * 1.05), "sic": rng.choice(["7372", "2834", "6022", None, "x"]),
             "foreign": rng.random() < 0.2,
             "opening": rng.choice([{"indication": price * rng.uniform(0.8, 2.0)}, {"indication": "n/a"}, None, [], {"indication": -1}])}
        if rng.random() < 0.5:
            key = rng.choice(["prospectus", "registration"])
            e[key] = rng.choice(breaks)(e[key])
        out.append(e)
    return out


def stress(model, banks, n=2000, burst=200, budget_ms=50.0):
    """Every synthetic event scored once in sequence (latency per event), then `burst` at once on threads, as
    if the whole calendar opened together. Reports errors, unknowns and the p50/p99 latency against a budget."""
    market_now = {"spy_20d": 0.01, "heat_30d": 0.15, "deals_30d": math.log1p(6)}
    events = synthetic_events(n)
    lat, results = [], []
    for e in events:
        t = time.perf_counter()
        results.append(score_live(e, model, banks, market_now))
        lat.append(1000 * (time.perf_counter() - t))
    burst_results, threads = [], []
    lock = threading.Lock()

    def run(e):
        r = score_live(e, model, banks, market_now)
        with lock:
            burst_results.append(r)
    t0 = time.perf_counter()
    for e in events[:burst]:
        threads.append(threading.Thread(target=run, args=(e,)))
        threads[-1].start()
    for t in threads:
        t.join()
    wall = 1000 * (time.perf_counter() - t0)
    lat.sort()
    return {"events": n, "errors": sum(r["status"] == "error" for r in results + burst_results),
            "unknown": sum(r["status"] == "unknown" for r in results), "ok": sum(r["status"] == "ok" for r in results),
            "p50_ms": lat[len(lat) // 2], "p99_ms": lat[int(len(lat) * 0.99)], "max_ms": lat[-1], "budget_ms": budget_ms,
            "burst": burst, "burst_wall_ms": wall,
            "error_examples": [r["why"] for r in results + burst_results if r["status"] == "error"][:3]}


# ----------------------------------------------------------------------------- the report

def spy_closes():
    """SPY daily closes {date: close} since 2014, from the same Yahoo chart ipo_data uses; cached for the day."""
    path = ROOT / "bench_data" / "ipo" / "spy.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fetched") == dt.date.today().isoformat():
            return cached["closes"]
    t0 = int(dt.datetime(2014, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    import urllib.request
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1={t0}&period2={int(time.time())}&interval=1d"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    closes = {dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(): c
              for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]) if c}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched": dt.date.today().isoformat(), "closes": closes}), encoding="utf-8")
    return closes


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}%"


def num(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def coverage_lines(raw, rows):
    """Coverage by year (survivorship), and what usable() left out of the priced IPOs, with how they traded."""
    lines = ["## Coverage (survivorship)", "", "| Year | IPOs | priced | share |", "|---|---|---|---|"]
    for y in sorted({r["prospectus_date"][:4] for r in raw}):
        n = sum(r["prospectus_date"][:4] == y for r in raw)
        k = sum(r["prospectus_date"][:4] == y for r in rows)
        lines.append(f"| {y} | {n} | {k} | {100 * k / n:.0f}% |")
    lines += ["", "An IPO with no prices was delisted (or its ticker changed): the failures leave the sample, which flatters "
              "later returns more than first days.", ""]
    lines += ["## Excluded from the sample", "", "| Reason | IPOs | median first day | mean first day |", "|---|---|---|---|"]
    for reason, rs in excluded(raw).items():
        fd = [first_day(r) for r in rs]
        lines.append(f"| {reason} | {len(rs)} | {pct(statistics.median(fd) if fd else None)} | "
                     f"{pct(statistics.fmean(fd) if fd else None)} |")
    lines += ["", "Suspect prices are an open below half the offer or above four times it, so they are judged on the listing "
              "day itself; they are left out because they are almost always a rescaled history or a misread offer, and "
              f"counted here so the selection is visible. An offer outside {OFFER_BAND[0]:g}x the range's low to "
              f"{OFFER_BAND[1]:g}x its high is caught from the filings alone.", ""]
    return lines


def walk_forward_lines(folds, raw, rows):
    """The fixed walk-forward's table and pooled scores: (lines, pooled scorecard, PR-AUC interval)."""
    lines = ["## 1. Walk-forward by year", "", "Each year predicted by a model fitted only on the years before it (the "
             "penalty fixed in advance, the F1 threshold chosen on the training years).", "",
             "| Test year | IPOs | pops | base rate | PR-AUC | ROC-AUC | F1 | precision | recall | Brier skill | p (PR-AUC) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in folds:
        c = scorecard(f["scores"], f["ys"], f["threshold"], f["base_rate"])
        lines.append(f"| {f['year']} | {c['n']} | {c['pops']} | {100 * c['base_rate']:.1f}% | {num(c['pr_auc'])} | {num(c['roc_auc'])} | "
                     f"{num(c['f1'])} | {num(c['precision'])} | {num(c['recall'])} | {num(c['brier_skill'])} | "
                     f"{num(permutation_p(f['scores'], f['ys']), 3)} |")
    s, y, fs = pooled(folds)
    allc = scorecard(s, y, statistics.fmean(f["threshold"] for f in fs), statistics.fmean(f["base_rate"] for f in fs))
    ap_lo, ap_hi = bootstrap_ci(s, y)
    roc_lo, roc_hi = bootstrap_ci(s, y, roc_auc)
    lines += ["", f"All test years: PR-AUC {num(allc['pr_auc'])} (95% CI {num(ap_lo)} to {num(ap_hi)}) against a base rate of "
              f"{share(sum(y), len(y))} (what a random ranking scores), a lift of {allc['pr_auc'] / allc['base_rate']:.2f}x "
              f"({ap_lo / allc['base_rate']:.2f}x to {ap_hi / allc['base_rate']:.2f}x); {p_text(permutation_p(s, y))}; "
              f"ROC-AUC {num(allc['roc_auc'])} (95% CI {num(roc_lo)} to {num(roc_hi)}); F1 {num(allc['f1'])}.", ""]
    years = sorted({r["prospectus_date"][:4] for r in raw})
    covered = {yr for yr in years if sum(r["prospectus_date"][:4] == yr for r in rows) >= COVERED * sum(r["prospectus_date"][:4] == yr for r in raw)}
    fc = [f for f in folds if str(f["year"]) in covered]
    if fc:
        s3, y3, _ = pooled(fc)
        lines += [f"Only the test years where at least {100 * COVERED:.0f}% of IPOs kept their prices "
                  f"({', '.join(str(f['year']) for f in fc)}): PR-AUC {num(average_precision(s3, y3))} against a base rate of "
                  f"{share(sum(y3), len(y3))}, ROC-AUC {num(roc_auc(s3, y3))}.", ""]
    else:
        lines += [f"No test year has prices for {100 * COVERED:.0f}% of its IPOs.", ""]
    return lines, allc, (ap_lo, ap_hi)


def reoptimised_lines(dev, market, folds, allc, ci):
    """The walk-forward with the penalty and window re-chosen each year, beside the fixed one."""
    reopt = walk_forward(dev, market, settings=choose_settings(market))
    s4, y4, _ = pooled(reopt)
    lines = ["## 1. Walk-forward, re-optimised each year", "",
             f"Each January the penalty (from {', '.join(str(x) for x in L2_GRID)}) and the training window (every earlier year, "
             "or the last three) are chosen by predicting the last two training years from the years before them; "
             "the test year plays no part.", "",
             "| Test year | chosen penalty | window | PR-AUC re-optimised | PR-AUC fixed |", "|---|---|---|---|---|"]
    fixed_by_year = {f["year"]: f for f in folds}
    for f in reopt:
        fx = fixed_by_year.get(f["year"])
        span = "all years" if f["window"] is None else f"last {f['window']} years"
        lines.append(f"| {f['year']} | {f['l2']:g} | {span} | "
                     f"{num(average_precision(f['scores'], f['ys']))} | {num(average_precision(fx['scores'], fx['ys'])) if fx else 'n/a'} |")
    r_lo, r_hi = bootstrap_ci(s4, y4)
    lines += ["", f"All test years: re-optimised PR-AUC {num(average_precision(s4, y4))} (95% CI {num(r_lo)} to {num(r_hi)}) against "
              f"fixed {num(allc['pr_auc'])} ({num(ci[0])} to {num(ci[1])}).", ""]
    return lines


def regime_lines(folds, spy):
    """Scores by calendar regime, and by SPY's state on each listing day."""
    lines = ["## 1. Market regimes", "", "| Regime | IPOs | base rate | PR-AUC (95% CI) | lift over base | ROC-AUC | F1 |", "|---|---|---|---|---|---|---|"]
    for name, span in REGIMES.items():
        s2, y2, f2 = pooled(folds, span)
        if not y2 or not sum(y2):
            continue
        c = scorecard(s2, y2, statistics.fmean(f["threshold"] for f in f2), statistics.fmean(f["base_rate"] for f in f2))
        lo, hi = bootstrap_ci(s2, y2)
        lines.append(f"| {name} | {c['n']} | {share(c['pops'], c['n'])} | {num(c['pr_auc'])} ({num(lo)} to {num(hi)}) | "
                     f"{num(c['pr_auc'] / c['base_rate'], 2)}x | {num(c['roc_auc'])} | {num(c['f1'])} |")
    lab = [(spy_regime(spy, r["prices"]["listing_date"]), sc, yv) for f in folds for r, sc, yv in zip(f["rows"], f["scores"], f["ys"])]
    for reg in ("bull", "drawdown", "sideways", "ordinary"):
        sc = [x[1] for x in lab if x[0] == reg]
        yv = [x[2] for x in lab if x[0] == reg]
        if len(yv) >= 20 and sum(yv):
            lo, hi = bootstrap_ci(sc, yv)
            lines.append(f"| SPY {reg} on the listing day | {len(yv)} | {share(sum(yv), len(yv))} | {num(average_precision(sc, yv))} "
                         f"({num(lo)} to {num(hi)}) | {num(average_precision(sc, yv) / (sum(yv) / len(yv)), 2)}x | {num(roc_auc(sc, yv))} | n/a |")
        else:
            lines.append(f"| SPY {reg} on the listing day | {len(yv)} | too few to score | | | | |")
    lines.append("")
    return lines


def holdout_lines(dev, hold, market, final, again):
    """The sealed holdout: scored once (and logged) with --final; later runs show the result of record.
    Returns (lines, final): final is False when the recorded result was shown instead of a new score."""
    looks = [json.loads(x) for x in HOLDOUT_LOG.read_text(encoding="utf-8").splitlines() if x.strip()] if HOLDOUT_LOG.exists() else []
    if final and looks and not again:
        rec = looks[0]
        return ["## 1. Sealed holdout (the result of record)", "", f"Scored once on {rec['run'][:10]}: {rec['text']} "
                f"Looked at {len(looks)} time(s); `--final --again` scores it again and logs another look.", ""], False
    if not final:
        return ["## 1. Sealed holdout", "", f"Not scored: IPOs listed from {HOLDOUT_FROM} stay sealed until `--final`.", ""], False
    model, banks, t, base = fit_year(dev, market)
    hs = [model.prob(features(r, market, banks)) for r in hold]
    hy = [int(popped(first_day(r))) for r in hold]
    lines = ["## 1. Sealed holdout", ""]
    if hy and sum(hy):
        c = scorecard(hs, hy, t, base)
        text = (f"{c['n']} IPOs listed from {HOLDOUT_FROM}, never used while building: PR-AUC {num(c['pr_auc'])} against a "
                f"base rate of {pct(c['base_rate'], 0)} (p = {num(permutation_p(hs, hy), 3)}), ROC-AUC {num(c['roc_auc'])}, "
                f"F1 {num(c['f1'])}.")
        lines += [text, f"This is look number {len(looks) + 1}.", ""]
        HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HOLDOUT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": dt.datetime.now().isoformat(timespec="seconds"), "text": text}) + "\n")
    else:
        lines += ["Too few holdout IPOs with prices to score.", ""]
    return lines, True


def leakage_lines(dev, spy20):
    audit = leakage_audit(dev, lambda rs: _all_features(rs, spy20))
    return ["## 2. Leakage audit", "",
            f"- Ranges from a filing on or after the listing day: {len(audit['range_after_listing'])}, of which the features use "
            f"{len(audit['range_after_listing_used'])} (they are treated as having no range).",
            f"- Final prospectuses filed more than five days after listing: {len(audit['prospectus_late'])}.",
            f"- Perturbation: rewriting an IPO's trading changed {len(audit['leaks'])} earlier or same-day IPOs' features.",
            f"- The planted canary (a feature that reads the first day) was {'caught' if audit['canary_caught'] else 'MISSED'}.",
            "- League tables (bank share) are rebuilt from each fold's training years only.", ""]


def sensitivity_lines(dev, market, folds):
    base, perm = sensitivity(folds)
    _, drop = drop_one(dev, market, folds)
    lines = ["## 2. Feature sensitivity", "", f"PR-AUC over the test years: {num(base)}. Loss when a feature is shuffled "
             "(models kept) or dropped (every year refitted); a pure-noise feature is the floor.", "",
             "| Feature | shuffled | dropped |", "|---|---|---|"]
    for k in sorted(FEATURES, key=lambda k: -perm[k]):
        lines.append(f"| {k} | {perm[k]:+.4f} | {drop[k]:+.4f} |")
    return lines + [f"| pure noise, added and refitted (the floor) | n/a | {drop['noise (added)']:+.4f} |", ""]


def imbalance_lines(allc):
    return ["## 2. Class imbalance", "", f"Over the test years {allc['pops']} of {allc['n']} IPOs popped ({100 * allc['base_rate']:.1f}%). "
            f"Accuracy {100 * allc['accuracy']:.1f}% looks fine until set beside always saying no: {100 * allc['accuracy_always_no']:.1f}%. "
            f"PR-AUC {num(allc['pr_auc'])} (random: {num(allc['base_rate'])}), F1 {num(allc['f1'])}, precision "
            f"{num(allc['precision'])}, recall {num(allc['recall'])}, Brier skill {num(allc['brier_skill'])}.", ""]


def trading_lines(folds):
    """Slippage buying at the open, the break-even friction, and the winner's curse buying at the offer."""
    lines = ["## 3. Slippage, buying at the open", "", "| Picks | exit | slippage | n | mean | median | win rate |", "|---|---|---|---|---|---|---|"]
    for sl in FRICTION:
        for (name, label), v in at_open(folds, sl).items():
            lines.append(f"| {name} | {label} | {100 * sl:.2f}% each way | {v['n']} | {pct(v['mean'], 2)} | {pct(v['median'], 2)} | {100 * v['win_rate']:.1f}% |")
    be1, be21 = break_even_friction(folds), break_even_friction(folds, "close_21")
    lines += ["", f"Break-even friction, each way, for buying every IPO at the open: {100 * be1:.2f}% held to the first close, "
              f"{100 * be21:.2f}% held 21 days (0 means it loses money before any cost).", ""]
    curse = winners_curse(folds)
    lines += ["## 3. Winner's curse, buying at the offer", "", "Return to the first close on what was asked for, and on what "
              "a retail book would have filled (all of a flop, less of a hot deal).", "",
              "| Picks | n | as asked | " + " | ".join(f"allocated k={k} (filled)" for k in CURSE_K) + " |",
              "|---|---|---|" + "---|" * len(CURSE_K)]
    for name, v in curse.items():
        lines.append(f"| {name} | {v['n']} | {pct(v['as asked'], 2)} | " +
                     " | ".join(f"{pct(v[f'allocated k={k}'], 2)} ({100 * v[f'filled k={k}']:.1f}%)" for k in CURSE_K) + " |")
    lines.append("")
    return lines


def stress_lines(dev, market):
    model, banks, _, _ = fit_year(dev, market)
    st = stress(model, banks)
    return ["## 3. Live-feed stress at the open", "",
            f"{st['events']} synthetic events, half broken (truncated, empty, junk markup, 2 MB, wrong types, garbled "
            f"indications): {st['ok']} scored, {st['unknown']} returned \"unknown\", {st['errors']} crashed. Latency p50 "
            f"{st['p50_ms']:.2f} ms, p99 {st['p99_ms']:.2f} ms, max {st['max_ms']:.1f} ms (budget {st['budget_ms']:.0f} ms). "
            f"{st['burst']} at once on threads: {st['burst_wall_ms']:.0f} ms wall.", ""]


def report(final=False, again=False):
    raw = ipo_data.load()
    rows = usable(raw)
    dev, hold = split(rows, final)
    spy = spy_closes()
    spy20 = spy_returns(spy)
    market = market_features(rows if final else dev, spy20)
    lines = [f"# IPO evaluation, {dt.date.today().isoformat()}", "",
             f"{len(raw)} IPOs with a final prospectus since {ipo_data.FIRST_YEAR}; {len(rows)} with an offer price and trustworthy first-day "
             f"prices; a pop is a first-day close {100 * POP:.0f}% or more above the offer.", ""]
    lines += coverage_lines(raw, rows)
    folds = walk_forward(dev, market)
    wf, allc, ci = walk_forward_lines(folds, raw, rows)
    lines += wf + reoptimised_lines(dev, market, folds, allc, ci) + regime_lines(folds, spy)
    hl, final = holdout_lines(dev, hold, market, final, again)
    lines += hl + leakage_lines(dev, spy20) + sensitivity_lines(dev, market, folds) + imbalance_lines(allc)
    lines += trading_lines(folds) + stress_lines(dev, market)

    out = ROOT / "bench_runs" / f"ipo_eval_{dt.date.today().isoformat()}{'_final' if final else ''}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out}")
    return out


if __name__ == "__main__":  # pragma: no cover - the command line entry; report() is tested
    report(final="--final" in sys.argv, again="--again" in sys.argv)
