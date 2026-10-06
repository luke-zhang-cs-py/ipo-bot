"""Walk-forward tests of simple prediction algorithms, month after month, on real prices.

    python benchmark.py                 both universes, 12-month and 1-month horizons
    python benchmark.py --refresh       fetch prices again instead of using today's cache
    python benchmark.py --ledger        also score the bot's own logged forecasts (forecasts/ledger.jsonl)

Every month-end, each algorithm sees only the prices up to that day and predicts which stocks will do best
over the next 12 months (the bot's horizon) and the next month. The prediction is then scored against what
happened. The same test is repeated at every month-end and summarised over consecutive 3-year blocks, so a
result that held in one period and vanished in the next shows up as such.

The bot itself is not in the backtest, and cannot fairly be: the model already knows how these years turned
out. It is tested forward instead: each rating it gives is logged with the date and price (record_forecast),
and --ledger scores the ones that have matured against what these algorithms predicted on the same day.

Limits worth knowing:
- The universes are today's survivors (companies that were large then and still listed now), which flatters
  every long-only result. Compare algorithms with each other, not with zero.
- 12-month windows starting a month apart overlap, so t-statistics use Newey-West errors.
- Prices are Yahoo's adjusted closes (splits and dividends), fetched for this test only and cached in
  bench_data/ (git-ignored). No trading costs or taxes.
"""
import datetime as dt
import json
import math
import pathlib
import random
import statistics
import sys
import urllib.request

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "bench_data"
LEDGER = HERE / "forecasts" / "ledger.jsonl"
START = dt.date(2004, 1, 1)
BLOCK_YEARS = 3

UNIVERSES = {
    "US large caps": ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "BAC", "WFC", "XOM", "CVX", "JNJ", "PFE", "MRK",
                      "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "CSCO", "INTC", "IBM", "ORCL", "T", "VZ",
                      "CAT", "GE", "MMM", "BA", "UNH"],
    "Global (country ETFs)": ["SPY", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "FXI", "EWY", "EWT",
                              "EWH", "EWS", "EWQ"],
    # Sector funds never delist, so this one has no survivor bias.
    "US sectors (SPDR ETFs)": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
}


# ----------------------------------------------------------------------------- prices

def fetch(symbol, refresh=False):
    """Month-end adjusted closes {"YYYY-MM": price}, cached for the day."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{symbol}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["fetched"] == dt.date.today().isoformat():
            return cached["months"]
    p1 = int(dt.datetime(START.year, START.month, START.day, tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    months = {}
    for ts, px in zip(res["timestamp"], res["indicators"]["adjclose"][0]["adjclose"]):
        if px:
            months[dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m")] = px   # the last day wins
    months.pop(dt.date.today().strftime("%Y-%m"), None)                                  # this month is not over
    path.write_text(json.dumps({"fetched": dt.date.today().isoformat(), "months": months}), encoding="utf-8")
    return months


def panel(symbols, refresh=False):
    """(months, {symbol: [price per month]}) over the months every symbol has."""
    raw = {s: fetch(s, refresh) for s in symbols}
    months = sorted(set.intersection(*(set(m) for m in raw.values())))
    return months, {s: [raw[s][m] for m in months] for s in symbols}


# ----------------------------------------------------------------------------- the algorithms
# Each takes one stock's prices up to and including month i, and returns a score: higher = expected to do better.

def momentum(p, i):          # 12-month return, skipping the latest month (Jegadeesh-Titman)
    return p[i - 1] / p[i - 12] - 1


def reversal(p, i):          # last month's losers bounce back
    return -(p[i] / p[i - 1] - 1)


def trend(p, i):             # price against its 10-month average (Faber)
    return p[i] / statistics.fmean(p[i - 9:i + 1]) - 1


def low_volatility(p, i):    # calmer stocks, over 3 years
    r = [p[k] / p[k - 1] - 1 for k in range(i - 35, i + 1)]
    return -statistics.pstdev(r)


def historical_mean(p, i):   # the past 5 years' average return will continue
    return (p[i] / p[i - 60]) ** (1 / 60) - 1


ALGORITHMS = {"Momentum 12-1": momentum, "Trend (10-mo avg)": trend, "Low volatility": low_volatility,
              "Historical mean": historical_mean, "Short-term reversal": reversal}
LOOKBACK = 60


def blend(scores_by_algo):   # equal-weight average of the ranks of momentum, trend and low volatility
    parts = [ranks(scores_by_algo[a]) for a in ("Momentum 12-1", "Trend (10-mo avg)", "Low volatility")]
    return [statistics.fmean(x) for x in zip(*parts)]


# ----------------------------------------------------------------------------- statistics

def ranks(xs):
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    out = [0.0] * len(xs)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[k]]:
            j += 1
        for m in range(k, j + 1):
            out[order[m]] = (k + j) / 2 + 1
        k = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def newey_west_t(xs, lag):
    """t-statistic of the mean, with errors widened for the overlap of consecutive windows."""
    n = len(xs)
    if n < 3:
        return float("nan")
    m = statistics.fmean(xs)
    d = [x - m for x in xs]
    var = sum(x * x for x in d) / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        var += 2 * w * sum(d[k] * d[k - L] for k in range(L, n)) / n
    return m / math.sqrt(var / n) if var > 0 else float("nan")


def max_drawdown(curve):
    peak, worst = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


# ----------------------------------------------------------------------------- the walk-forward test

def run(universe, refresh=False, seed=7, keep=None):
    """keep: a dict to fill with every month's scores and outcomes, for the permutation test."""
    symbols = UNIVERSES[universe]
    months, px = panel(symbols, refresh)
    names = list(ALGORITHMS) + ["Blend (mom+trend+lowvol)", "Random (no skill)"]
    rng = random.Random(seed)
    if keep is not None:
        keep["months"] = months
    per = {h: {a: [] for a in names} for h in (1, 12)}   # per horizon and algorithm: one record per month-end
    for i in range(LOOKBACK, len(months) - 1):
        raw = {a: [f(px[s], i) for s in symbols] for a, f in ALGORITHMS.items()}
        raw["Blend (mom+trend+lowvol)"] = blend(raw)
        raw["Random (no skill)"] = [rng.random() for _ in symbols]
        if keep is not None:
            keep.setdefault("raw", {})[i] = raw
        for h in (1, 12):
            if i + h >= len(months):
                continue
            fwd = [px[s][i + h] / px[s][i] - 1 for s in symbols]
            if keep is not None:
                keep.setdefault(h, {})[i] = fwd
            avg = statistics.fmean(fwd)
            med = statistics.median(fwd)
            for a in names:
                sc = raw[a]
                order = sorted(range(len(symbols)), key=lambda k: sc[k])
                third = len(symbols) // 3
                bottom, top = order[:third], order[-third:]
                per[h][a].append({
                    "month": months[i], "ic": spearman(sc, fwd),
                    "spread": statistics.fmean(fwd[k] for k in top) - statistics.fmean(fwd[k] for k in bottom),
                    "hits": sum(fwd[k] > med for k in top) + sum(fwd[k] < med for k in bottom),
                    "calls": sum(fwd[k] != med for k in top + bottom),   # the median stock is neither
                    "top_ret": statistics.fmean(fwd[k] for k in top), "all_ret": avg,
                })
    return months, per


LUCK_SEEDS = 200


def luck_band(keep, algo, h, seeds=LUCK_SEEDS):
    """The middle 95% of mean rank IC when the algorithm's own scores are handed to the wrong stocks (one fixed
    swap per run). The swap keeps how slowly the signal changes, which matters: a slow signal over overlapping
    12-month windows swings much more by chance than a ranking redrawn every month, and Newey-West t runs
    a little hot on these small universes too. This is the bar that counts."""
    months = sorted(keep[h])
    n = len(keep[h][months[0]])
    rng = random.Random(1000)
    means = []
    for _ in range(seeds):
        perm = list(range(n))
        rng.shuffle(perm)
        means.append(statistics.fmean(spearman([keep["raw"][i][algo][k] for k in perm], keep[h][i]) for i in months))
    means.sort()
    return means[round(0.025 * (seeds - 1))], means[round(0.975 * (seeds - 1))]


def summary(per, h):
    rows = []
    for a, recs in per[h].items():
        ics = [r["ic"] for r in recs]
        spreads = [r["spread"] for r in recs]
        rows.append({
            "algo": a, "n": len(recs), "ic": statistics.fmean(ics), "ic_t": newey_west_t(ics, h - 1),
            "ic_pos": sum(x > 0 for x in ics) / len(ics), "spread": statistics.fmean(spreads),
            "spread_t": newey_west_t(spreads, h - 1),
            "hit": sum(r["hits"] for r in recs) / sum(r["calls"] for r in recs),
        })
    return sorted(rows, key=lambda r: -r["ic"])


def blocks(per, h):
    """Mean IC per algorithm in consecutive BLOCK_YEARS-year blocks of start dates."""
    any_algo = next(iter(per[h].values()))
    years = sorted({int(r["month"][:4]) for r in any_algo})
    edges = list(range(years[0], years[-1] + 1, BLOCK_YEARS))
    labels = [f"{y}-{str(min(y + BLOCK_YEARS - 1, years[-1]))[2:]}" for y in edges]
    out = {}
    for a, recs in per[h].items():
        out[a] = [statistics.fmean([r["ic"] for r in recs if y <= int(r["month"][:4]) < y + BLOCK_YEARS] or [0])
                  for y in edges]
    return labels, out


def backtest(per, months):
    """Hold each algorithm's top third for a month, rebalance monthly: growth of $1 against the equal-weight universe."""
    out = {}
    base = per[1][next(iter(per[1]))]
    for a, recs in list(per[1].items()) + [("Equal-weight universe", [{"top_ret": r["all_ret"]} for r in base])]:
        rets = [r["top_ret"] for r in recs]
        curve = [1.0]
        for r in rets:
            curve.append(curve[-1] * (1 + r))
        years = len(rets) / 12
        sd = statistics.pstdev(rets) * math.sqrt(12)
        out[a] = {"cagr": curve[-1] ** (1 / years) - 1, "vol": sd,
                  "sharpe": (statistics.fmean(rets) * 12) / sd if sd else float("nan"), "mdd": max_drawdown(curve)}
    return out


# ----------------------------------------------------------------------------- the bot's own forecasts

def score_ledger(refresh=False):
    """Matured bot forecasts against the algorithms' calls for the same stock on the same month-end."""
    if not LEDGER.exists():
        return "No bot forecasts logged yet (forecasts/ledger.jsonl). Each rating the bot gives is logged; " \
               "the first ones can be scored 12 months after they are made."
    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    today = dt.date.today()
    matured, pending = [], []
    for r in rows:
        made = dt.date.fromisoformat(r["date"])
        due = dt.date(made.year + (made.month + r["horizon_months"] - 1) // 12,
                      (made.month + r["horizon_months"] - 1) % 12 + 1, 1)
        (matured if due <= today.replace(day=1) else pending).append((r, made, due))
    lines = [f"{len(rows)} bot forecasts logged: {len(matured)} matured, {len(pending)} pending."]
    if pending:
        lines.append("Next to mature: " + ", ".join(f"{r['symbol']} ({due:%b %Y})"
                                                    for r, _, due in sorted(pending, key=lambda x: x[2])[:5]))
    hits = 0
    for r, made, due in matured:
        months = fetch(r["symbol"], refresh)
        end = (due - dt.timedelta(days=1)).strftime("%Y-%m")
        if end not in months:
            lines.append(f"- {r['symbol']}: no price for {end}")
            continue
        realised = months[end] / r["price"] - 1
        said_up = r["expected_return_pct"] > 0
        hits += said_up == (realised > 0)
        lines.append(f"- {r['date']} {r['symbol']} {r['rating']}: expected {r['expected_return_pct']:+.1f}%, "
                     f"got {100 * realised:+.1f}%")
    if matured:
        lines.append(f"Direction right on {hits} of {len(matured)}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- report

def pct(x, d=1):
    text = f"{100 * x:+.{d}f}%"
    return text[1:] if float(text[:-1]) == 0 else text      # no "-0.0%" from rounding


def report(refresh=False, ledger=False):
    out = [f"# Prediction benchmark, {dt.date.today():%d %b %Y}", ""]
    for uni in UNIVERSES:
        keep = {}
        months, per = run(uni, refresh, keep=keep)
        out += [f"## {uni} ({len(UNIVERSES[uni])} names, predictions every month-end "
                f"{per[1][next(iter(per[1]))][0]['month']} to {per[1][next(iter(per[1]))][-1]['month']})", ""]
        for h in (12, 1):
            out += [f"### {h}-month horizon ({len(next(iter(per[h].values())))} consecutive tests)", "",
                    "| Algorithm | Rank IC | IC from luck (95%) | Beyond luck? | t | Months IC > 0 | Top - bottom third | t | Rating hit rate |",
                    "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
            for r in summary(per, h):
                lo, hi = luck_band(keep, r["algo"], h)
                verdict = "yes, better" if r["ic"] > hi else "yes, worse" if r["ic"] < lo else "no"
                out.append(f"| {r['algo']} | {r['ic']:+.3f} | {lo:+.3f} to {hi:+.3f} | {verdict} | {r['ic_t']:+.1f} | "
                           f"{100 * r['ic_pos']:.0f}% | {pct(r['spread'])} | {r['spread_t']:+.1f} | {100 * r['hit']:.1f}% |")
            labels, b = blocks(per, h)
            out += ["", f"Rank IC in consecutive {BLOCK_YEARS}-year blocks ({h}-month horizon):", "",
                    "| Algorithm | " + " | ".join(labels) + " | Blocks > 0 |",
                    "|---|" + "---:|" * (len(labels) + 1)]
            for a, ics in sorted(b.items(), key=lambda kv: -statistics.fmean(kv[1])):
                out.append(f"| {a} | " + " | ".join(f"{x:+.2f}" for x in ics) + f" | {sum(x > 0 for x in ics)}/{len(ics)} |")
            out.append("")
        out += ["### Holding each algorithm's top third, rebalanced monthly", "",
                "| Portfolio | Annual return | Volatility | Sharpe (no risk-free) | Worst drawdown |", "|---|---:|---:|---:|---:|"]
        for a, s in sorted(backtest(per, months).items(), key=lambda kv: -kv[1]["sharpe"]):
            out.append(f"| {a} | {pct(s['cagr'])} | {100 * s['vol']:.1f}% | {s['sharpe']:.2f} | {pct(s['mdd'])} |")
        out.append("")
    out += ["## How to read this", "",
            "- **Rank IC**: correlation between the predicted order and the order that happened, each month. "
            "0 is no skill; +0.05 is respectable for a stock signal; random scores near 0.",
            "- **Beyond luck?**: whether the rank IC falls outside what the same algorithm's scores, handed to the "
            "wrong stocks, produced in 95% of 200 runs. Each algorithm gets its own range because slow signals swing "
            "more by chance. This is the test to trust; any row, Random included, lands outside by chance 1 time in 20.",
            "- **t**: Newey-West, allowing for overlapping windows. It runs a little hot on these small universes "
            "(|t| > 2 in up to 10% of random runs), so read it alongside Beyond luck.",
            "- **Rating hit rate**: the top third read as Overweight, the bottom third as Underweight. The share of those "
            "calls on the right side of the median. 50% is a coin.",
            "- **Blocks**: the same test in consecutive periods. A signal that only worked in one block is fragile.",
            "- Survivor universes flatter long-only returns. Compare rows, not rows with zero.", ""]
    if ledger:
        out += ["## The bot's own forecasts", "", score_ledger(refresh), ""]
    return "\n".join(out)


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    text = report(refresh="--refresh" in argv, ledger="--ledger" in argv)
    runs = HERE / "bench_runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
