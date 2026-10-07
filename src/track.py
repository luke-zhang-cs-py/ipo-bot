"""Live tracking: score the bot's logged recommendations once they mature.

    python src/track.py                 score forecasts/ledger.jsonl
    python src/track.py other.jsonl     score another log in the same format

The log (written by the bot's record_forecast tool, one JSON object per line):
    date, symbol, rating, conviction, expected_return_pct, price, price_date, horizon_months,
    bull_value, bull_prob, base_value, base_prob, bear_value, bear_prob

After the horizon (12 months by default) each one is scored on three questions:
1. Did the Overweight (buy) calls beat the benchmark index (SPY) over the same dates?
2. Were the scenario probabilities well calibrated? The outcome is the scenario the final price landed
   nearest to (below the bear-base midpoint: bear; above the base-bull midpoint: bull; otherwise base).
   Brier score = sum over the three scenarios of (probability - outcome)^2, from 0 (perfect) to 2;
   saying 1/3 each scores 0.667, so a Brier skill above 0 means the probabilities helped.
3. Were high-conviction calls right more often than low-conviction ones? "Right": an Overweight beat the
   index, an Underweight lagged it, an Equal-weight finished within 10 points of it.
Prices are daily closes from Yahoo's public chart endpoint (splits and dividends not adjusted).
"""
import datetime as dt
import json
import pathlib
import statistics
import sys
import urllib.request

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = pathlib.Path(__file__).resolve().parents[1]   # the repo root: .env, prompts/, memos/ and forecasts/ live there
LEDGER = HERE / "forecasts" / "ledger.jsonl"
BENCHMARK = "SPY"
EQUAL_BAND = 10.0           # an Equal-weight call is right if it ends within this many points of the index
_cache = {}


def add_months(d, n):
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    for day in (d.day, 30, 29, 28):
        try:
            return dt.date(y, m, day)
        except ValueError:
            continue


def closes(symbol, start, end):
    """{date: close} for trading days between start and end."""
    key = (symbol, start, end)
    if key not in _cache:
        p1 = int(dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc).timestamp())
        p2 = int(dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc).timestamp()) + 86400
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read())["chart"]["result"][0]
        q = res["indicators"]["quote"][0]["close"]
        _cache[key] = {dt.datetime.fromtimestamp(t, dt.timezone.utc).date(): c
                       for t, c in zip(res.get("timestamp") or [], q) if c is not None}
    return _cache[key]


def close_on(symbol, day):
    """The last close on or before `day` (within a week), or None."""
    got = closes(symbol, day - dt.timedelta(days=8), day)
    before = [d for d in got if d <= day]
    return got[max(before)] if before else None


def outcome_bucket(r, final):
    bear, base, bull = r["bear_value"], r["base_value"], r["bull_value"]
    if final <= (bear + base) / 2:
        return "bear"
    if final >= (base + bull) / 2:
        return "bull"
    return "base"


def brier(r, bucket):
    return sum((r[f"{c}_prob"] / 100 - (1.0 if c == bucket else 0.0)) ** 2 for c in ("bull", "base", "bear"))


def right(rating, excess_pts):
    if rating == "Overweight":
        return excess_pts > 0
    if rating == "Underweight":
        return excess_pts < 0
    return abs(excess_pts) <= EQUAL_BAND


def score(rows, today=None, price=close_on):
    today = today or dt.date.today()
    done, pending = [], []
    for r in rows:
        made = dt.date.fromisoformat(r.get("price_date") or r["date"])
        due = add_months(made, int(r.get("horizon_months", 12)))
        if due > today:
            pending.append((r, due))
            continue
        final = price(r["symbol"], due)
        b0, b1 = price(BENCHMARK, made), price(BENCHMARK, due)
        if final is None or not b0 or not b1:
            pending.append((r, due))
            continue
        ret = 100 * (final / r["price"] - 1)
        bench = 100 * (b1 / b0 - 1)
        row = {**r, "due": due.isoformat(), "final": final, "return_pct": ret, "bench_pct": bench,
               "excess_pts": ret - bench, "right": right(r["rating"], ret - bench)}
        if all(r.get(k) is not None for k in ("bull_value", "base_value", "bear_value", "bull_prob", "base_prob", "bear_prob")):
            row["bucket"] = outcome_bucket(r, final)
            row["brier"] = brier(r, row["bucket"])
        done.append(row)
    return done, pending


def report(done, pending):
    lines = [f"{len(done)} matured, {len(pending)} pending."]
    if pending:
        nxt = sorted(pending, key=lambda x: x[1])[:5]
        lines.append("Next to mature: " + ", ".join(f"{r['symbol']} ({due:%d %b %Y})" for r, due in nxt))
    if not done:
        return "\n".join(lines)
    ow = [d for d in done if d["rating"] == "Overweight"]
    lines += ["", f"1. Overweight calls against {BENCHMARK}:"]
    if ow:
        lines.append(f"   {len(ow)} calls; average return {statistics.fmean(d['return_pct'] for d in ow):+.1f}% against "
                     f"{statistics.fmean(d['bench_pct'] for d in ow):+.1f}% for {BENCHMARK}; beat it "
                     f"{sum(d['excess_pts'] > 0 for d in ow)} of {len(ow)} times.")
    else:
        lines.append("   none matured yet.")
    scored = [d for d in done if "brier" in d]
    lines += ["", "2. Scenario probabilities:"]
    if scored:
        b = statistics.fmean(d["brier"] for d in scored)
        lines.append(f"   Brier {b:.3f} over {len(scored)} forecasts (0 perfect; 1/3 each scores 0.667); "
                     f"skill {1 - b / (2 / 3):+.2f}.")
        for c in ("bull", "base", "bear"):
            said = statistics.fmean(d[f"{c}_prob"] for d in scored)
            got = 100 * statistics.fmean(d["bucket"] == c for d in scored)
            lines.append(f"   {c.title()}: given {said:.0f}% on average, happened {got:.0f}% of the time.")
    else:
        lines.append("   no matured forecast has all three scenario values and probabilities.")
    lines += ["", "3. By conviction (right = Overweight beat the index, Underweight lagged it, Equal-weight within "
              f"{EQUAL_BAND:.0f} points):"]
    for c in ("High", "Medium", "Low"):
        part = [d for d in done if d.get("conviction") == c]
        if part:
            lines.append(f"   {c}: right {sum(d['right'] for d in part)} of {len(part)} "
                         f"({100 * statistics.fmean(d['right'] for d in part):.0f}%).")
    if len(done) < 30:
        lines += ["", f"Only {len(done)} matured: too few to tell skill from luck. Read these as early signs."]
    return "\n".join(lines)


def load(path=LEDGER):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    rows = load(argv[0] if argv else LEDGER)
    if not rows:
        print("No forecasts logged yet. The bot logs each rating it gives on a listed stock to forecasts/ledger.jsonl.")
        return 0
    print(report(*score(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
