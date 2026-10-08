"""Check a memo's KEY NUMBERS block: the maths, the scenario rules, sources, dates and flags.

    python src/verify.py memos/2026-10-06_101500.md      check one saved memo
    python src/verify.py memo.md --json                   the results as JSON

The block is the JSON the system prompt asks for at the end of every full assessment (ACCURACY CHECKS 11).
Every check prints PASS or FAIL with the numbers it compared. Exit code 1 if anything fails.
These checks are mechanical: a block can pass and the analysis still be wrong. They catch arithmetic that
doesn't add up, scenarios that contradict the rating, figures with no source or date, and outliers or
stale data nobody flagged.
"""
import datetime as dt
import json
import re
import sys

REL_TOL = 0.005          # 0.5%: rounding in a memo, not a different number
OVERWEIGHT, UNDERWEIGHT = 15.0, -10.0
CONVICTION_OK = {"Medium", "High"}
RATINGS = {"Overweight", "Equal-weight", "Underweight", "NOT RATED"}


class BlockError(ValueError):
    pass


def extract(memo):
    """The key_numbers object from the last ```json block in the memo."""
    blocks = re.findall(r"```json\s*(.*?)```", memo, re.S | re.I)
    bad = None
    for raw in reversed(blocks):           # an invalid block (an example, a stray snippet) does not hide a valid one
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            bad = bad or e
            continue
        if isinstance(data, dict) and isinstance(data.get("key_numbers"), dict):
            return data["key_numbers"]
    if bad is not None:
        raise BlockError(f"the KEY NUMBERS block is not valid JSON (line {bad.lineno}: {bad.msg})")
    raise BlockError("no ```json block with a \"key_numbers\" object")


def close(a, b, rel=REL_TOL, abs_tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _input(k, name):
    inputs = k.get("inputs")
    v = inputs.get(name) if isinstance(inputs, dict) else None
    return _num(v.get("value")) if isinstance(v, dict) else None


def _trading_days_between(a, b):
    """Weekdays after a, up to and including b (holidays not counted)."""
    if b <= a:
        return 0
    days = (b - a).days
    return sum((a + dt.timedelta(d)).weekday() < 5 for d in range(1, days + 1))


def _date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _flagged(k, *words):
    text = " ".join(str(f) for f in (k.get("flags") or [])).lower()
    return all(w.lower() in text for w in words)


# ----------------------------------------------------------------------------- the checks

def market_cap(price, shares):
    return price * shares


def enterprise_value(mcap, debt, cash, preferred=0.0, minority=0.0):
    return mcap + debt + preferred + minority - cash


def pwv(scen):
    return sum(scen[c]["value"] * scen[c]["prob"] / 100 for c in ("bull", "base", "bear"))


def expected_rating(expected_return_pct, conviction):
    if expected_return_pct >= OVERWEIGHT and conviction in CONVICTION_OK:
        return "Overweight"
    if expected_return_pct <= UNDERWEIGHT:
        return "Underweight"
    return "Equal-weight"


def check(k):
    """[(name, passed, detail)] for every rule that applies to this block."""
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    inputs = k.get("inputs") or {}
    outputs = k.get("outputs") or {}
    # the block is written by a model: a wrong shape is a failed check, never a crash
    add("inputs is an object", isinstance(inputs, dict), type(inputs).__name__)
    add("outputs is an object", isinstance(outputs, dict), type(outputs).__name__)
    inputs = inputs if isinstance(inputs, dict) else {}
    outputs = outputs if isinstance(outputs, dict) else {}
    unknown = set(k.get("unknown") or [])
    sources = k.get("sources") or []
    as_of = _date(k.get("as_of"))
    add("as_of date present", as_of is not None, str(k.get("as_of")))
    subj = k.get("subject") or {}
    add("identity: company, ticker, exchange and share class given",
        all(str(subj.get(f) or "").strip() for f in ("company", "ticker", "exchange", "share_class")),
        json.dumps(subj))

    # A9, A3, A10: every input is sourced and dated, or null and named as unknown
    for name, v in inputs.items():
        if not isinstance(v, dict):
            add(f"input {name} is an object", False, repr(v))
            continue
        val = v.get("value")
        if val is None:
            add(f"{name}: missing value is listed in unknown", name in unknown)
            continue
        add(f"{name}: value is a plain number", _num(val) is not None, repr(val))
        add(f"{name}: has a source", bool(str(v.get("source") or "").strip()))
        add(f"{name}: has an as-of date", _date(v.get("as_of")) is not None, str(v.get("as_of")))
    add("sources are listed", bool(sources), f"{len(sources)} sources")

    # A3: a price more than one trading day old must be flagged stale
    p = inputs.get("price") if isinstance(inputs.get("price"), dict) else None
    if p and _date(p.get("as_of")) and as_of and (p.get("basis") or "last close") != "offer midpoint":
        lag = _trading_days_between(_date(p["as_of"]), as_of)
        if lag > 1:
            add("stale price is flagged", _flagged(k, "stale"), f"price is {lag} trading days old")

    # A4: reconcile
    price, shares = _input(k, "price"), _input(k, "diluted_shares")
    mc = _num(outputs.get("market_cap"))
    if price is not None and shares is not None:
        want = market_cap(price, shares)
        add("market cap = price x fully diluted shares", close(mc, want), f"block {mc}, recomputed {want:,.0f}")
    elif mc is not None:
        add("market cap has both inputs", False, "price or diluted_shares is missing but market_cap is given")
    ev = _num(outputs.get("enterprise_value"))
    debt, cash = _input(k, "debt"), _input(k, "cash")
    if mc is not None and debt is not None and cash is not None:
        want = enterprise_value(mc, debt, cash, _input(k, "preferred") or 0.0, _input(k, "minority_interest") or 0.0)
        add("EV = market cap + debt + preferred + minority interest - cash", close(ev, want),
            f"block {ev}, recomputed {want:,.0f}")
    elif ev is not None:
        add("EV has its inputs", False, "market_cap, debt or cash is missing but enterprise_value is given")
    values = {**{n: _input(k, n) for n in inputs}, "market_cap": mc, "enterprise_value": ev}
    multiples = [m for m in outputs.get("multiples") or [] if isinstance(m, dict)]
    add("every multiple is an object", len(multiples) == len(outputs.get("multiples") or []), "")
    for m in multiples:
        num, den, val = values.get(m.get("numerator")), values.get(m.get("denominator")), _num(m.get("value"))
        if num is None or den in (None, 0):
            add(f"multiple {m.get('name')}: inputs present", False, f"{m.get('numerator')} / {m.get('denominator')}")
        else:
            add(f"multiple {m.get('name')} recomputes", close(val, num / den, rel=0.01),
                f"block {val}, recomputed {num / den:.4g}")
    for seg in [s for s in k.get("segments") or [] if isinstance(s, dict)]:
        total = values.get(seg.get("total"))
        parts = [_num(x.get("value")) if isinstance(x, dict) else None for x in seg.get("parts") or []]
        if total is not None and parts and None not in parts:
            add(f"segments add up to {seg.get('total')}", close(sum(parts), total, rel=0.01),
                f"parts {sum(parts):,.0f}, total {total:,.0f}")

    # A5: one period per multiple
    for m in multiples:
        periods = {str(inputs.get(x, {}).get("period") or "") for x in (m.get("numerator"), m.get("denominator"))
                   if isinstance(inputs.get(x), dict)}
        named = str(m.get("name") or "")                 # "EV/Revenue LTM" promises LTM inputs
        kinds = {w for per in periods | {named} for w in ("LTM", "NTM", "FY") if re.search(rf"\b{w}", per.upper())}
        add(f"multiple {m.get('name')}: one kind of period", len(kinds) <= 1,
            ", ".join(sorted(p for p in periods if p)) or "n/a")

    # A6: outliers must be flagged
    rev, rev0 = _input(k, "revenue"), _input(k, "revenue_prior")
    if ev is not None and rev:
        if ev / rev > 50:
            add("EV/revenue above 50x is flagged", _flagged(k, "outlier"), f"{ev / rev:.1f}x")
    if rev and rev0 and rev0 > 0 and rev / rev0 - 1 > 3:
        add("revenue growth above 300% is flagged", _flagged(k, "outlier"), f"{100 * (rev / rev0 - 1):.0f}%")
    for name in ("net_income", "operating_income", "gross_profit"):
        x = _input(k, name)
        if x is not None and rev:
            margin = x / rev
            if not -1 <= margin <= 1:
                add(f"{name} margin outside -100%..100% is flagged", _flagged(k, "outlier"), f"{100 * margin:.0f}%")

    # A8: scenarios
    scen = k.get("scenarios")
    rating = k.get("rating")
    add("rating is one of the four", rating in RATINGS, str(rating))
    if rating == "NOT RATED":
        add("NOT RATED has no scenarios", not scen)
    elif isinstance(scen, dict):
        try:
            vals = {c: float(scen[c]["value"]) for c in ("bull", "base", "bear")}
            probs = {c: float(scen[c]["prob"]) for c in ("bull", "base", "bear")}
        except (KeyError, TypeError, ValueError):
            add("scenarios have bull, base and bear values and probabilities", False, json.dumps(scen)[:200])
        else:
            add("probabilities add up to 100%", abs(sum(probs.values()) - 100) < 0.01, f"{sum(probs.values())}")
            add("probabilities in 5% steps", all(abs(p / 5 - round(p / 5)) < 1e-9 for p in probs.values()),
                str(list(probs.values())))
            add("bear < base < bull", vals["bear"] < vals["base"] < vals["bull"],
                f"{vals['bear']} / {vals['base']} / {vals['bull']}")
            want = sum(vals[c] * probs[c] / 100 for c in vals)   # from the checked numbers, not the raw text
            add("PWV matches the probabilities and values", close(_num(scen.get("pwv")), want),
                f"block {scen.get('pwv')}, recomputed {want:.4g}")
            ref = _num(scen.get("reference_price"))
            er = _num(scen.get("expected_return_pct"))
            if ref:
                want_er = 100 * (want / ref - 1)
                add("expected return = PWV / reference price - 1", er is not None and abs(er - want_er) < 0.5,
                    f"block {er}, recomputed {want_er:.2f}%")
                for c, v in vals.items():
                    if v > 5 * ref or v < ref / 5:
                        add(f"{c} value beyond 5x / one-fifth of the price is flagged", _flagged(k, "outlier"),
                            f"{v} vs {ref}")
            if er is not None:
                want_r = expected_rating(er, k.get("conviction"))
                add("rating follows the thresholds", rating == want_r,
                    f"expected return {er}%, conviction {k.get('conviction')} -> {want_r}; block says {rating}")
                add("no positive rating with an expected loss", not (er < 0 and rating == "Overweight"))
            ipo = k.get("ipo_ratings")
            if isinstance(ipo, dict):
                add("Participate only with an Overweight rating",
                    (ipo.get("at_offer") == "Participate") == (rating == "Overweight"), json.dumps(ipo))
                bb = _num(ipo.get("buy_below"))
                if ipo.get("aftermarket") == "Buy below":
                    add("buy-below = PWV / 1.15", close(bb, want / 1.15, rel=0.01),
                        f"block {bb}, recomputed {want / 1.15:.4g}")
            stop = _num(k.get("stop_working_price"))
            add("a rating comes with a stop-working price", stop is not None, str(k.get("stop_working_price")))
            if stop is not None and ref:
                add("stop-working price below the reference price", stop < ref, f"{stop} vs {ref}")
    else:
        add("a rated stock has scenarios", False, "scenarios missing")
    return out


def check_memo(memo):
    try:
        k = extract(memo)
    except BlockError as e:
        return [("KEY NUMBERS block present and valid", False, str(e))]
    return [("KEY NUMBERS block present and valid", True, "")] + check(k)


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    bad = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            results = check_memo(f.read())
        bad += sum(not ok for _, ok, _ in results)
        if "--json" in argv:
            print(json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in results], indent=1))
        else:
            print(f"\n{path}")
            for n, ok, d in results:
                print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"  ({d})" if d and not ok else ""))
            print(f"  {sum(ok for _, ok, _ in results)} of {len(results)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
