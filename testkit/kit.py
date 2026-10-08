"""The test kit. Every command that calls the model costs API credit and needs --yes.

Free, offline:
    python testkit/kit.py calc                    calculation cases: the formulas against exact expected outputs
    python testkit/kit.py verify <memo.md>        check a memo's KEY NUMBERS block (same as verify.py)
    python testkit/kit.py track                   score the bot's matured recommendations (same as track.py)
    python testkit/kit.py find-ipos 2026-07-01 2026-09-30   list first-time IPOs from SEC prospectuses for the backtest

Live (the bot answers; add --audit to have the second-pass auditor check every answer too):
    python testkit/kit.py calc --live --yes       the bot does the calculation cases
    python testkit/kit.py golden --yes            the golden set (testkit/golden_set.json, entries you have filled in)
    python testkit/kit.py traps --yes             hallucination traps: the only passing answer is "can't find it"
    python testkit/kit.py stale --yes             tools off: the bot must not give a current price or rate
    python testkit/kit.py rules --yes             hype, inside information, guarantees, pump posts, personal amounts
    python testkit/kit.py consistency --yes       the same question repeated and reworded: ratings and numbers must match
    python testkit/kit.py backtest --yes          IPOs as of the day before pricing, scored on 1, 6 and 12-month prices
    python testkit/kit.py audit <memo.md> [--sources s.json] --yes
    python testkit/kit.py regress --yes [--with-consistency] [--audit]
                                          calc, traps, stale, rules and golden in one run; scores appended to
                                          testkit/history.jsonl with the prompt's fingerprint, compared with last time

Answers and their checks are saved under testkit/runs/<time>/<suite>/.
"""
import datetime as dt
import hashlib
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))   # the bot's modules
import checks  # noqa: E402
import verify  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]   # the repo root
KIT = HERE / "testkit"
RUNS = KIT / "runs"
HISTORY = KIT / "history.jsonl"
MODEL_CUTOFF = "2026-06-30"       # the model's training data runs to June 2026: a clean backtest starts after it
STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
LIVE_REL_TOL = 1e-3               # the bot's numbers against the expected outputs: 0.1%
GOLDEN_TOLERANCE_PCT = 1.0        # a golden figure's default tolerance, when the entry sets none
FULL_MARKS = 0.999                # a score this high counts as 100% (float rounding)
TRAP_FUTURE_DAYS = 60             # {future_date} in traps.json: this many days from today
HORIZONS = (1, 6, 12)             # backtest returns, months after listing
EFTS_PAGE = 100                   # hits per page from SEC full-text search
WHY_CHARS = 400                   # how much of a failure reason a suite item keeps


def load(name):
    return json.loads((KIT / name).read_text(encoding="utf-8"))


def run_dir(suite):
    d = RUNS / STAMP / suite
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompt_fingerprint():
    return hashlib.sha256((HERE / "prompts" / "system_prompt.md").read_bytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- running the bot

class Runner:
    """Asks the bot fresh questions, saves each answer with its checks, and audits on request."""

    def __init__(self, audit=False, client=None):
        import ipo_bot
        import tools
        self.ipo_bot, self.tools = ipo_bot, tools
        self.client = client
        self.audit = audit
        tools.LEDGER = RUNS / STAMP / "ledger.jsonl"     # test forecasts stay out of the real log

    def ask(self, question, use_tools=True, as_of=None):
        bot = self.ipo_bot.Bot(client=self.client, log=lambda s: None, use_tools=use_tools, as_of=as_of)
        self.tools.AS_OF["date"] = as_of
        try:
            memo = bot.ask(question)
        finally:
            self.tools.AS_OF["date"] = None
        verdict = None
        if self.audit:
            import audit
            verdict = audit.audit(memo, bot.sources, client=self.client)
        return memo, bot.sources, verdict

    @staticmethod
    def save(suite, name, question, memo, sources, results, verdict=None):
        d = run_dir(suite)
        name = re.sub(r"[^\w.-]", "_", str(name))      # a ticker such as BRK/B must not become a folder
        lines = "\n".join(f"- [{'x' if ok else ' '}] {n}" for n, ok in results)
        audit_txt = ""
        if verdict:
            import audit
            audit_txt = "\n\nAudit:\n" + audit.report(verdict)
        (d / f"{name}.md").write_text(f"> {question}\n\n{memo}\n\n---\nChecks:\n{lines}{audit_txt}\n", encoding="utf-8")
        (d / f"{name}.sources.json").write_text(json.dumps(sources, indent=1, default=str), encoding="utf-8")


def tally(suite, items):
    passed = sum(i["pass"] for i in items)
    print(f"\n{suite}: {passed} of {len(items)} passed")
    for i in items:
        print(f"  {'PASS' if i['pass'] else 'FAIL'}  {i['id']}" + (f"  {i.get('why', '')}" if not i["pass"] else ""))
    return {"suite": suite, "passed": passed, "total": len(items), "items": items}


def with_audit(item, verdict):
    if verdict is not None:
        item["audit"] = verdict["verdict"]
        if verdict["verdict"] != "pass":
            item["pass"] = False
            # the auditor fails a memo on its own rules or, when those all pass, on a failed mechanical check
            reasons = [f"{c['id']} {c['reason']}" for c in verdict["checks"] if c["result"] == "fail"] or \
                      [m["check"] for m in verdict.get("mechanical", []) if not m["pass"]]
            item["why"] = "; ".join(filter(None, [item.get("why"), "audit failed: " + "; ".join(reasons)]))[:WHY_CHARS]
    return item


# ----------------------------------------------------------------------------- B2 calculations

def compute(case):
    """The expected outputs, computed from the inputs with verify.py's formulas."""
    i, out = case["inputs"], {}
    if "price" in i and "diluted_shares" in i:
        out["market_cap"] = verify.market_cap(i["price"], i["diluted_shares"])
    if "market_cap" in out and "debt" in i and "cash" in i:
        out["enterprise_value"] = verify.enterprise_value(out["market_cap"], i["debt"], i["cash"],
                                                         i.get("preferred", 0), i.get("minority_interest", 0))
    if "enterprise_value" in out and "revenue" in i:
        out["EV/Revenue"] = out["enterprise_value"] / i["revenue"]
    if "enterprise_value" in out and "ebitda" in i:
        out["EV/EBITDA"] = out["enterprise_value"] / i["ebitda"]
    if "market_cap" in out and "net_income" in i:
        out["P/E"] = out["market_cap"] / i["net_income"]
    if "revenue" in i and "revenue_prior" in i:
        out["revenue_growth_pct"] = 100 * (i["revenue"] / i["revenue_prior"] - 1)
    if "revenue" in i and "operating_income" in i:
        out["operating_margin_pct"] = 100 * i["operating_income"] / i["revenue"]
    if "offer_price_low" in i:
        mid = (i["offer_price_low"] + i["offer_price_high"]) / 2
        for k, p in (("low", i["offer_price_low"]), ("mid", mid), ("high", i["offer_price_high"])):
            out[f"market_cap_{k}"] = verify.market_cap(p, i["diluted_shares"])
    if "price_eur" in i:
        out["market_cap_usd"] = i["price_eur"] * i["diluted_shares"] * i["eurusd"]
    if "scenarios" in case:
        s = case["scenarios"]
        out["pwv"] = verify.pwv(s)
        out["expected_return_pct"] = 100 * (out["pwv"] / s["reference_price"] - 1)
        out["rating"] = verify.expected_rating(out["expected_return_pct"], case.get("conviction"))
        out["buy_below"] = out["pwv"] / 1.15
    return out


def same(a, b, rel=1e-9):
    if _number(a) and _number(b):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)
    return not (_number(a) or _number(b)) and a == b     # a rating matches exactly; True is not 1


def calc_offline():
    items = []
    for case in load("calc_cases.json")["cases"]:
        got = compute(case)
        wrong = {k: (got.get(k), v) for k, v in case["expected"].items() if not same(got.get(k), v)}
        items.append({"id": case["id"], "pass": not wrong, "why": f"got vs expected: {wrong}" if wrong else ""})
    return tally("calc (formulas)", items)


def find_value(k, name):
    """A named result anywhere a KEY NUMBERS block might put it."""
    # the block is written by a model: a wrong shape is a missing value, never a crash
    outputs = _obj(k.get("outputs"))
    multiples = outputs.get("multiples") if isinstance(outputs.get("multiples"), list) else []
    found = [outputs.get(name)]
    found += [m.get("value") for m in multiples
              if isinstance(m, dict) and str(m.get("name", "")).lower().startswith(name.lower())]
    found += [_obj(k.get("scenarios")).get(name), k.get(name), _obj(k.get("ipo_ratings")).get(name)]
    return next((v for v in found if v is not None and not isinstance(v, (dict, list))), None)


def _number(x):
    """A real number: True and False are not figures."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _obj(x):
    """x if it is an object, else an empty one."""
    return x if isinstance(x, dict) else {}


def calc_live(runner):
    items = []
    today = dt.date.today().isoformat()
    for case in load("calc_cases.json")["cases"]:
        given = dict(case["inputs"])
        if "scenarios" in case:
            given["scenarios"] = case["scenarios"]
            given["conviction"] = case.get("conviction")     # compute() allows a case without one
        names = ", ".join(case["expected"])
        q = (f"Treat these as figures given by the user (source: user, as of {today}); do not look anything up and "
             f"do not rate a real company: {json.dumps(given)}. Compute {case['ask']}. Use calculate for every result "
             f"and end with the KEY NUMBERS block (subject: \"user example\"), putting each result in \"outputs\" "
             f"under these names: {names}.")
        memo, sources, verdict = runner.ask(q)
        try:
            k = verify.extract(memo)
            # a scenario case may put the rating in "rating"; numbers are compared to 0.1%, ratings exactly
            got = {n: find_value(k, n) for n in case["expected"]}
            wrong = {n: (got[n], v) for n, v in case["expected"].items() if not same(got[n], v, rel=LIVE_REL_TOL)}
            used_tool = any(s["tool"] == "calculate" for s in sources)
            ok = not wrong and used_tool
            why = "; ".join(([f"got vs expected: {wrong}"] if wrong else []) + ([] if used_tool else ["calculate never called"]))
        except verify.BlockError as e:
            ok, why = False, str(e)
        runner.save("calc", case["id"], q, memo, sources, [("results match the expected outputs", ok)], verdict)
        items.append(with_audit({"id": case["id"], "pass": ok, "why": why}, verdict))
    return tally("calc (live)", items)


# ----------------------------------------------------------------------------- B1 golden set

def golden(runner):
    items = []
    for e in load("golden_set.json")["entries"]:
        filled = {f: v for f, v in e["expected"].items() if _number(v)}
        if not filled:
            continue
        as_of = e["as_of"] if e["type"] == "ipo" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e.get("as_of"))) else None
        memo, sources, verdict = runner.ask(e["question"], as_of=as_of)
        try:
            k = verify.extract(memo)
        except verify.BlockError as err:
            runner.save("golden", e["id"], e["question"], memo, sources, [("KEY NUMBERS block", False)], verdict)
            items.append(with_audit({"id": e["id"], "pass": False, "score": 0.0, "why": str(err)}, verdict))
            continue
        inputs = _obj(k.get("inputs"))
        extraction, citation, misses = [], [], []
        for f, want in filled.items():
            given = _obj(inputs.get(f))
            got = given.get("value")
            ok = _number(got) and abs(got - want) <= e.get("tolerance_pct", GOLDEN_TOLERANCE_PCT) / 100 * abs(want)
            extraction.append(ok)
            if not ok:
                misses.append(f"{f}: got {got}, expected {want}")
            token = str(_obj(e.get("expected_source")).get(f) or "")
            if token:
                citation.append(token.lower() in str(given.get("source", "")).lower())
        mech = verify.check(k)
        subj = _obj(k.get("subject"))
        identity = str(subj.get("ticker", "")).upper().replace("-", ".") == e["subject"]["ticker"].upper()
        parts = {"extraction": statistics.fmean(extraction), "math": statistics.fmean(ok for _, ok, _ in mech),
                 "citation": statistics.fmean(citation) if citation else 1.0, "identity": float(identity)}
        score = statistics.fmean(parts.values())
        results = [(f"{n} {v:.0%}", v >= FULL_MARKS) for n, v in parts.items()]
        runner.save("golden", e["id"], e["question"], memo, sources, results, verdict)
        items.append(with_audit({"id": e["id"], "pass": score >= FULL_MARKS, "score": round(score, 3), "parts": parts,
                                 "why": "; ".join(misses)[:WHY_CHARS]}, verdict))
    if not items:
        print("\ngolden: no entries filled in yet. Add the real figures to testkit/golden_set.json first.")
        return {"suite": "golden", "passed": 0, "total": 0, "items": [], "score": None}
    out = tally("golden", items)
    out["score"] = round(statistics.fmean(i["score"] for i in items), 3)
    print(f"  average score {out['score']:.0%} (extraction, maths, citation and identity, equally weighted)")
    return out


# ----------------------------------------------------------------------------- B3, B4, B6

def judged(runner, suite, case, question, memo, sources, results, verdict):
    """Save one answer with its checks; the suite item passes when every check does."""
    runner.save(suite, case["id"], question, memo, sources, results, verdict)
    bad = [n for n, ok in results if not ok]
    name = f"{case['id']} {case['kind']}" if "kind" in case else case["id"]
    return with_audit({"id": name, "pass": not bad, "why": "; ".join(bad)}, verdict)


def traps(runner):
    items = []
    future = (dt.date.today() + dt.timedelta(days=TRAP_FUTURE_DAYS)).strftime("%d %B %Y")
    for case in load("traps.json")["cases"]:
        q = case["question"].replace("{future_date}", future)
        memo, sources, verdict = runner.ask(q)
        results = checks.check(memo, {"cant_find"})
        for pat in case.get("forbid", []):
            results.append((f"no invented number ({pat})", not re.search(pat, memo)))
        try:
            k = verify.extract(memo)
            results.append(("KEY NUMBERS, if any, rate it NOT RATED", k.get("rating") == "NOT RATED"))
        except verify.BlockError:
            pass
        items.append(judged(runner, "traps", case, q, memo, sources, results, verdict))
    return tally("traps", items)


def stale(runner):
    items = []
    for case in load("stale.json")["cases"]:
        memo, sources, verdict = runner.ask(case["question"], use_tools=False)
        results = checks.check(memo, {"no_live_figures"})
        items.append(judged(runner, "stale", case, case["question"], memo, sources, results, verdict))
    return tally("stale", items)


def rules(runner):
    items = []
    for case in load("rules.json")["cases"]:
        memo, sources, verdict = runner.ask(case["question"])
        results = checks.check(memo, set(case["expect"]))
        items.append(judged(runner, "rules", case, case["question"], memo, sources, results, verdict))
    return tally("rules", items)


# ----------------------------------------------------------------------------- B5 consistency

def spread_pct(xs):
    xs = [x for x in xs if _number(x)]
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    return 100 * (max(xs) - min(xs)) / abs(m) if m else None


def consistency(runner):
    spec = load("consistency.json")
    qs = [spec["question"]] * spec["repeats"] + spec["rewordings"]
    rows = []
    for n, q in enumerate(qs, 1):
        memo, sources, verdict = runner.ask(q)
        try:
            k = verify.extract(memo)
            scen = _obj(k.get("scenarios"))
            rows.append({"rating": k.get("rating"), "pwv": scen.get("pwv"), "er": scen.get("expected_return_pct"),
                         "stop": k.get("stop_working_price")})
        except verify.BlockError as e:
            rows.append({"rating": None, "pwv": None, "er": None, "stop": None, "error": str(e)})
        runner.save("consistency", f"run{n}", q, memo, sources, [("KEY NUMBERS block", "error" not in rows[-1])], verdict)
    tol = spec["tolerance"]
    ratings = {r["rating"] for r in rows}
    er = [r["er"] for r in rows if _number(r["er"])]
    pwv_s, stop_s = spread_pct([r["pwv"] for r in rows]), spread_pct([r["stop"] for r in rows])
    items = [
        {"id": "same rating every time", "pass": len(ratings) == 1 and None not in ratings, "why": str(sorted(map(str, ratings)))},
        {"id": f"PWV within {tol['pwv_pct']}%", "pass": pwv_s is not None and pwv_s <= tol["pwv_pct"],
         "why": f"spread {pwv_s if pwv_s is None else round(pwv_s, 1)}%"},
        {"id": f"expected return within {tol['expected_return_pts']} points",
         "pass": bool(er) and len(er) == len(rows) and max(er) - min(er) <= tol["expected_return_pts"],
         "why": f"{min(er) if er else None} to {max(er) if er else None}"},
        {"id": f"stop-working price within {tol['stop_working_pct']}%", "pass": stop_s is not None and stop_s <= tol["stop_working_pct"],
         "why": f"spread {stop_s if stop_s is None else round(stop_s, 1)}%"},
    ]
    out = tally("consistency", items)
    print("  runs: " + "; ".join(f"{r['rating']} PWV {r['pwv']} ER {r['er']} stop {r['stop']}" for r in rows))
    return out


# ----------------------------------------------------------------------------- B7 backtest

def find_ipos(start, end):
    """First-time IPOs with a final prospectus (424B4) filed between the dates: blank-check companies and
    companies that already filed annual or quarterly reports (follow-on offerings) are left out."""
    import tools
    url = f"https://efts.sec.gov/LATEST/search-index?forms=424B4&dateRange=custom&startdt={start}&enddt={end}"
    hits, seen = [], {}
    while True:
        data = json.loads(tools._get(url + f"&from={len(hits)}", tools._sec_headers()))
        batch = data.get("hits", {}).get("hits", [])
        hits += batch
        if len(batch) < EFTS_PAGE:
            break
    for h in hits:
        s = h["_source"]
        name = (s.get("display_names") or [""])[0]
        if re.search(r"(?i)acquisition|\bspac\b|capital corp\.? [ivx]+\b", name):
            continue
        cik = (s.get("ciks") or [""])[0]
        if cik and (cik not in seen or s["file_date"] < seen[cik]["prospectus_date"]):
            m = re.match(r"(.*?)\s+\(([A-Z0-9., ]+)\)\s+\(CIK", name)
            seen[cik] = {"company": (m.group(1) if m else name.split("(CIK")[0]).strip(),
                         "ticker": (m.group(2).split(",")[0].strip() if m else ""), "cik": cik.lstrip("0"),
                         "prospectus_date": s["file_date"]}
    out = []
    for cik, e in sorted(seen.items(), key=lambda kv: kv[1]["prospectus_date"]):
        sub = json.loads(tools._get(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json", tools._sec_headers()))
        if str(sub.get("sic")) == "6770":
            continue                                   # SIC 6770: a blank-check company (SPAC)
        recent = sub.get("filings", {}).get("recent", {})
        earlier = [d for f, d in zip(recent.get("form", []), recent.get("filingDate", []))
                   if f in ("10-K", "10-Q", "20-F", "40-F") and d < e["prospectus_date"]]
        if earlier or sub.get("filings", {}).get("files"):
            continue                                   # already reporting: a follow-on, not an IPO
        d = dt.date.fromisoformat(e["prospectus_date"])
        out.append({**e, "as_of": (d - dt.timedelta(days=1)).isoformat(), "listing_date": None, "offer_price": None,
                    "verified": False})
    return out


def find_ipos_cmd(argv):
    start, end = argv[0], argv[1]
    found = find_ipos(start, end)
    path = KIT / "backtest_ipos.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
    have = {e["cik"] for e in doc["entries"]}
    new = [e for e in found if e["cik"] not in have]
    doc["entries"] += new
    doc["_about"] = BACKTEST_ABOUT
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{len(found)} first-time IPOs with a 424B4 from {start} to {end}; {len(new)} added to {path.name}.")
    for e in new:
        print(f"  {e['prospectus_date']}  {e['ticker'] or '?':<6} {e['company']}")
    print("Fill in listing_date and offer_price from each 424B4, and set verified to true.")


BACKTEST_ABOUT = ("IPOs for the no-hindsight backtest. 'as_of' is the day before the final prospectus (424B4), so the bot "
                  "sees the price range but not the final price; its data tools are cut off at that date and web search is "
                  "off. Only IPOs after the model's training cutoff (" + MODEL_CUTOFF + ") are clean: for earlier ones "
                  "the model may remember what happened. Fill in listing_date and offer_price from the 424B4 and set "
                  "verified to true; unverified entries are run but not scored.")


def backtest(runner, allow_hindsight=False):
    path = KIT / "backtest_ipos.json"
    if not path.exists():
        sys.exit("No testkit/backtest_ipos.json yet: run  python testkit/kit.py find-ipos <start> <end>  first.")
    today = dt.date.today()
    items, rows = [], []
    for e in json.loads(path.read_text(encoding="utf-8"))["entries"]:
        if e["as_of"] <= MODEL_CUTOFF and not allow_hindsight:
            print(f"  skip {e['ticker']}: as of {e['as_of']}, inside the model's training data (use --allow-hindsight)")
            continue
        item, row = backtest_answer(runner, e)
        if e.get("verified") and e.get("offer_price") and e.get("listing_date"):
            row.update(after_listing(e, today))
        items.append(item)
        rows.append(row)
    out = tally("backtest (answers)", items)
    print(backtest_table(rows))
    out["rows"] = rows
    return out


def backtest_answer(runner, e):
    """The bot's call on one IPO as of the day before its prospectus: (suite item, table row)."""
    name = e["ticker"] or e["cik"]
    q = (f"Rate the {e['company']} ({e['ticker'] or 'ticker not yet assigned'}) IPO. Use only information available "
         f"on or before {e['as_of']}: the SEC filings up to that date. Give both IPO ratings and the KEY NUMBERS block.")
    memo, sources, verdict = runner.ask(q, as_of=e["as_of"])
    try:
        k = verify.extract(memo)
    except verify.BlockError:
        k = {}
    call = _obj(k.get("ipo_ratings"))
    ok = bool(k) and bool(call.get("at_offer"))
    results = ([("both IPO ratings in KEY NUMBERS", ok)] + [(n, ok2) for n, ok2, _ in verify.check(k)] if k
               else [("KEY NUMBERS block", False)])
    runner.save("backtest", name, q, memo, sources, results, verdict)
    row = {"ticker": e["ticker"], "as_of": e["as_of"], "at_offer": call.get("at_offer"), "buy_below": call.get("buy_below"),
           "rating": k.get("rating")}
    return with_audit({"id": name, "pass": ok, "why": "" if ok else "no IPO ratings"}, verdict), row


def after_listing(e, today):
    """Returns from the offer price, and the S&P 500's from the day before listing, at each horizon that has passed."""
    import track
    listed = dt.date.fromisoformat(e["listing_date"])
    row = {"first_close": track.close_on(e["ticker"], listed)}
    spy0 = track.close_on(track.BENCHMARK, listed - dt.timedelta(days=1))
    for m in HORIZONS:
        when = track.add_months(listed, m)
        if when > today:
            row[f"{m}m"] = None
            continue
        px, spy = track.close_on(e["ticker"], when), track.close_on(track.BENCHMARK, when)
        row[f"{m}m"] = None if not px else round(100 * (px / e["offer_price"] - 1), 1)
        row[f"{m}m_spy"] = None if not (spy and spy0) else round(100 * (spy / spy0 - 1), 1)
    return row


def backtest_table(rows):
    """The calls and what followed, as a Markdown table, then the average return of each call at each horizon."""
    def pct(x):
        return "-" if x is None else f"{x:+.1f}%"
    lines = ["", "| IPO | as of | at the offer | buy below | 1m | 6m | 12m | S&P 500 1m / 6m / 12m |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        own = " | ".join("pending" if r.get(f"{m}m") is None else pct(r[f"{m}m"]) for m in HORIZONS)
        spy = " / ".join(pct(r.get(f"{m}m_spy")) for m in HORIZONS)
        lines.append(f"| {r['ticker']} | {r['as_of']} | {r['at_offer']} | {r['buy_below']} | {own} | {spy} |")
    for m in HORIZONS:
        for call in ("Participate", "Pass"):
            xs = [r[f"{m}m"] for r in rows if r["at_offer"] == call and r.get(f"{m}m") is not None]
            if xs:
                lines.append(f"  {call} calls, {m} months after listing: {statistics.fmean(xs):+.1f}% from the offer price "
                             f"(n = {len(xs)})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- B10 regression

def regress(runner, with_consistency=False):
    suites = [calc_offline(), calc_live(runner), traps(runner), stale(runner), rules(runner)]
    g = golden(runner)
    if g["total"]:
        suites.append(g)
    if with_consistency:
        suites.append(consistency(runner))
    entry = {"date": dt.datetime.now().isoformat(timespec="seconds"), "prompt": prompt_fingerprint(),
             "model": runner.ipo_bot.MODEL, "audited": runner.audit,
             "suites": {s["suite"]: {"passed": s["passed"], "total": s["total"], **({"score": s["score"]} if s.get("score") is not None else {})}
                        for s in suites}}
    entry["passed"] = sum(s["passed"] for s in suites)
    entry["total"] = sum(s["total"] for s in suites)
    previous = last_history()
    # a file cut off mid-line (an interrupted write, a hand edit) must not swallow this entry into that line
    gap = "\n" if HISTORY.exists() and HISTORY.read_bytes()[-1:] not in (b"", b"\n") else ""
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(gap + json.dumps(entry) + "\n")
    print(f"\nRegression: {entry['passed']} of {entry['total']} passed (prompt {entry['prompt']}).")
    if previous:
        print(f"Last run {previous.get('date')} (prompt {previous.get('prompt')}): "
              f"{previous.get('passed')} of {previous.get('total')}.")
        for name, now in entry["suites"].items():
            before = _obj(_obj(previous.get("suites")).get(name))
            if before and (before.get("passed"), before.get("total")) != (now["passed"], now["total"]):
                print(f"  {name}: {before.get('passed')}/{before.get('total')} -> {now['passed']}/{now['total']}")
    return entry


def last_history():
    """The last readable entry in history.jsonl, or None: a damaged line must not cost the run just paid for."""
    if not HISTORY.exists():
        return None
    for line in reversed(HISTORY.read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:             # a blank, cut-off or hand-edited line
            continue
        if isinstance(entry, dict):
            return entry
    return None


# ----------------------------------------------------------------------------- command line

LIVE = {"golden", "traps", "stale", "rules", "consistency", "backtest", "regress", "audit"}


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if not argv:
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    from ipo_bot import load_env
    load_env()                       # SEC_USER_AGENT and the API keys from .env
    live = cmd in LIVE or (cmd == "calc" and "--live" in rest)
    if live and "--yes" not in rest:
        sys.exit(f"'{cmd}' calls the model and costs API credit. Add --yes to go ahead.")
    if cmd == "calc" and not live:
        return all_passed(calc_offline())
    if cmd == "verify":
        return verify.main(rest)
    if cmd == "track":
        import track
        return track.main(rest)
    if cmd == "find-ipos":
        if len(rest) < 2:
            sys.exit("find-ipos needs a start and an end date, e.g. find-ipos 2026-07-01 2026-09-30")
        find_ipos_cmd(rest)
        return 0
    if cmd == "audit":
        import audit
        return audit.main([a for a in rest if a != "--yes"])
    suites = {"calc": calc_live, "golden": golden, "traps": traps, "stale": stale, "rules": rules,
              "consistency": consistency,
              "backtest": lambda r: backtest(r, allow_hindsight="--allow-hindsight" in rest),
              "regress": lambda r: regress(r, with_consistency="--with-consistency" in rest)}
    if cmd not in suites:
        sys.exit(f"unknown command {cmd!r}\n\n{__doc__}")
    import anthropic
    try:
        result = suites[cmd](Runner(audit="--audit" in rest))
    except anthropic.AuthenticationError:
        sys.exit("Claude rejected the API key: set ANTHROPIC_API_KEY or run `ant auth login`.")
    print(f"\nAnswers saved under {RUNS / STAMP}")     # regress too: its answers are saved the same way
    return all_passed(result)


def all_passed(result):
    """The exit code: 0 when every item passed (a golden set with nothing filled in has nothing to fail)."""
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":  # pragma: no cover - runs only as a script; main() itself is tested
    sys.exit(main(sys.argv[1:]))
