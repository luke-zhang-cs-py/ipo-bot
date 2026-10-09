"""The browser demo (docs/app/ipo-core.js) against the Python it ports: portfolio sizing and the KEY NUMBERS checks.

Runs the same inputs through both and fails on any difference, so the demo cannot drift from the bot. Needs Node
(set IPO_BOT_NODE to its path if it is not on PATH); skipped without it.
"""
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import portfolio  # noqa: E402
import verify  # noqa: E402
from test_accuracy import BLOCK, memo_with  # noqa: E402

NODE = os.environ.get("IPO_BOT_NODE") or shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="Node is not installed")
CORE = ROOT / "docs" / "app" / "ipo-core.js"
EXAMPLE = json.loads((ROOT / "portfolio.example.json").read_text(encoding="utf-8"))

DRIVER = """
const core = require(process.argv[1]);
let input = '';
process.stdin.on('data', (d) => input += d).on('end', () => {
  const job = JSON.parse(input), out = {};
  out.sizes = job.sizes.map(([pf, args]) => {
    try { return core.sizePosition(core.valued(core.validate(pf)), ...args); } catch (e) { return { error: e.message }; }
  });
  out.views = job.views.map((pf) => { const v = core.valued(core.validate(pf)); return { equity: v.equity, breaches: v.rule_breaches }; });
  out.memos = job.memos.map((m) => core.checkMemo(m).map(([name, ok]) => [name, ok]));
  process.stdout.write(JSON.stringify(out));
});
"""


def run_js(job):
    r = subprocess.run([NODE, "-e", DRIVER, str(CORE)], input=json.dumps(job), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def py_size(pf, args):
    try:
        return portfolio.size_position(portfolio.valued(portfolio.validate(copy.deepcopy(pf))), *args)
    except portfolio.PortfolioError as e:
        return {"error": str(e)}


SIZES = [
    (EXAMPLE, ["DDD", 50, 45, "Medium", "Energy"]), (EXAMPLE, ["DDD", 50, 45, "Low", "Energy"]),
    (EXAMPLE, ["DDD", 50, 45, "High", "Energy"]), (EXAMPLE, ["AAA", 180, 170, "High", None]),
    (EXAMPLE, ["EEE", 10, 9.9, "High", "Technology"]), ({**EXAMPLE, "cash": 600}, ["DDD", 50, 49.9, "High", "Energy"]),
    (EXAMPLE, ["DDD", 50, 55, "Medium", None]), (EXAMPLE, ["DDD", 50, 45, "Certain", None]),
    (EXAMPLE, ["BBB", 55, 44, "Medium", None]), ({**EXAMPLE, "cash": 0}, ["ZZZ", 12.5, 10, "High", "Health"]),
    (EXAMPLE, ["DDD", 50, 45, "toString", None]), (EXAMPLE, ["DDD", 50, 45, "Medium", "constructor"]),
    ({**EXAMPLE, "holdings": EXAMPLE["holdings"] + [{"symbol": "UNP", "shares": 5, "sector": "Energy"}]}, ["unp", 50, 45, "Medium", None]),
]


GUARDED = {**EXAMPLE, "day_pnl": -100.0, "rules": {**EXAMPLE["rules"], "max_daily_loss_pct": 2, "earnings_blackout_hours": 48, "max_volume_pct": 1}}
GUARD_CASES = [
    (GUARDED, ["DDD", 50, 45, "Medium", "Energy"], {"next_earnings": "2026-10-10", "avg_volume": 900_000, "now": "2026-10-07T12:00:00Z"}),
    (GUARDED, ["DDD", 50, 45, "Medium", "Energy"], {"next_earnings": "2026-10-08T20:00:00Z", "avg_volume": 900_000, "now": "2026-10-07T12:00:00Z"}),
    (GUARDED, ["DDD", 50, 45, "Medium", "Energy"], {"next_earnings": "2026-10-01", "avg_volume": 2_000, "now": "2026-10-07T12:00:00Z"}),
    (GUARDED, ["DDD", 50, 45, "Medium", "Energy"], {}),
    ({**GUARDED, "day_pnl": -400.0}, ["DDD", 50, 45, "Medium", "Energy"], {"next_earnings": "2026-12-01", "avg_volume": 900_000, "now": "2026-10-07"}),
    ({**GUARDED, "day_pnl": 250.0}, ["DDD", 50, 45, "High", "Energy"], {"avg_volume": 600, "now": "2026-10-07"}),
    ({**EXAMPLE, "rules": {**EXAMPLE["rules"], "max_order_value": 500}}, ["DDD", 50, 45, "High", "Energy"], {}),
    ({**EXAMPLE, "rules": {**EXAMPLE["rules"], "max_gross_exposure_pct": 70}}, ["DDD", 50, 45, "High", "Energy"], {}),
    ({**EXAMPLE, "equity_high": 20_000, "rules": {**EXAMPLE["rules"], "max_drawdown_pct": 8}}, ["DDD", 50, 45, "Medium", "Energy"], {}),
    ({**EXAMPLE, "rules": {**EXAMPLE["rules"], "max_drawdown_pct": 8}}, ["DDD", 50, 45, "Medium", "Energy"], {}),
    ({**EXAMPLE, "halted": True}, ["DDD", 50, 45, "Medium", "Energy"], {}),
]
GUARD_DRIVER = """
const core = require(process.argv[1]);
let input = '';
process.stdin.on('data', (d) => input += d).on('end', () => {
  const out = JSON.parse(input).map(([pf, args, opts]) => {
    try { return core.sizePosition(core.valued(core.validate(pf)), ...args, opts); } catch (e) { return { error: e.message }; }
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def test_the_guardrails_match_the_python():
    r = subprocess.run([NODE, "-e", GUARD_DRIVER, str(CORE)], input=json.dumps(GUARD_CASES), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for (pf, args, opts), got in zip(GUARD_CASES, json.loads(r.stdout)):
        view = portfolio.valued(portfolio.validate(copy.deepcopy(pf)))
        want = portfolio.size_position(view, *args, **opts)
        for key in ("shares", "binding_rule", "limits_in_shares", "warnings"):
            assert got[key] == want[key], (opts, key, got[key], want[key])


def broken_blocks():
    def edit(fn):
        b = copy.deepcopy(BLOCK)
        fn(b["key_numbers"])
        return b
    yield BLOCK
    yield edit(lambda k: k["outputs"].update(market_cap=6_000_000_000))
    yield edit(lambda k: k["outputs"].update(enterprise_value=5_000_000_000))
    yield edit(lambda k: k["outputs"]["multiples"][0].update(value=7.0))
    yield edit(lambda k: k["scenarios"]["bull"].update(prob=35))
    yield edit(lambda k: k["scenarios"]["bear"].update(value=70))
    yield edit(lambda k: k["scenarios"].update(pwv=70))
    yield edit(lambda k: k.update(rating="Equal-weight"))
    yield edit(lambda k: k.update(conviction="Low"))
    yield edit(lambda k: k["inputs"]["debt"].update(value=None))
    yield edit(lambda k: k["inputs"]["price"].update(as_of="2026-10-01"))
    yield edit(lambda k: k["inputs"]["revenue"].update(period="NTM to 2027-06-30"))
    yield edit(lambda k: k["segments"][0]["parts"][0].update(value=600_000_000))
    yield edit(lambda k: k.pop("stop_working_price"))
    yield edit(lambda k: k.update(rating="NOT RATED", scenarios=None))
    yield edit(lambda k: k.update(ipo_ratings={"at_offer": "Participate", "aftermarket": "Buy below", "buy_below": 62 / 1.15}))
    yield edit(lambda k: (k["inputs"]["revenue"].update(value=100_000_000), k["outputs"]["multiples"][0].update(value=55.0)))
    # scenario numbers go through Python's float(): null, "" and lists fail one check; numeric strings and booleans convert
    for bad in (None, "", [], {}, "80", " 8e1 ", True, "eighty"):
        yield edit(lambda k, bad=bad: k["scenarios"]["bull"].update(value=bad))
    yield edit(lambda k: k["scenarios"]["base"].pop("prob"))
    yield edit(lambda k: k["scenarios"].update(bear=35))
    # a badly shaped block is a list of failed checks, never a crash
    # a lone value in a list field is a list of one (not its characters); null, "" and {} are an empty list
    yield edit(lambda k: (k["inputs"]["price"].update(as_of="2026-10-01"), k.update(flags="stale")))
    yield edit(lambda k: (k["inputs"]["price"].update(as_of="2026-10-01"), k.update(flags="")))
    yield edit(lambda k: k.update(sources="the S-1"))
    yield edit(lambda k: k.update(sources={}))
    yield edit(lambda k: k.update(sources=0))
    yield edit(lambda k: (k["inputs"]["debt"].update(value=None), k.update(unknown="debt")))
    yield edit(lambda k: (k["inputs"]["debt"].update(value=None), k.update(unknown="de")))
    yield edit(lambda k: k.update(subject="ACME Corp"))
    yield edit(lambda k: k.update(subject=["ACME"]))
    yield edit(lambda k: k.update(subject=""))
    yield edit(lambda k: k["outputs"]["multiples"][0].update(numerator=["enterprise_value"]))
    yield edit(lambda k: k["outputs"]["multiples"][0].update(denominator={"name": "revenue"}))
    yield edit(lambda k: k["outputs"].update(multiples={"name": "EV/Revenue", "numerator": "enterprise_value", "denominator": "revenue", "value": 1.0}))
    yield edit(lambda k: k["outputs"].update(multiples="EV/Revenue"))
    yield edit(lambda k: k["segments"][0].update(total=["revenue"]))
    yield edit(lambda k: k.update(rating=["Overweight"]))
    yield edit(lambda k: k.update(conviction=["Medium"]))
    yield edit(lambda k: (k["inputs"]["price"].update(as_of="2026-10-01"), k.update(flags=[{"note": "stale price"}])))
    yield edit(lambda k: k["outputs"].update(multiples={}))
    yield edit(lambda k: k["outputs"].update(multiples=["EV/Revenue", {"name": None, "numerator": "toString", "denominator": "revenue"}]))
    yield edit(lambda k: k.update(inputs=[1, 2]))
    yield edit(lambda k: k.update(outputs="none"))
    yield edit(lambda k: k.update(segments=[{"total": "revenue", "parts": "ab"}, "x"]))
    yield edit(lambda k: k.update(scenarios=[1, 2, 3]))
    yield edit(lambda k: k.update(ipo_ratings=["Participate"]))
    yield edit(lambda k: k.update(rating="NOT RATED", scenarios={}))
    # a source is what Python's str() makes of it: {} is nothing, [" "] is the text "[' ']"
    for bad in ({}, [" "], [], " ", 0, 1.0, ["[1] quote"]):
        yield edit(lambda k, bad=bad: k["inputs"]["debt"].update(source=bad))
    # a date must be written YYYY-MM-DD: not in a list, not the compact or week forms fromisoformat also takes
    for bad in (["2026-10-06"], "20261006", "2026-W41-1", "2026-02-30", "0000-10-06", "2026-13-01", "2026-10-06T09:30:00Z",
                20261006, None, "", "0999-12-31"):
        yield edit(lambda k, bad=bad: k.update(as_of=bad))
        yield edit(lambda k, bad=bad: k["inputs"]["revenue"].update(as_of=bad))
        yield edit(lambda k, bad=bad: k["inputs"]["price"].update(as_of=bad))
    # a multiple's name is str() of whatever was written, for the period check and in every check name
    for name in ({"p": "NTM"}, ["EV/Revenue", "NTM"], 1.0, 2.5, 1e16, 1.5e-5, -0.0, 7, True, None, "",
                 ["x", 1.0, None, True, False, "it's", 'say "hi"', "both ' and \"", "a\nb\\c\t", "\xa0", "\u200b", {"a": [1.0, 2]}]):
        yield edit(lambda k, name=name: k["outputs"]["multiples"][0].update(name=name))
    # unknown holds str() of each entry: 1.0 names the input "1.0", not "1"
    yield edit(lambda k: (k["inputs"].update({"1.0": {"value": None}}), k.update(unknown=[1.0])))
    yield edit(lambda k: (k["inputs"].update({"1.0": {"value": None}}), k.update(unknown=1.0)))
    yield edit(lambda k: (k["inputs"].update({"['debt']": {"value": None}}), k.update(unknown=[["debt"]])))
    yield edit(lambda k: (k["inputs"].update({"{'a': 1.0}": {"value": None}}), k.update(unknown=[{"a": 1.0}])))
    yield edit(lambda k: (k["inputs"]["price"].update(as_of="2026-10-01"), k.update(flags=[["stale", 1.0]])))


def test_sizing_matches_the_python_to_the_cent():
    js = run_js({"sizes": SIZES, "views": [], "memos": []})["sizes"]
    for (pf, args), got in zip(SIZES, js):
        want = py_size(pf, args)
        if "error" in want:
            assert "error" in got, (args, got)
            continue
        for key in ("shares", "binding_rule", "limits_in_shares"):
            assert got[key] == want[key], (args, key, got[key], want[key])
        for key in ("cost", "risk_budget", "loss_at_stop", "weight_after_pct", "cash_after"):
            assert abs(got[key] - want[key]) <= 0.011, (args, key, got[key], want[key])


def test_portfolio_values_and_breaches_match():
    pfs = [EXAMPLE, {**EXAMPLE, "cash": 50}, {**EXAMPLE, "rules": {**EXAMPLE["rules"], "max_sector_pct": 20}}]
    js = run_js({"sizes": [], "views": pfs, "memos": []})["views"]
    for pf, got in zip(pfs, js):
        v = portfolio.valued(portfolio.validate(copy.deepcopy(pf)))
        assert abs(got["equity"] - v["equity"]) < 1e-6
        assert got["breaches"] == v["rule_breaches"]


def test_every_memo_check_matches_the_python():
    memos = [memo_with(b) for b in broken_blocks()] + ["no block here", "```json\n{oops\n```",
             memo_with(BLOCK) + "\n```json\n{oops\n```"]          # a stray invalid snippet does not hide the real block
    js = run_js({"sizes": [], "views": [], "memos": memos})["memos"]
    for memo, got in zip(memos, js):
        want = [[name, ok] for name, ok, _ in verify.check_memo(memo)]
        assert got == want, (memo[:80], [x for x in got if x not in want], [x for x in want if x not in got])


# ----------------------------------------------------------------------------- projection charts (JS only)
PROJECT = """
const core = require(process.argv[1]);
let input = '';
process.stdin.on('data', (d) => input += d).on('end', () => {
  const job = JSON.parse(input);
  const out = {
    ratings: job.ratings.map(([er, conv]) => core.expectedRating(er, conv)),
    thresholds: core.THRESHOLDS,
    runs: job.runs.map((o) => { try { return core.project(o); } catch (e) { return { error: e.message }; } }),
  };
  process.stdout.write(JSON.stringify(out));
});
"""
SCEN = {"price": 50, "bull": {"value": 80, "prob": 30}, "base": {"value": 62, "prob": 50}, "bear": {"value": 35, "prob": 20},
        "stop": 40, "months": 12, "vol": 35, "conviction": "Medium"}


def run_project(job):
    r = subprocess.run([NODE, "-e", PROJECT, str(CORE)], input=json.dumps(job), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_the_charts_rate_by_the_bots_own_thresholds():
    grid = [[er, conv] for er in (-30, -10.01, -10, -9.99, 0, 14.99, 15, 15.01, 40) for conv in ("Low", "Medium", "High")]
    out = run_project({"ratings": grid, "runs": []})
    assert out["ratings"] == [verify.expected_rating(er, conv) for er, conv in grid]
    t = out["thresholds"]                                   # the numbers the page draws its gauge and notes from
    assert (t["overweight"], t["underweight"], set(t["conviction_ok"])) == (verify.OVERWEIGHT, verify.UNDERWEIGHT, verify.CONVICTION_OK)


def test_the_projection_lands_on_the_scenarios_and_the_pwv():
    a, b, other = run_project({"ratings": [], "runs": [SCEN, SCEN, {**SCEN, "seed": 2}]})["runs"]
    assert a == b and a["bands"] != other["bands"]                        # the seed fixes the paths
    want = verify.pwv({c: SCEN[c] for c in ("bull", "base", "bear")})
    assert abs(a["pwv"] - want) < 1e-9 and abs(a["weighted"][-1] - want) < 1e-9
    assert abs(a["expected_return_pct"] - 100 * (want / 50 - 1)) < 1e-9
    assert a["rating"] == verify.expected_rating(a["expected_return_pct"], "Medium")
    for c in ("bull", "base", "bear"):
        assert abs(a["scenario"][c][0] - 50) < 1e-9 and abs(a["scenario"][c][-1] - SCEN[c]["value"]) < 1e-9
    assert abs(a["mean_end"] / want - 1) < 0.03                            # the paths average out at the PWV
    assert all(abs(n / 2000 - SCEN[c]["prob"] / 100) < 0.04 for c, n in a["picks"].items())


def test_the_bands_are_ordered_and_start_at_todays_price():
    run = run_project({"ratings": [], "runs": [SCEN]})["runs"][0]
    bands = [run["bands"][k] for k in ("p10", "p25", "p50", "p75", "p90")]
    assert all(abs(b[0] - 50) < 1e-9 for b in bands)
    for i in range(len(run["t"])):
        assert all(lo <= hi + 1e-9 for lo, hi in zip([b[i] for b in bands], [b[i] for b in bands][1:]))
    assert run["p_touch_stop"] >= run["p_end_below_stop"]                  # ending below the stop means it was touched
    assert sum(run["histogram"]["counts"]) >= 0.97 * 2000
    assert run["range80"][0] < run["pwv"] < run["range80"][1]


def test_more_volatility_widens_the_bands_and_bad_input_is_refused():
    calm, wild, *bad = run_project({"ratings": [], "runs": [
        {**SCEN, "vol": 15}, {**SCEN, "vol": 70},
        {**SCEN, "bull": {"value": 80, "prob": 35}}, {**SCEN, "bear": {"value": 70, "prob": 20}},
        {**SCEN, "price": 0}, {**SCEN, "months": 0}]})["runs"]
    width = lambda r: r["bands"]["p90"][len(r["t"]) // 2] - r["bands"]["p10"][len(r["t"]) // 2]
    assert width(wild) > width(calm) and wild["p_touch_stop"] > calm["p_touch_stop"]
    assert [("error" in r) for r in bad] == [True] * 4
    assert "100%" in bad[0]["error"] and "bear < base < bull" in bad[1]["error"]


# ----------------------------------------------------------------------------- sorting trade options (JS only)
SORT = """
const core = require(process.argv[1]);
let input = '';
process.stdin.on('data', (d) => input += d).on('end', () => {
  const job = JSON.parse(input), view = core.valued(core.validate(job.portfolio));
  const rows = job.ideas.map((i) => core.tradeMetrics(i, view));
  const out = { rows: rows.map(({ idea, ...r }) => r), sorts: {} };
  for (const key of Object.keys(core.TRADE_SORTS)) for (const dir of ['best', 'worst'])
    out.sorts[key + ':' + dir] = core.sortTrades(rows, key, dir).map((r) => r.symbol);
  try { core.sortTrades(rows, 'nonsense'); } catch (e) { out.bad = e.message; }
  process.stdout.write(JSON.stringify(out));
});
"""
IDEAS = [
    {**SCEN, "symbol": "EXM", "sector": "Technology"},
    {**SCEN, "symbol": "NEWCO", "sector": "Technology", "price": 28, "stop": 21, "vol": 75,
     "bull": {"value": 52, "prob": 25}, "base": {"value": 31, "prob": 45}, "bear": {"value": 16, "prob": 30}},
    {**SCEN, "symbol": "UTIL", "sector": "Utilities", "price": 64, "stop": 56, "vol": 18, "conviction": "High",
     "bull": {"value": 74, "prob": 25}, "base": {"value": 68, "prob": 55}, "bear": {"value": 58, "prob": 20}},
    {**SCEN, "symbol": "NOSTOP", "stop": None},
]


def run_sort():
    r = subprocess.run([NODE, "-e", SORT, str(CORE)], input=json.dumps({"portfolio": EXAMPLE, "ideas": IDEAS}),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_each_trade_option_is_projected_and_sized_by_the_same_rules():
    out = run_sort()
    rows = {r["symbol"]: r for r in out["rows"]}
    view = portfolio.valued(portfolio.validate(copy.deepcopy(EXAMPLE)))
    for idea in IDEAS[:3]:
        r = rows[idea["symbol"]]
        want = portfolio.size_position(copy.deepcopy(view), idea["symbol"], idea["price"], idea["stop"], idea["conviction"], idea["sector"])
        assert r["shares"] == want["shares"] and abs(r["loss_at_stop"] - want["loss_at_stop"]) <= 0.011
        pwv = verify.pwv({c: idea[c] for c in ("bull", "base", "bear")})
        assert abs(r["reward_risk"] - (pwv - idea["price"]) / (idea["price"] - idea["stop"])) < 1e-9
        assert r["rating"] == verify.expected_rating(100 * (pwv / idea["price"] - 1), idea["conviction"])
    assert rows["NOSTOP"]["reward_risk"] is None and rows["NOSTOP"]["shares"] is None and rows["NOSTOP"]["p_touch_stop"] is None


def test_trade_options_sort_best_first_with_blanks_last():
    out = run_sort()
    rows = {r["symbol"]: r for r in out["rows"]}
    for key, best in (("expected_return_pct", "high"), ("reward_risk", "high"), ("p_end_above_price", "high"),
                      ("p_touch_stop", "low"), ("shares", "high")):
        for dir in ("best", "worst"):
            order = out["sorts"][f"{key}:{dir}"]
            have = [s for s in order if rows[s][key] is not None]
            assert order[len(have):] == [s for s in order if rows[s][key] is None]          # blanks always last
            values = [rows[s][key] for s in have]
            high_first = (best == "high") == (dir == "best")
            assert values == sorted(values, reverse=high_first), (key, dir, order)
    assert out["sorts"]["symbol:best"] == sorted(rows)
    rank = {"Overweight": 2, "Equal-weight": 1, "Underweight": 0}
    by_rating = out["sorts"]["rating:best"]
    assert [rank[rows[s]["rating"]] for s in by_rating] == sorted((rank[rows[s]["rating"]] for s in by_rating), reverse=True)
    assert "unknown sort" in out["bad"]
