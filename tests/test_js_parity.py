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
]


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
    memos = [memo_with(b) for b in broken_blocks()] + ["no block here", "```json\n{oops\n```"]
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
    got = run_project({"ratings": grid, "runs": []})["ratings"]
    assert got == [verify.expected_rating(er, conv) for er, conv in grid]


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
