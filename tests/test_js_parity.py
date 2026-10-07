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
sys.path.insert(0, str(ROOT))
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
