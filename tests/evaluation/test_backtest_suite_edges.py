"""backtest_suite.py offline: a break-even past the top of the search, too few months for a regime or a
re-optimisation, and the full report (with --again) on made-up universes and a made-up SPY."""
import random
import runpy
import sys

import pytest

from eval_helpers import ROOT, plain_streams

import backtest_suite as bs  # noqa: E402
import benchmark as bm  # noqa: E402
import robustness as rb  # noqa: E402
from test_benchmark import fake_panel  # noqa: E402


def test_an_edge_that_survives_the_dearest_cost_reports_the_top_of_the_search():
    n, months = 30, list(range(24))
    keep = {"raw": {i: {"Sure": list(range(n))} for i in months},
            1: {i: [0.05 if k >= 20 else 0.0 for k in range(n)] for i in months}}   # the top third always wins
    assert bs.break_even(keep, "Sure", months) == 2000.0
    assert bs.break_even(keep, "Sure", months, hi=50.0) == 50.0
    keep[1] = {i: [0.0 if k >= 20 else 0.05 for k in range(n)] for i in months}     # and now always loses
    assert bs.break_even(keep, "Sure", months) == 0.0


def test_too_few_months_means_no_regime_and_no_reoptimisation():
    keep = {"months": [f"2020-{m:02d}" for m in range(1, 13)], 1: {i: [0.0] * 3 for i in range(11)}}
    assert bs.by_regime(keep, {m: "bull" for m in keep["months"][:5]}) == {}
    assert bs.reoptimised(keep, window=36) == ([], [], (None, []), [])
    assert bs.month_index(keep)["2020-03"] == 2


def spy_months():
    """SPY month-ends 2010-2024: a steady climb, a bull run, a slump, a short flat quiet stretch, then ordinary."""
    rng, p, out = random.Random(4), 100.0, {}
    for k in range(180):
        y = 2010 + k // 12
        drift, noise = ((0.012, 0.02) if y < 2016 else (0.025, 0.01) if y < 2018 else (-0.03, 0.01) if y < 2019
                        else (0.0, 0.002) if k < 118 else (0.006, 0.02))                  # flat until 2019-10
        p *= 1 + drift + rng.gauss(0, noise)
        out[f"{y}-{k % 12 + 1:02d}"] = p
    return out


def toy(monkeypatch, tmp_path):
    _, px = fake_panel(n_months=150)
    recent = [f"{2012 + m // 12}-{m % 12 + 1:02d}" for m in range(150)]       # design years and 2024 tests
    _, old_px = fake_panel(n_months=100, seed=8)
    old_px = {f"O{s}": v for s, v in old_px.items()}
    old = [f"{1980 + m // 12}-{m % 12 + 1:02d}" for m in range(100)]          # before SPY's data: no regimes
    monkeypatch.setattr(bm, "UNIVERSES", {"Toy": list(px), "Old": list(old_px)})
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (recent, px) if symbols[0] in px else (old, old_px))
    monkeypatch.setattr(bm, "fetch", lambda symbol, refresh=False: spy_months())
    monkeypatch.setattr(bm, "ROOT", tmp_path)
    monkeypatch.setattr(bs, "OOS_LOG", tmp_path / "bench_runs" / "oos_log.jsonl")
    perm = rb.perm_p
    monkeypatch.setattr(rb, "perm_p", lambda keep, algo, h, months, perms=None, seed=1: perm(keep, algo, h, months, perms=40, seed=seed))


def test_the_report_and_main_record_the_holdout_once_and_count_each_look(tmp_path, monkeypatch):
    toy(monkeypatch, tmp_path)
    out = plain_streams(monkeypatch)
    assert bs.main([]) == 0
    first = list((tmp_path / "bench_runs").glob("backtest_suite_*.md"))[0].read_text(encoding="utf-8")
    assert first in out.getvalue()
    assert "## 1. Out of sample: chosen on 2018-01 to 2023-12, tested from 2024-01" in first
    assert "First run, " in first and "this is the result of record." in first
    assert "| Toy | 1 mo | Momentum 12-1 |" in first and "| Old |" not in first.split("## 2.")[0]
    regimes = first.split("## 2. Regimes")[1].split("## 3.")[0]
    assert "### Toy" in regimes and "### Old" not in regimes                  # Old predates SPY: no labels
    assert "| Random (no skill) |" in regimes
    costs = first.split("## 3. Costs")[1].split("## 4.")[0]
    assert "### Toy: holding everything returned" in costs and "### Old: holding everything returned" in costs
    assert costs.count(" bp |") >= 2 or "never ahead" in costs
    walk = first.split("## 4. Walk-forward")[1]
    assert "| Toy |" in walk and "| Old |" not in walk                        # Old has no full window before a January
    assert "switches |" in walk
    assert bs.report().count("Not re-run: `--again` re-tests") == 1           # a second run shows the record
    text = bs.report(again=True)                                              # a deliberate second look
    assert "the test has been looked at 3 time(s)" not in text and "looked at 2 time(s)" in text
    assert len((tmp_path / "bench_runs" / "oos_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_regimes_too_thin_to_score_are_named(tmp_path, monkeypatch):
    toy(monkeypatch, tmp_path)
    text = bs.report()
    regimes = text.split("## 2. Regimes")[1].split("## 3.")[0]
    labels = bs.spy_regimes([f"{2012 + m // 12}-{m % 12 + 1:02d}" for m in range(150)])
    keep = {}
    bm.run("Toy", keep=keep)
    counts = {r: sum(1 for i in keep[1] if labels.get(keep["months"][i]) == r) for r in ("bull", "drawdown", "sideways", "ordinary")}
    thin = [r for r, c in counts.items() if c < 6]
    assert thin and f"Too few months to score: {thin[0]} ({counts[thin[0]]} months)" in regimes
    assert [r for r, c in counts.items() if c >= 6]


def test_the_script_entry_point_passes_again(monkeypatch, tmp_path):
    seen = []

    class Stop(Exception):
        pass

    def run(universe, refresh=False, seed=7, keep=None):
        seen.append(universe)
        raise Stop
    monkeypatch.setattr(bm, "run", run)
    monkeypatch.setattr(bm, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["backtest_suite.py", "--again"])
    plain_streams(monkeypatch)
    with pytest.raises(Stop):
        runpy.run_path(str(ROOT / "evaluation" / "backtest_suite.py"), run_name="__main__")
    assert seen == [next(iter(bm.UNIVERSES))]
