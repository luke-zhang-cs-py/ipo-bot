"""robustness.py offline: the first month's cost, a one-month run, and the full report on a made-up universe."""
import math
import runpy
import sys

import pytest

from test_eval1_cov_helpers import ROOT, plain_streams

import benchmark as bm  # noqa: E402
import robustness as rb  # noqa: E402
from test_benchmark import fake_panel  # noqa: E402


def one_month_keep():
    return {"raw": {0: {"A": [3.0, 2.0, 1.0]}, 1: {"A": [1.0, 2.0, 3.0]}},
            1: {0: [0.10, 0.0, -0.10], 1: [0.0, 0.0, 0.10]}}


def test_the_first_month_pays_to_buy_only_and_later_months_pay_both_ways():
    keep = one_month_keep()
    ret, turn = rb.top_third_net(keep, "A", [0], 100)        # 1% a side; twelve "years" of one month
    assert (1 + ret) ** (1 / 12) == pytest.approx(1.10 * (1 - 0.01))   # bought once: 1%, not 2%
    assert math.isnan(turn)                                   # no month-to-month turnover in one month
    ret2, turn2 = rb.top_third_net(keep, "A", [0, 1], 100)   # month 2 replaces the whole third: sell + buy
    assert turn2 == 1.0
    assert (1 + ret2) ** (2 / 12) == pytest.approx(1.10 * 0.99 * 1.10 * 0.98)


def test_equal_weight_and_benjamini_hochberg_with_nothing_to_test():
    assert rb.equal_weight(one_month_keep(), [0, 1]) == pytest.approx((1.0 * (1 + 0.1 / 3)) ** 6 - 1)
    assert rb.benjamini_hochberg([]) == []
    assert rb.pct is bm.pct


def toy(monkeypatch, tmp_path):
    _, px = fake_panel(n_months=150)
    months = [f"{2007 + m // 12}-{m % 12 + 1:02d}" for m in range(150)]      # both sides of the 2018 split
    monkeypatch.setattr(bm, "UNIVERSES", {"Toy": list(px)})
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (months, px))
    monkeypatch.setattr(bm, "ROOT", tmp_path)
    perm = rb.perm_p
    monkeypatch.setattr(rb, "perm_p", lambda keep, algo, h, months, perms=None, seed=1: perm(keep, algo, h, months, perms=40, seed=seed))
    return months, px


def test_the_report_and_main_on_a_made_up_universe(tmp_path, monkeypatch):
    toy(monkeypatch, tmp_path)
    out = plain_streams(monkeypatch)
    assert rb.main() == 0
    saved = list((tmp_path / "bench_runs").glob("robustness_*.md"))
    text = saved[0].read_text(encoding="utf-8")
    assert len(saved) == 1 and text in out.getvalue()
    assert "## 1. Multiple testing: 12 tests, false-discovery rate 10%" in text      # 6 algorithms x 2 horizons
    assert "| Toy | 12 mo | Momentum 12-1 |" in text and "**yes**" in text           # a real trend survives
    assert text.count("Random control, which should not pass: Toy 12 mo p = ") == 1
    holdout = text.split("## 2. Holdout")[1].split("## 3.")[0]
    assert "picked on the first half" in holdout and "| Toy |" in holdout
    costs = text.split("## 3. After trading costs")[1]
    keep = {}
    bm.run("Toy", keep=keep)
    ew = rb.equal_weight(keep, sorted(keep[1]))
    assert f"### Toy: holding everything returned {bm.pct(ew)} a year" in costs
    assert costs.count("| Random (no skill) |") == 1


def test_the_script_entry_point_runs_main(monkeypatch):
    class Stop(Exception):
        pass

    def run(universe, refresh=False, seed=7, keep=None):
        raise Stop(universe)
    monkeypatch.setattr(bm, "run", run)
    plain_streams(monkeypatch)
    with pytest.raises(Stop) as e:
        runpy.run_path(str(ROOT / "evaluation" / "robustness.py"), run_name="__main__")
    assert e.value.args[0] == next(iter(bm.UNIVERSES))
