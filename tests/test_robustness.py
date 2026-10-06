"""The robustness tests' statistics. Offline: made-up prices."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import benchmark as bm  # noqa: E402
import robustness as rb  # noqa: E402
from test_benchmark import fake_panel  # noqa: E402


def test_benjamini_hochberg():
    # m = 5, q = 0.10: thresholds 0.02, 0.04, 0.06, 0.08, 0.10. The largest p under its line is the 3rd (0.05).
    assert rb.benjamini_hochberg([0.01, 0.05, 0.9, 0.03, 0.5], q=0.10) == [True, True, False, True, False]
    assert rb.benjamini_hochberg([0.2, 0.3], q=0.10) == [False, False]


@pytest.fixture
def keep(monkeypatch):
    months, px = fake_panel(n_months=150)
    monkeypatch.setitem(bm.UNIVERSES, "fake", list(px))
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (months, px))
    k = {}
    bm.run("fake", keep=k)
    return k


def test_a_real_signal_gets_a_small_p_and_random_does_not(keep):
    months = sorted(keep[12])
    _, p_real = rb.perm_p(keep, "Momentum 12-1", 12, months, perms=200)
    _, p_rand = rb.perm_p(keep, "Random (no skill)", 12, months, perms=200)
    assert p_real < 0.05 and p_rand > 0.05


def test_costs_only_lower_returns_and_scale_with_turnover(keep):
    months = sorted(keep[1])
    free, turn = rb.top_third_net(keep, "Momentum 12-1", months, 0)
    dear, _ = rb.top_third_net(keep, "Momentum 12-1", months, 50)
    churn_free, churn = rb.top_third_net(keep, "Random (no skill)", months, 0)
    churn_dear, _ = rb.top_third_net(keep, "Random (no skill)", months, 50)
    assert dear < free and churn > turn
    assert (churn_free - churn_dear) > (free - dear)        # more turnover, more cost


def test_the_holdout_never_lets_a_first_half_outcome_reach_past_the_split(keep):
    h = 12
    first = [i for i in sorted(keep[h]) if rb.month_of(keep, i + h) < "2017-01"]
    assert first and all(keep["months"][i + h] < "2017-01" for i in first)
