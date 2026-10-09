"""The backtest suite's machinery on synthetic data: regimes read only the past, the out-of-sample test is
recorded once, break-even costs, and re-optimisation that only looks back."""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import backtest_suite as bs  # noqa: E402
import benchmark as bm  # noqa: E402


def months(start=2010, n=180):
    return [f"{start + k // 12}-{k % 12 + 1:02d}" for k in range(n)]


def synthetic_keep(n_stocks=30, seed=3, skill=None):
    """Monthly scores and next-month returns. `skill`: the algorithm whose scores predict returns."""
    rng = random.Random(seed)
    ms = months()
    keep = {"months": ms, "raw": {}, 1: {}, 12: {}}
    for i in range(len(ms) - 1):
        fwd = [rng.gauss(0.01, 0.05) for _ in range(n_stocks)]
        keep[1][i] = fwd
        keep["raw"][i] = {a: [rng.random() for _ in range(n_stocks)] for a in bs.ALGOS + ["Random (no skill)"]}
        if skill:
            keep["raw"][i][skill] = [f + rng.gauss(0, 0.05) for f in fwd]
    return keep


def test_a_month_is_labelled_from_spy_up_to_it_never_after(monkeypatch):
    ms = months(2004, 120)
    spy = {m: 100 * (1.02 ** k if k < 60 else 1.02 ** 60 * 0.97 ** (k - 60)) for k, m in enumerate(ms)}
    monkeypatch.setattr(bm, "fetch", lambda sym: dict(spy))
    before = bs.spy_regimes(ms)
    assert before["2008-06"] == "bull" and before["2009-12"] == "drawdown"
    # rewrite everything after 2008-06: the labels up to it must not move
    later = {m: (v if m <= "2008-06" else v * 5) for m, v in spy.items()}
    monkeypatch.setattr(bm, "fetch", lambda sym: dict(later))
    after = bs.spy_regimes(ms)
    assert {m: after[m] for m in ms if m <= "2008-06"} == {m: before[m] for m in ms if m <= "2008-06"}


def test_how_quiet_a_month_is_is_judged_against_the_past_only(monkeypatch):
    ms = months(2004, 60)
    rng = random.Random(1)
    spy = {m: 100 + (rng.random() - 0.5) * 0.5 for m in ms}                 # flat and quiet throughout
    monkeypatch.setattr(bm, "fetch", lambda sym: dict(spy))
    before = bs.spy_regimes(ms)
    # make the months after 2006-06 wild (still flat on a year): the median volatility of all months jumps,
    # but the labels up to 2006-06 must not move
    later = {m: (v if m <= "2006-06" else 100 * (1 + 0.04 * (-1) ** k)) for k, (m, v) in enumerate(sorted(spy.items()))}
    monkeypatch.setattr(bm, "fetch", lambda sym: dict(later))
    after = bs.spy_regimes(ms)
    assert {m: after[m] for m in ms if m <= "2006-06"} == {m: before[m] for m in ms if m <= "2006-06"}


def test_a_flat_quiet_market_is_sideways(monkeypatch):
    ms = months(2004, 60)
    rng = random.Random(1)
    spy = {m: 100 + (rng.random() - 0.5) * 0.5 if k < 40 else 100 * (1 + 0.05 * rng.gauss(0, 1)) for k, m in enumerate(ms)}
    monkeypatch.setattr(bm, "fetch", lambda sym: dict(spy))
    labels = bs.spy_regimes(ms)
    assert "sideways" in {labels[m] for m in ms[13:40]}


def test_the_out_of_sample_test_is_recorded_once(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "OOS_LOG", tmp_path / "oos.jsonl")
    keep = synthetic_keep(skill="Momentum 12-1")
    keep["months"] = months(2015, len(keep["months"]))       # spans the design years and the test years
    first, looks, fresh = bs.out_of_sample({"toy": keep}, perms=50)
    assert fresh and looks == 1 and any(r["algo"] == "Momentum 12-1" for r in first["results"])
    again, looks2, fresh2 = bs.out_of_sample({"toy": synthetic_keep(seed=99)}, perms=50)
    assert not fresh2 and looks2 == 1 and again == first       # the result of record, not a re-test on new data
    _, looks3, fresh3 = bs.out_of_sample({"toy": keep}, again=True, perms=50)
    assert fresh3 and looks3 == 2                              # a deliberate second look, and counted


def test_break_even_cost_is_zero_without_an_edge_and_positive_with_one():
    keep = synthetic_keep(skill="Momentum 12-1")
    ms = sorted(keep[1])
    assert bs.break_even(keep, "Momentum 12-1", ms) > 25
    noise = bs.break_even(keep, "Random (no skill)", ms)
    assert noise < bs.break_even(keep, "Momentum 12-1", ms)


def test_reoptimisation_picks_from_the_past_and_finds_a_persistent_winner():
    keep = synthetic_keep(skill="Trend (10-mo avg)")
    picked, everyone, (best, cheat), picks = bs.reoptimised(keep)
    assert picks and all(a == "Trend (10-mo avg)" for _, a in picks)       # the one real signal, every year
    assert sum(picked) / len(picked) > sum(everyone) / len(everyone) + 0.1
    assert best == "Trend (10-mo avg)"
    first_year = int(picks[0][0])
    assert first_year >= int(keep["months"][0][:4]) + bs.WINDOW // 12      # a full window of known months first
