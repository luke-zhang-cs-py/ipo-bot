"""The calibration test's own arithmetic, and that it never uses an outcome before it is known. Offline: made-up prices."""
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import benchmark as bm  # noqa: E402
import calibration as cal  # noqa: E402


def test_the_running_line_matches_a_direct_fit():
    xs = [0.1, 0.4, -0.3, 0.2, -0.1]
    ys = [1.0, 2.2, -0.5, 1.4, 0.3]
    a, b = cal.fit_line(xs, ys)
    mx, my = sum(xs) / 5, sum(ys) / 5
    b0 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    assert b == pytest.approx(b0) and a == pytest.approx(my - b0 * mx)


def test_quantile_and_positions():
    assert cal.quantile([1, 2, 3, 4, 5], 0.5) == 3 and cal.quantile([0, 10], 0.25) == 2.5
    p = cal.positions([5, 1, 3])
    assert p == pytest.approx([1 / 3, -1 / 3, 0.0])        # best +, worst -, centred on 0


def test_ece_is_zero_when_probabilities_match_frequencies():
    assert cal.ece([0.25] * 4, [1, 0, 0, 0]) == pytest.approx(0)
    assert cal.ece([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def panel(n_months=130, n=12, seed=5, skill=0.01):
    """Each stock has its own drift, and its past return reveals it: a signal with real, stable skill."""
    rng = random.Random(seed)
    months = [f"{2008 + m // 12}-{m % 12 + 1:02d}" for m in range(n_months)]
    px = {}
    for k in range(n):
        drift = skill * (k - n / 2) / n
        p, path = 100.0, []
        for _ in months:
            p *= 1 + 0.008 + drift + rng.gauss(0, 0.05)
            path.append(p)
        px[f"S{k}"] = path
    return months, px


def test_forecasts_only_use_outcomes_known_at_the_time():
    months, px = panel()
    symbols = list(px)
    h, cut = 12, 110
    before = cal.forecasts(months, px, symbols, h)
    changed = {s: p[:cut + 1] + [x * (2 if k % 2 else 0.5) for k, x in enumerate(p[cut + 1:])] for s, p in px.items()}
    after = cal.forecasts(months, changed, symbols, h)
    for a in before:
        old = [(r["f"], r["p"], r["ranges"][80]) for r in before[a] if r["month"] <= months[cut]]
        new = [(r["f"], r["p"], r["ranges"][80]) for r in after[a] if r["month"] <= months[cut]]
        assert old and old == new, a


def test_a_real_signal_scores_better_than_luck_and_its_ranges_hold():
    months, px = panel(n_months=200, n=20, skill=0.02)
    per = cal.forecasts(months, px, list(px), 1)
    hist, rnd = cal.score(per["Historical mean"]), cal.score(per[cal.RANDOM])
    assert hist["r2_os"] > rnd["r2_os"] and hist["bss"] > rnd["bss"] and hist["slope"] > 0.3
    for c, s in hist["cover"].items():
        assert abs(s - c / 100) < 0.05, (c, s)     # ranges from past errors hold when the world is stable


def test_the_bot_thresholds_split_the_ratings():
    recs = [{"f": f, "y": 0.0, "beat": 0} for f in (0.2, 0.15, 0.05, -0.1, -0.3)]
    r = cal.ratings(recs)
    assert (r["Overweight"]["n"], r["Equal-weight"]["n"], r["Underweight"]["n"]) == (2, 1, 2)


def test_the_null_band_marks_only_scores_outside_it():
    assert cal.outside(0.5, (0.0, 0.2)) == "*" and cal.outside(-0.1, (0.0, 0.2)) == "!"
    assert cal.outside(0.1, (0.0, 0.2)) == "" and cal.outside(0.01, (0.02, 0.04), better_high=False) == "*"


def test_the_sector_universe_is_in_both_tests():
    assert "US sectors (SPDR ETFs)" in bm.UNIVERSES


def test_the_bots_bull_and_bear_cases_are_checked_against_their_probabilities(tmp_path, monkeypatch):
    import json
    rows = [{"date": "2025-01-10", "symbol": s, "rating": "Overweight", "expected_return_pct": 20, "price": 100,
             "price_date": "2025-01-09", "horizon_months": 12, "bull_value": 140, "bull_prob": 25,
             "bear_value": 80, "bear_prob": 20} for s in ("AAA", "BBB", "CCC", "DDD")]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(bm, "LEDGER", ledger)
    ends = {"AAA": 150, "BBB": 75, "CCC": 110, "DDD": 120}           # one bull, one bear, two in between
    monkeypatch.setattr(bm, "fetch", lambda s, refresh=False: {"2025-12": ends[s]})
    text = cal.ledger_calibration()
    assert "4 matured" in text
    assert "Bear case or beyond: happened 25% of the time; the bot gave it 20%" in text
    assert "Bull case or beyond: happened 25% of the time; the bot gave it 25%" in text
    assert "Overweight: 4 calls, expected +20.0%, realised +13.8%" in text
