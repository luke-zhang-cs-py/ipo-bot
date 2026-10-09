"""calibration.py offline: the small helpers' edge cases, the permutation bands, the ledger's skips, and the
full report on a made-up universe."""
import json
import math
import runpy
import sys

import pytest

from eval_helpers import ROOT, plain_streams

import benchmark as bm  # noqa: E402
import calibration as cal  # noqa: E402
from test_calibration import panel  # noqa: E402


def test_quantile_of_nothing_and_a_line_through_too_few_points():
    assert cal.quantile([], 0.5) == 0.0
    assert cal.Line().coef() == (0.0, 0.0)
    one = cal.Line()
    one.add(1.0, 3.0)
    assert one.coef() == (3.0, 0.0)
    assert cal.fit_line([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == (2.0, 0.0)      # no spread in x: no slope


def test_walk_without_ranges_scores_the_same_core_and_reuses_it_in_score():
    months, px = panel(n_months=120, n=9)
    prep = cal.prepare(months, px, list(px), 1)
    full, bare = cal.walk(prep, "Momentum 12-1"), cal.walk(prep, "Momentum 12-1", ranges=False)
    assert bare and all("ranges" not in r and "normal80" not in r for r in bare)
    assert [(r["f"], r["p"]) for r in bare] == [(r["f"], r["p"]) for r in full]
    core, s = cal.core_scores(bare), cal.score(full)
    assert {k: s[k] for k in core} == pytest.approx(core)
    assert s["n"] == len(full) and set(s["cover"]) == set(cal.RANGES)
    assert s["mean_f"] == pytest.approx(sum(r["f"] for r in full) / len(full))


def test_the_null_bands_are_per_algorithm_and_contain_the_random_control():
    months, px = panel(n_months=100, n=9)
    prep = cal.prepare(months, px, list(px), 1)
    bands = cal.null_bands(prep, seeds=4)
    assert set(bands) == set(prep["pos"][prep["idx"][0]]) and set(bands[cal.RANDOM]) == set(cal.NULL_KEYS)
    assert all(lo <= hi for b in bands.values() for lo, hi in b.values())


def test_reliability_blocks_and_the_market_level():
    recs = [{"month": f"{2010 + k // 24}-01", "fx": k / 100, "yx": k / 50, "naive": 0.01 * (-1) ** k,
             "y": 0.02 * (-1) ** k} for k in range(50)]
    groups = cal.reliability(recs)
    assert len(groups) == 5 and groups[0][1] == pytest.approx(2 * groups[0][0])
    slopes = cal.block_slopes(recs)
    assert slopes[0] == ("2010-12", pytest.approx(2.0)) and len(slopes) == 1
    assert math.isnan(cal.block_slopes(recs[:5])[0][1])                      # 10 or fewer forecasts: no slope
    m = cal.market_level(recs)
    assert m["n"] == 3 and 0 <= m["sign"] <= 1


def write_ledger(tmp_path, monkeypatch, rows):
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    monkeypatch.setattr(bm, "LEDGER", path)


def test_the_ledger_skips_what_has_not_matured_or_cannot_be_priced(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "LEDGER", tmp_path / "missing.jsonl")
    assert cal.ledger_calibration().startswith("No bot forecasts logged yet")
    base = {"rating": "Equal-weight", "expected_return_pct": 5, "price": 100, "horizon_months": 12}
    write_ledger(tmp_path, monkeypatch, [
        {**base, "date": "2099-01-10", "symbol": "LATER"},                  # not matured
        {**base, "date": "2024-01-10", "symbol": "NOPRICE"},                # no close for its end month
        {**base, "date": "2024-01-10", "symbol": "NOFACTOR"},               # no adjusted close for the entry day
        {**base, "date": "2024-01-10", "symbol": "OK", "bull_value": 130},  # a bull value without a probability
    ])
    monkeypatch.setattr(bm, "fetch", lambda s, refresh=False: {} if s == "NOPRICE" else {"2024-12": 90.0})
    monkeypatch.setattr(bm, "entry_factor", lambda s, day: None if s == "NOFACTOR" else 1.0)
    text = cal.ledger_calibration()
    assert text.splitlines()[0] == "4 logged, 1 matured and priced."
    assert "slope" not in text and "case or beyond" not in text               # one is too few; no probabilities
    assert "Equal-weight: 1 calls, expected +5.0%, realised -10.0%, up 0% of the time." in text
    assert "Overweight" not in text


def test_thirty_matured_forecasts_are_enough_to_read(tmp_path, monkeypatch):
    rows = [{"date": "2024-01-10", "symbol": f"S{k}", "rating": "Overweight", "expected_return_pct": 10 + k,
             "price": 100, "horizon_months": 12} for k in range(30)]
    write_ledger(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(bm, "fetch", lambda s, refresh=False: {"2024-12": 100.0 + 2 * int(s[1:])})
    monkeypatch.setattr(bm, "entry_factor", lambda s, day: 1.0)
    text = cal.ledger_calibration()
    assert "slope 2.00, intercept -20.0%" in text and "n = 30." in text and "too few" not in text


def test_the_report_and_main_on_a_made_up_universe(tmp_path, monkeypatch):
    months, px = panel(n_months=130, n=9)
    short = {f"Q{k}": p[:110] for k, p in enumerate(px.values())}             # 110 months: too few for 12-month
    monkeypatch.setattr(bm, "UNIVERSES", {"Toy": list(px), "Short": list(short)})
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (months, px) if symbols[0] == "S0" else (months[:110], short))
    monkeypatch.setattr(bm, "ROOT", tmp_path)
    monkeypatch.setattr(bm, "LEDGER", tmp_path / "none.jsonl")
    bands = cal.null_bands
    monkeypatch.setattr(cal, "null_bands", lambda prep, seeds=None: bands(prep, seeds=3))
    out = plain_streams(monkeypatch)
    assert cal.main(["--ledger"]) == 0
    saved = list((tmp_path / "bench_runs").glob("calibration_*.md"))
    text = saved[0].read_text(encoding="utf-8")
    assert len(saved) == 1 and text in out.getvalue()
    assert "## Short (9 names)" in text and "### 12-month forecasts: too few months (36" in text
    assert text.split("## Short")[1].count("### 1-month forecasts (") == 1
    assert "## Toy (9 names)" in text and "### 12-month forecasts (" in text and "### 1-month forecasts (" in text
    prep = cal.prepare(months, px, list(px), 12)
    s = cal.score(cal.walk(prep, cal.RANDOM))
    assert f"{s['n']:,} stock forecasts per algorithm" in text
    assert "The bot's rating rules applied" in text and text.count("The bot's rating rules applied") == 1
    assert "| Fifth 1 |" in text and "Market level (shared by every algorithm" in text
    assert "## The bot's own forecasts" in text and "No bot forecasts logged yet" in text
    assert "## The bot's own forecasts" not in cal.report()


def test_the_script_entry_point_passes_its_arguments(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def panel_(symbols, refresh=False):
        seen.append(refresh)
        raise Stop
    monkeypatch.setattr(bm, "panel", panel_)
    monkeypatch.setattr(sys, "argv", ["calibration.py", "--refresh"])
    plain_streams(monkeypatch)
    with pytest.raises(Stop):
        runpy.run_path(str(ROOT / "evaluation" / "calibration.py"), run_name="__main__")
    assert seen == [True]
