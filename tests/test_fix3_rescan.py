"""The rescan's fixes: calibration's report with no month-ends at all, robustness's one-month turnover, and the
one percentage formatter shared by tracker and ipo_eval."""
import math

from test_eval1_cov_helpers import plain_streams  # noqa: F401  (puts evaluation/ and src/ on the path)

import benchmark as bm  # noqa: E402
import calibration as cal  # noqa: E402
import ipo_eval as E  # noqa: E402
import robustness as rb  # noqa: E402
import tracker  # noqa: E402
from test_calibration import panel  # noqa: E402
from test_eval1_cov_robustness import toy  # noqa: E402


def test_the_calibration_report_says_too_few_months_when_there_is_no_month_end_at_all(tmp_path, monkeypatch):
    months, px = panel(n_months=130, n=9)
    short = {s: p[: bm.LOOKBACK + 1] for s, p in px.items()}          # LOOKBACK + 1 months: no 1-month forecast date
    assert cal.prepare(months[: bm.LOOKBACK + 1], short, list(short), 1)["idx"] == []
    monkeypatch.setattr(bm, "UNIVERSES", {"Tiny": list(short)})
    monkeypatch.setattr(bm, "panel", lambda symbols, refresh=False: (months[: bm.LOOKBACK + 1], short))
    monkeypatch.setattr(bm, "LEDGER", tmp_path / "none.jsonl")
    text = cal.report()                                               # used to raise IndexError on prep["idx"][0]
    assert "### 1-month forecasts: too few months" in text and "### 12-month forecasts: too few months" in text


def test_one_month_turnover_reads_n_a_not_nan_percent(tmp_path, monkeypatch):
    toy(monkeypatch, tmp_path)
    monkeypatch.setattr(rb, "top_third_net", lambda keep, algo, months, bps: (0.05, float("nan")))
    text = rb.report()
    costs = text.split("## 3. After trading costs")[1]
    assert "nan%" not in text and "| Random (no skill) | n/a |" in costs


def test_tracker_and_ipo_eval_format_percentages_with_benchmarks_formatter():
    assert E.pct is bm.pct
    assert tracker.pct(0.012345) == "+1.23%" and tracker.pct(0.012345, 3) == "+1.234%"   # decimals positional
    assert tracker.pct(None) == "n/a" and tracker.pct(float("nan")) == "n/a" and tracker.pct(math.inf) == "n/a"
    assert tracker.pct(-0.00001) == "0.00%" and E.pct(-0.0001) == "0.0%"                # no "-0.00%" from rounding
