"""ipo_eval's remaining branches and its whole report, offline on a small synthetic IPO market: SPY and the dataset
are fakes, the report's files go to tmp_path, and the model is cut to three features and two test years."""
import datetime as dt
import io
import json
import math
import pathlib
import random
import statistics
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import ipo_eval as E  # noqa: E402

SMALL = ["revision", "no_range", "log_price"]


def ipos(start, end, n, seed=1, pop_odds=None):
    """n IPOs listed between two dates whose pop odds rise with the range revision (pop_odds=0: none pop)."""
    rng = random.Random(seed)
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    rows = []
    for i in range(n):
        day = d0 + dt.timedelta(days=int(i * (d1 - d0).days / n))
        lo = rng.uniform(8, 30)
        rev = rng.gauss(0, 0.12)
        offer = round((lo + 2) * (1 + rev), 2)
        p = 1 / (1 + math.exp(-(-1.0 + 2.5 * rev / 0.12))) if pop_odds is None else pop_odds
        first = rng.uniform(0.25, 0.9) if rng.random() < p else rng.uniform(-0.25, 0.15)
        rows.append({"adsh": f"{start}-{i}", "cik": str(i), "sic": rng.choice(["7372", "2834", "6022", "5500"]),
                     "prospectus_date": (day + dt.timedelta(days=1)).isoformat(), "offer_price": offer,
                     "shares": rng.randint(2, 30) * 1e6, "lead_bank": rng.choice(["GS", "MS", "Maxim", None]),
                     "foreign": rng.random() < 0.2, "range": [lo, lo + 4], "range_date": (day - dt.timedelta(days=9)).isoformat(),
                     "prices": {"listing_date": day.isoformat(), "open": offer * (1 + first * 0.8), "close": offer * (1 + first),
                                "close_21": offer * (1 + first) * rng.uniform(0.8, 1.2) if i % 3 else None,
                                "close_252": None, "suspect": False}})
    return rows


def spy_daily(start="2014-01-01", end="2026-01-01", step=0.001):
    d, k, out = dt.date.fromisoformat(start), 0, {}
    while d.isoformat() < end:
        out[d.isoformat()] = 100 * (1 + step) ** k
        d += dt.timedelta(days=1)
        k += 1
    return out


# ----------------------------------------------------------------------------- the sample

def test_the_sample_leaves_out_misread_offers_and_suspect_prices_and_says_so():
    good = ipos("2016-01-01", "2016-06-01", 3)
    suspect = dict(good[0], adsh="sus", prices={**good[0]["prices"], "suspect": True})
    off = dict(good[1], adsh="off", offer_price=good[1]["range"][1] * 2.5)            # far above the filed range
    unpriced = dict(good[2], adsh="np", prices=None)
    no_range = dict(good[2], adsh="nr", range=None)
    rows = good + [suspect, off, unpriced, no_range]
    # the main sample drops misreads caught from the filings; suspect opens leave only the robustness sample
    assert [r["adsh"] for r in E.usable(rows)] == [r["adsh"] for r in good] + ["sus", "nr"]
    assert [r["adsh"] for r in E.usable(rows, drop_suspect=True)] == [r["adsh"] for r in good] + ["nr"]
    assert E.offer_off_range(off) and not E.offer_off_range(no_range) and not E.offer_off_range(good[0])
    ex = E.excluded(rows)
    assert [r["adsh"] for r in ex["suspect prices (robustness sample only)"]] == ["sus"]
    assert [r["adsh"] for r in ex["offer outside the filed range"]] == ["off"]
    assert not E.priced(unpriced) and E.priced(good[0])


# ----------------------------------------------------------------------------- small pieces

def test_sic_groups_spy_returns_and_every_regime():
    assert [E.sic_group(s) for s in ("2834", "8731", "3841", "7372", "4899", "6022", "5500", None, "x")] == \
        ["biotech", "biotech", "biotech", "tech", "tech", "finance", "", "", ""]
    spy = {f"2020-01-{d:02d}": float(d) for d in range(1, 31)}
    r = E.spy_returns(spy)
    assert sorted(r) == [f"2020-01-{d:02d}" for d in range(22, 31)] and r["2020-01-22"] == 21 / 1 - 1
    days = [(dt.date(2019, 1, 1) + dt.timedelta(days=k)).isoformat() for k in range(300)]
    flat = {d: 100 + 0.01 * (k % 2) for k, d in enumerate(days)}
    assert E.spy_regime(flat, days[-1]) == "sideways"
    gentle = {d: 100 * 1.0004 ** k for k, d in enumerate(days)}                      # up about 10% in a year
    assert E.spy_regime(gentle, days[-1]) == "ordinary"
    choppy = {d: 100 * (1.03 if k % 2 else 0.97) for k, d in enumerate(days)}
    assert E.spy_regime(choppy, days[-1]) == "ordinary"                              # flat but loud: not sideways


def test_features_without_shares_or_range_and_metrics_with_one_class():
    r = {"adsh": "a", "offer_price": 10.0, "prices": {"listing_date": "2020-01-02"}}
    f = E.features(r, {}, {})
    assert f["log_proceeds"] == math.log(50e6) and f["no_range"] == 1.0 and f["revision"] == 0.0 and f["heat_30d"] == 0.0
    assert E.best_threshold([], []) == 0.5
    card = E.scorecard([0.1, 0.2], [0, 0], 0.5, 0.0)
    assert math.isnan(card["pr_auc"]) and math.isnan(card["roc_auc"]) and math.isnan(card["brier_skill"])
    assert E.share(1, 0) == "n/a" and E.pct(None) == "n/a" and E.pct(float("nan")) == "n/a" and E.pct(0.123) == "+12.3%"
    assert E.num(None) == "n/a" and E.num(float("nan")) == "n/a" and E.num(0.5, 2) == "0.50"


def test_a_logit_stopped_early_still_predicts_and_moves_toward_the_signal():
    feats = [{"a": x / 10} for x in range(-20, 21)]
    ys = [int(f["a"] > 0) for f in feats]
    one = E.Logit(["a"]).fit(feats, ys, iters=1)                  # a single Newton step: the loop runs out, no break
    full = E.Logit(["a"]).fit(feats, ys)
    assert 0.5 < one.prob({"a": 1.0}) < 1 and 0.5 < full.prob({"a": 1.0}) < 1 and one.w != full.w


def test_walk_forward_skips_years_without_enough_history():
    rows = ipos("2016-01-01", "2018-12-31", 90)
    folds = E.walk_forward(rows, E.market_features(rows, {}), years=range(2016, 2019), names=SMALL)
    assert [f["year"] for f in folds] == [2018]                 # 2016: no history; 2017: under 50 IPOs before it


def test_trading_checks_with_missing_exits_and_a_market_that_loses_at_the_open():
    rows = ipos("2018-01-01", "2018-12-31", 10)
    for r in rows:
        r["prices"]["close_21"] = None
        r["prices"]["close"] = r["prices"]["open"] * 0.9          # every IPO falls from its open
    fold = {"rows": rows, "scores": [0.1 * k for k in range(10)]}
    out = E.at_open([fold], 0.0)
    assert out[("every IPO", "21 days")]["n"] == 0 and math.isnan(out[("every IPO", "21 days")]["mean"])
    assert out[("every IPO", "day 1")]["mean"] < 0 and E.break_even_friction([fold]) == 0.0
    assert E.top_picks(fold, 0.2) == [rows[9], rows[8]] and E.top_picks({"rows": rows[:2], "scores": [0.1, 0.9]}) == [rows[1]]


def test_score_live_says_unknown_for_a_non_event_and_error_when_the_model_fails():
    class Broken:
        def prob(self, f):
            raise ValueError("bad model")
    ev = {"prospectus": "The initial public offering price is $20.00 per share."}
    assert E.score_live("not a dict", Broken(), {}, {}) == {"status": "unknown", "why": "not an event"}
    assert E.score_live(ev, Broken(), {}, {}) == {"status": "error", "why": "ValueError: bad model"}


def test_spy_closes_are_fetched_once_a_day_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ROOT", tmp_path)
    fetched = []

    def urlopen(req, timeout):
        fetched.append(req.full_url)
        t = [int(dt.datetime(2020, 1, d, tzinfo=dt.timezone.utc).timestamp()) for d in (2, 3)]
        return io.BytesIO(json.dumps({"chart": {"result": [{"timestamp": t, "indicators": {"quote": [{"close": [300.0, None]}]}}]}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert E.spy_closes() == {"2020-01-02": 300.0} and len(fetched) == 1      # no cache (not even its folder) yet
    assert E.spy_closes() == {"2020-01-02": 300.0} and len(fetched) == 1      # today's cache
    path = tmp_path / "bench_data" / "ipo" / "spy.json"
    path.write_text(json.dumps({"fetched": "2000-01-01", "closes": {}}), encoding="utf-8")
    assert E.spy_closes() == {"2020-01-02": 300.0} and len(fetched) == 2      # a stale cache is refetched


# ----------------------------------------------------------------------------- the report

@pytest.fixture
def small_report(tmp_path, monkeypatch):
    """The report on a small market: three features, test years 2018-19, a short stress run, files in tmp_path."""
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "HOLDOUT_LOG", tmp_path / "bench_runs" / "ipo_holdout_log.jsonl")
    monkeypatch.setattr(E.walk_forward, "__defaults__", (range(2018, 2020), SMALL, E.L2, None, None, False))
    monkeypatch.setattr(E.fit_year, "__defaults__", (SMALL, E.L2))
    monkeypatch.setattr(E, "FEATURES", SMALL)
    monkeypatch.setattr(E, "L2_GRID", (1.0, 10.0))
    stress = E.stress
    monkeypatch.setattr(E, "stress", lambda model, banks: stress(model, banks, n=40, burst=5))
    spy = spy_daily()
    monkeypatch.setattr(E, "spy_closes", lambda: spy)

    def run(raw, **kw):
        monkeypatch.setattr(E.ipo_data, "load", lambda: raw)
        return E.report(**kw).read_text(encoding="utf-8")
    return run


def dataset(holdout_pops=None, unpriced=0):
    dev = ipos("2015-01-01", "2019-12-31", 200, seed=2)
    hold = ipos("2024-01-01", "2025-06-30", 40, seed=3, pop_odds=holdout_pops)
    sus = dict(dev[5], adsh="sus", prices={**dev[5]["prices"], "suspect": True})
    off = dict(dev[6], adsh="off", offer_price=dev[6]["range"][1] * 3)
    missing = [dict(r, adsh=f"np{k}", prices=None) for k, r in enumerate(dev[:unpriced])]
    return dev + hold + [sus, off] + missing


def test_the_report_in_development_keeps_the_holdout_sealed(small_report, tmp_path):
    text = small_report(dataset())
    assert "Not scored: IPOs listed from 2024-01-01 stay sealed" in text
    assert "| suspect prices (robustness sample only) | 1 |" in text and "| offer outside the filed range | 1 |" in text
    assert "| 2018 |" in text and "| 2019 |" in text and "Only the test years where at least 60%" in text
    assert "| SPY bull on the listing day |" in text and "| SPY drawdown on the listing day | 0 | too few to score" in text
    assert "| normal 2018-19 |" in text and "bull 2020-21" not in text            # no test year in that span
    assert "| revision |" in text and "The planted canary (a feature that reads the first day) was caught" in text
    assert "40 synthetic events" in text and not (tmp_path / "bench_runs" / "ipo_holdout_log.jsonl").exists()


def test_the_holdout_is_scored_once_then_shown_as_the_result_of_record(small_report, tmp_path):
    text = small_report(dataset(), final=True)
    assert "IPOs listed from 2024-01-01, never used while building" in text and "This is look number 1." in text
    log = tmp_path / "bench_runs" / "ipo_holdout_log.jsonl"
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    assert list((tmp_path / "bench_runs").glob("ipo_eval_*_final.md"))
    text = small_report(dataset(), final=True)
    assert "Sealed holdout (the result of record)" in text and "Looked at 1 time(s)" in text
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1        # showing the record is not another look


def test_a_second_look_with_no_holdout_pops_and_a_thinly_priced_sample(small_report, tmp_path):
    text = small_report(dataset(holdout_pops=0.0, unpriced=200), final=True, again=True)
    assert "Too few holdout IPOs with prices to score." in text
    assert "No test year has prices for 60% of its IPOs." in text
    assert not (tmp_path / "bench_runs" / "ipo_holdout_log.jsonl").exists()
