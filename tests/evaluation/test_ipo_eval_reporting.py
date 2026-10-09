"""The IPO dataset and evaluation fixes: ranges read from stale covers and lost ligatures, assumed prices kept out
of above/below range, the pre-listing rule for the range everywhere, the price-bias and exclusion reporting,
per-fold calibration on training years only, and the range-revision-only baseline."""
import json
import math

import pytest

from test_ipo_data import PROSPECTUS, FakeSec, ipo, sub  # noqa: F401  (paths set there)
from test_ipo_eval_edges import SMALL, dataset, ipos, small_report  # noqa: F401  (small_report is a fixture)

import ipo_data as D  # noqa: E402
import ipo_eval as E  # noqa: E402

SNOWFLAKE = ("Proposed Maximum Offering Price Per Share (2) Class A common stock, par value $0.0001 per share 32,200,000 "
             "$110.00 It is currently estimated that the initial public offering price will be between $75.00 and $85.00 "
             "per share. Based on an assumed initial public offering price of $105.00 per share, which is the midpoint of "
             "the price range set forth on the cover page of this prospectus").encode()
LYFT = ("It is currently estimated that the initial public o_ering price per share will be between $70.00 and $72.00. "
        "based upon the assumed initial public offering price of $71.00 per share, which is the midpoint of the "
        "estimated offering price range").encode()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)


# ----------------------------------------------------------------------------- ranges, as the dataset stores them

def test_the_latest_range_records_where_it_came_from():
    filings = [("S-1", "2020-08-24", "acc-0", "d0.htm"), ("S-1/A", "2020-09-08", "acc-1", "d1.htm"),
               ("S-1/A", "2020-09-14", "acc-2", "d2.htm"), ("424B4", "2020-09-16", "acc-3", "d3.htm")]
    sec = FakeSec(heads={"/acc1/d1.htm": LYFT, "/acc2/d2.htm": SNOWFLAKE, "/acc0/d0.htm": b"between $ and $ per share"})
    assert D.latest_range(sec, "b", filings, "2020-09-16") == {"range": [100.0, 110.0], "range_date": "2020-09-14",
                                                               "range_source": "midpoint"}
    assert D.latest_range(sec, "b", filings, "2020-09-10") == {"range": [70.0, 72.0], "range_date": "2020-09-08",
                                                               "range_source": "stated"}
    assert D.latest_range(sec, "b", filings, "2020-09-01") == {"range": None, "range_date": None, "range_source": None}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path / "ipo" / "ipos.jsonl")
    monkeypatch.setattr(D, "INITIAL", tmp_path / "ipo" / "initial_ranges.json")
    D.OUT.parent.mkdir(parents=True)
    return tmp_path


def test_refresh_ranges_rereads_ranges_and_only_the_offers_they_no_longer_fit(paths, monkeypatch, capsys):
    prices = {"listing_date": "2020-03-02", "open": 30.0, "close": 33.0, "suspect": True}
    rows = [ipo("0000000042-20-000001", "42", "2020-03-01", file="p.htm", offer_price=40.0, range=[71.0, 71.0],
                range_date="2020-02-01", prices=prices, sources={"prospectus": "2020-03-01", "range": "2020-02-01"}),
            ipo("0000000043-20-000001", "43", "2020-03-01", file="p.htm", offer_price=40.0, prices=None),
            ipo("0000000044-20-000001", "44", "2020-03-01", file="p.htm", offer_price=40.0,
                prices={"listing_date": "2020-03-02", "close": 1.0}),
            ipo("0000000045-20-000001", "45", "2020-03-01", offer_price=72.0),
            ipo("gone", "46", "2020-03-01")]
    rows += [ipo(f"n{i}", str(100 + i), "2020-03-01") for i in range(95)]
    D.OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    page = sub(["S-1/A", "424B4"], ["2020-02-20", "2020-03-01"])
    pages = {f"CIK{int(c):010d}.json": page for c in ("42", "43", "44", "45")}
    pages.update({f"CIK{100 + i:010d}.json": sub(["424B4"], ["2020-03-01"]) for i in range(95)})
    heads = {"/acc0/d0.htm": "between $9.00 and $11.00 per share".encode(), "/p.htm": PROSPECTUS}
    monkeypatch.setattr(D, "Sec", lambda: FakeSec(pages, heads))
    D.refresh_ranges()
    by = {r["adsh"]: r for r in (json.loads(x) for x in D.OUT.read_text(encoding="utf-8").splitlines()[len(rows):])}
    a = by["0000000042-20-000001"]
    assert a["range"] == [9.0, 11.0] and a["range_date"] == "2020-02-20" and a["range_source"] == "stated"
    assert a["sources"] == {"prospectus": "2020-03-01", "range": "2020-02-20"}
    assert a["offer_price"] == 10.0 and a["prices"]["suspect"] is False           # $40 did not fit: re-read; 30/10 = 3x
    assert rows[0]["prices"]["suspect"] is True and by["0000000043-20-000001"]["offer_price"] == 10.0
    assert by["0000000044-20-000001"]["prices"] == {"listing_date": "2020-03-02", "close": 1.0}   # no open: left
    assert by["0000000045-20-000001"]["offer_price"] == 72.0                        # no prospectus file: left as it was
    assert "no page" in by["gone"]["refresh_error"]
    assert by["n0"]["range"] is None and "range" not in by["n0"].get("sources", {})
    assert "re-reading the ranges of 100 IPO rows" in capsys.readouterr().out
    called = []
    monkeypatch.setattr(D, "refresh_ranges", lambda: called.append("ranges"))
    D.main(["--ranges"])
    assert called == ["ranges"]


def test_an_offer_that_no_longer_reads_is_marked_suspect(paths, monkeypatch):
    rows = [ipo("0000000042-20-000001", "42", "2020-03-01", file="p.htm", offer_price=400.0,
                prices={"listing_date": "2020-03-02", "open": 30.0, "close": 33.0, "suspect": False})]
    D.OUT.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    heads = {"/acc0/d0.htm": b"between $9.00 and $11.00 per share", "/p.htm": b"no price here"}
    monkeypatch.setattr(D, "Sec", lambda: FakeSec({"CIK0000000042.json": sub(["S-1/A"], ["2020-02-20"])}, heads))
    D.refresh_ranges()
    new = json.loads(D.OUT.read_text(encoding="utf-8").splitlines()[-1])
    assert new["offer_price"] is None and new["prices"]["suspect"] is True


# ----------------------------------------------------------------------------- the range, as the model reads it

def test_a_range_filed_on_the_listing_day_cannot_flag_the_offer_as_a_misread():
    r = ipos("2019-01-01", "2019-02-01", 1)[0]
    late = dict(r, offer_price=r["range"][1] * 5, range_date=r["prices"]["listing_date"])
    assert E.prelisting_range(late) is None and not E.offer_off_range(late)
    assert E.offer_off_range(dict(late, range_date="2018-01-01"))
    assert E.prelisting_range(dict(r, prices=None)) == r["range"]                 # not listed yet: the range stands


def test_an_assumed_price_is_a_reference_not_a_range_to_beat():
    lyft = {"adsh": "l", "offer_price": 72.0, "range": [71.0, 71.0], "range_date": "2019-03-27",
            "prices": {"listing_date": "2019-03-29"}}
    f = E.features(lyft, {}, {})
    assert f["above_range"] == 0.0 and f["below_range"] == 0.0 and f["revision"] == pytest.approx(72 / 71 - 1)
    assert E.features(dict(lyft, range=[70.0, 72.0], range_source="stated", offer_price=72.5), {}, {})["above_range"] == 1.0
    assert E.features(dict(lyft, range=[70.0, 72.0], range_source="assumed", offer_price=60.0), {}, {})["below_range"] == 0.0
    assert E.features(dict(lyft, range=[100.0, 110.0], range_source="midpoint", offer_price=120.0), {}, {})["above_range"] == 1.0


def test_the_live_scorer_keeps_an_assumed_price_out_of_above_range():
    class Echo:
        def prob(self, f):
            return f["above_range"]
    ev = {"prospectus": "The initial public offering price is $12.00 per share.",
          "registration": "an assumed initial public offering price of $10.00 per share"}
    assert E.score_live(ev, Echo(), {}, {})["p_pop"] == 0.0
    ev["registration"] = "the offering price will be between $9.00 and $11.00 per share"
    assert E.score_live(ev, Echo(), {}, {})["p_pop"] == 1.0


# ----------------------------------------------------------------------------- calibration and comparisons

def test_platt_is_fitted_on_training_years_only_and_skipped_without_enough():
    rows = ipos("2015-01-01", "2018-12-31", 240, seed=4)
    market = E.market_features(rows, {})
    train = [r for r in rows if r["prices"]["listing_date"] < "2018-01-01"]
    cal = E.platt(train, market, SMALL)
    assert cal is not None and cal.prob({"z": 2.0}) > cal.prob({"z": -2.0})
    assert E.platt(train[-70:], market, SMALL) is None                               # under 50 IPOs before the last year
    flat = [dict(r, prices={**r["prices"], "close": r["offer_price"]}) if r["prices"]["listing_date"] >= "2017" else r
            for r in train]
    assert E.platt(flat, market, SMALL) is None                                       # the last year never pops
    folds = E.walk_forward(rows, market, years=range(2016, 2019), names=SMALL, calibrate=True)
    assert [f["year"] for f in folds] == [2016, 2017, 2018]
    assert folds[0]["calibrator"] is None and folds[0]["calibrated"] == folds[0]["scores"]   # 2015 alone: no inner split
    assert folds[-1]["calibrator"] is not None and folds[-1]["calibrated"] != folds[-1]["scores"]
    test_rows = {r["adsh"] for r in folds[-1]["rows"]}
    seen = []
    platt = E.platt

    def spy(train, market, names, l2):
        seen.append({r["adsh"] for r in train})
        return platt(train, market, names, l2)
    E.platt, saved = spy, E.platt
    try:
        E.walk_forward(rows, market, years=[2018], names=SMALL, calibrate=True)
    finally:
        E.platt = saved
    assert seen and not seen[0] & test_rows                                           # never a test IPO


def test_paired_comparisons_and_reliability():
    a, b, ys = [0.9, 0.8, 0.3, 0.2, 0.7, 0.1], [0.5, 0.4, 0.6, 0.5, 0.4, 0.6], [1, 1, 0, 0, 1, 0]
    d, (lo, hi), p = E.paired_bootstrap(a, b, ys, n=200)
    assert d > 0 and lo <= d <= hi and 0 <= p < 0.5
    d, (lo, hi), p = E.paired_bootstrap([0.1], [0.2], [1], n=20)                     # one IPO: no usable resample
    assert d == 0.0 and math.isnan(lo) and math.isnan(p)
    m, t, pv = E.diebold_mariano(a, b, ys, lag=1)
    assert m < 0 and t < 0 and 0 < pv < 1
    assert math.isnan(E.diebold_mariano([0.1, 0.2], [0.3, 0.4], [0, 1])[2])        # too few for a t
    rel = E.reliability([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], bins=2)
    assert rel == [(2, pytest.approx(0.15), 0.0), (2, pytest.approx(0.35), 1.0)]
    assert len(E.reliability([0.5], [1], bins=10)) == 1                               # empty deciles are skipped


def test_the_exclusion_table_says_when_a_sample_cannot_be_scored():
    text = "\n".join(E.exclusion_lines([], []))
    assert text.count("| 0 | too few to score |") == 2


def test_who_is_missing_with_nobody_missing():
    rows = ipos("2019-01-01", "2019-06-01", 5)
    rows[0] = dict(rows[0], shares=None)
    text = "\n".join(E.sample_bias_lines(rows))
    assert "| IPOs with an offer price | 5 | 0 |" in text and "| median offer price |" in text
    assert text.count("n/a") >= 10                                                     # every unpriced cell


def test_the_report_states_the_bias_the_exclusion_the_baseline_and_the_calibration(small_report):
    text = small_report(dataset(unpriced=30))
    assert "| 2015 |" in text and "of the IPOs have no Yahoo prices" in text
    assert "## Who is missing: pre-listing facts, priced against unpriced" in text
    assert "| Before the listing | with prices | no prices |" in text
    assert "| main: misread offers out (pre-listing facts only) |" in text
    assert "| also without opens outside 0.5-4x the offer (reads the listing day) |" in text
    assert "## 1. Against the range revision alone" in text and "| range revision only |" in text
    assert "Diebold-Mariano t" in text and "paired bootstrap 95% CI" in text
    assert "## 1. Calibration" in text and "| Decile | raw: predicted |" in text and "| 10 |" in text
    assert "nan" not in text
