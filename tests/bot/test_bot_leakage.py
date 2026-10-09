"""The accuracy-check fixes: IPO features as of their moment, the pop tolerance, the walk-forward split, the
publication-time leakage test (and the four leaks it must catch), the stock model's shrinkage, the hit rate."""

import datetime as dt
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from bot import evaluate, features, ipos, markets, models, stats

D = dt.date
NO_MACRO = pd.DataFrame(columns=["series", "date", "value", "available_at"])


def _filing(acc, form, filed):
    return {"accession": acc, "cik": "1", "form": form, "filed": filed, "company": "X Inc", "sic": "7372"}


AUDIT_FILINGS = [
    _filing("a1", "S-1", "2021-05-01"),
    _filing("a2", "S-1/A", "2021-06-01"),
    _filing("a3", "EFFECT", "2021-06-10"),
    _filing("a4", "424B4", "2021-06-14"),
]
AUDIT_DOCS = {
    "a2": json.dumps({"range": [14, 16], "shares": 10e6, "lead": "Jefferies"}),
    "a4": json.dumps({"offer": 19.0, "shares": 15e6, "lead": "Goldman Sachs"}),
}
AUDIT_TRADE = {"1": {"cik": "1", "symbol": "X", "date": "2021-06-11", "open": 25, "close": 27, "source": "yahoo"}}


# ---------------------------------------------------------------------------- IPO features as of the moment


def test_ipo_features_ignore_the_final_prospectus_filed_after_the_moment() -> None:
    d = ipos.build(AUDIT_FILINGS, AUDIT_DOCS, {}, AUDIT_TRADE)[0]
    assert (d.shares, d.lead) == (15e6, "Goldman Sachs")  # the latest values: the 424B4's
    assert d.asof("2021-06-10") == {"shares": 10e6, "lead": "Jefferies", "symbol": None, "foreign": False}
    r = features.ipo_rows([d], NO_MACRO, 0.2).iloc[0]
    assert r["moment"] == "2021-06-10" and r["log_size"] == pytest.approx(np.log(10e6 * 15))
    assert r["top_bank"] == 0.0  # Goldman Sachs appears only in the 424B4, filed after the prediction
    assert r["y"] == 1.0 and r["ret"] == pytest.approx(27 / 19 - 1)  # the outcome may use the 424B4


def test_a_moment_on_or_after_the_first_trade_is_dropped() -> None:
    no_effect = [f for f in AUDIT_FILINGS if f["form"] != "EFFECT"]
    d = ipos.build(no_effect, AUDIT_DOCS, {}, AUDIT_TRADE)[0]
    assert features.ipo_moment(d) is None and features.ipo_rows([d], NO_MACRO, 0.2).empty  # no EFFECT: not predicted
    same_day = {"1": {**AUDIT_TRADE["1"], "date": "2021-06-10"}}  # listed the day its EFFECT notice was filed
    assert features.ipo_rows(ipos.build(AUDIT_FILINGS, AUDIT_DOCS, {}, same_day), NO_MACRO, 0.2).empty
    # a range filed only after the moment is not a range the prediction could use
    late = ipos.build(AUDIT_FILINGS, {"a4": AUDIT_DOCS["a4"], "a2": "{}"}, {}, {})[0]
    late.ranges = [("2021-06-11", 14.0, 16.0)]
    assert features.ipo_rows([late], NO_MACRO, 0.2).empty
    untraded = ipos.build(AUDIT_FILINGS, AUDIT_DOCS, {}, {})
    assert len(features.ipo_rows(untraded, NO_MACRO, 0.2)) == 1


def test_a_hand_built_deal_keeps_its_foreign_flag() -> None:
    d = ipos.Deal(cik="1", company="X", foreign=True, registered="2025-01-02", shares=5.0, lead="Goldman Sachs")
    assert d.asof("2025-06-01") == {"shares": None, "lead": None, "symbol": None, "foreign": True}


# ---------------------------------------------------------------------------- the pop threshold


def test_popped_has_a_float_tolerance() -> None:
    assert 12 / 10 - 1 < 0.2 and features.popped(12 / 10 - 1, 0.2)
    assert not features.popped(0.1999, 0.2) and not features.popped(float("nan"), 0.2)
    trade = {"1": {**AUDIT_TRADE["1"], "close": 22.8}}  # 22.8 / 19 - 1 = 0.19999999999999996
    assert features.ipo_rows(ipos.build(AUDIT_FILINGS, AUDIT_DOCS, {}, trade), NO_MACRO, 0.2).iloc[0]["y"] == 1.0


# ---------------------------------------------------------------------------- the IPO walk-forward split


def test_a_test_quarter_deal_is_never_in_that_quarters_training_rows(monkeypatch) -> None:
    # ipo_rows keeps a deal only if its moment is before its first trade, so a deal predicted in the quarter
    # (A, B) trades in it too and is not among the rows traded before it
    n = 50
    rows = pd.DataFrame(
        {
            "cik": [str(i) for i in range(n)] + ["A", "B"],
            "company": "X",
            "moment": [f"2020-{1 + i // 5:02d}-10" for i in range(n)] + ["2021-04-01", "2021-04-05"],
            **{f: 0.0 for f in features.IPO_FEATURES},
            "y": [float(i % 2) for i in range(n)] + [1.0, 0.0],
            "ret": 0.1,
            "trade_date": [f"2020-{1 + i // 5:02d}-11" for i in range(n)] + ["2021-04-02", "2021-04-06"],
        }
    )
    seen = {}
    real = models.fit

    def spy(kind, df, feats, through, l2=1.0):
        seen[through] = set(df["cik"])
        return real(kind, df, feats, through, l2=l2)

    monkeypatch.setattr(models, "fit", spy)
    out = evaluate.ipo_walkforward(rows, min_train=20)
    assert "A" not in seen["2021-04-01"] and {"A", "B"} <= set(out["cik"])


# ---------------------------------------------------------------------------- the leakage test


def _deals(n=40):
    """n deals effective on consecutive trading days, each priced and listed the next day, with a 424B4 that
    raises the shares and names a top bank (facts the effective-date prediction must not see)."""
    days = markets.trading_days(D(2024, 1, 2), D(2024, 6, 28))
    fs, docs, trades = [], {}, {}
    rng = np.random.default_rng(3)
    for i in range(n):
        cik, eff, trade = str(100 + i), days[i + 5], days[i + 6]
        fs += [
            {"accession": f"{cik}r", "cik": cik, "form": "S-1", "filed": days[i].isoformat(), "company": f"C{i}"},
            {"accession": f"{cik}e", "cik": cik, "form": "EFFECT", "filed": eff.isoformat(), "company": f"C{i}"},
            {"accession": f"{cik}p", "cik": cik, "form": "424B4", "filed": trade.isoformat(), "company": f"C{i}"},
        ]
        lo = float(rng.uniform(10, 20))
        docs[f"{cik}r"] = json.dumps({"range": [lo, lo + 2], "shares": 10e6, "lead": "Jefferies"})
        hot = i % 3 == 0
        docs[f"{cik}p"] = json.dumps({"offer": lo + 3 if hot else lo, "shares": 15e6 if hot else 9e6, "lead": None})
        if i % 4 == 0:  # some prospectuses name a top bank, some name no lead at all
            docs[f"{cik}p"] = json.dumps({"offer": lo, "lead": "Goldman Sachs"})
        close = (lo + 3) * 1.5 if hot else lo * 1.02
        trades[cik] = {"date": trade.isoformat(), "open": lo, "close": close, "symbol": f"C{i}"}
    deals = ipos.build(fs, docs, {}, trades)
    return deals, [days[20 + 5].isoformat(), days[34 + 5].isoformat()]


def _leaky_ipo_rows(old, new):
    src = inspect.getsource(features)
    assert old in src
    ns = {"__name__": "features_leaky"}
    exec(compile(src.replace(old, new), "features_leaky", "exec"), ns)
    return ns["ipo_rows"]


def test_ipo_leakage_passes_on_the_fixed_code() -> None:
    deals, cuts = _deals()
    out = evaluate.leakage_ipos(deals, NO_MACRO, cuts, 0.2)
    assert out["passed"] and out["moment_not_before_trade"] == 0 and all(c["deals"] > 10 for c in out["checks"])


@pytest.mark.parametrize(
    "old, new",
    [
        ("for t, priced, r in traded if t < m and priced <= m]", "for t, priced, r in traded if t <= m]"),
        ('"range_rev": mid / mid0 - 1,', '"range_rev": (d.offer or mid) / mid0 - 1,'),
        ('math.log(known["shares"] * mid) if known["shares"]', "math.log(d.shares * mid) if d.shares"),
        ('float(known["lead"] in ipos.TOP_BANKS)', "float(d.lead in ipos.TOP_BANKS)"),
    ],
    ids=["heat_counts_same_day_listings", "range_rev_uses_final_offer", "424b4_shares", "424b4_lead"],
)
def test_ipo_leakage_catches_a_planted_leak(monkeypatch, old, new) -> None:
    deals, cuts = _deals()
    monkeypatch.setattr(features, "ipo_rows", _leaky_ipo_rows(old, new))
    out = evaluate.leakage_ipos(deals, NO_MACRO, cuts, 0.2)
    assert not out["passed"] and out["worst_diff"] > 0


def test_ipo_leakage_fails_when_a_row_is_predicted_after_its_first_trade(monkeypatch) -> None:
    deals, cuts = _deals()
    keep_late = _leaky_ipo_rows('if d.first_trade and d.first_trade["date"] <= m:', "if False:")
    for d in deals[:3]:
        d.effective = d.first_trade["date"]  # an EFFECT notice dated the day the stock listed
    monkeypatch.setattr(features, "ipo_rows", keep_late)
    out = evaluate.leakage_ipos(deals, NO_MACRO, cuts, 0.2)
    assert not out["passed"] and out["moment_not_before_trade"] == 3


def _closes_and_macro():
    idx = [d.isoformat() for d in markets.trading_days(D(2024, 1, 2), D(2024, 12, 31))]
    rng = np.random.default_rng(1)
    closes = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for s in ("A", "B", "SPX")}, index=idx
    )
    rows = []
    for i, d in enumerate(idx):
        nxt = (D.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
        rows += [
            {"series": "VIX", "date": d, "value": 15 + i % 7, "available_at": f"{d}T21:15:00Z"},
            # the Treasury curve is posted after the 22:00 UTC decision: a date-aligned join would peek at it
            {"series": "UST10Y", "date": d, "value": 4 + 0.01 * (i % 11), "available_at": f"{nxt}T00:30:00Z"},
            {"series": "UST3M", "date": d, "value": 5 - 0.01 * (i % 5), "available_at": f"{nxt}T00:30:00Z"},
        ]
    return closes, pd.DataFrame(rows), idx


def test_stock_leakage_passes_with_macro_on_the_fixed_code() -> None:
    closes, macro, idx = _closes_and_macro()
    assert evaluate.leakage_stocks(closes, macro, [idx[200], idx[230]])["passed"]


def test_stock_leakage_catches_macro_aligned_by_date(monkeypatch) -> None:
    closes, macro, idx = _closes_and_macro()

    def by_date(macro_rows, times):  # the leak: a value dated t is used at t's decision whatever its available_at
        out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(list(times), utc=True), name="t"))
        days = [t.date().isoformat() for t in times]
        for s, g in macro_rows.groupby("series"):
            val = g.drop_duplicates("date", keep="last").set_index("date")["value"].sort_index()
            out[s] = val.reindex(pd.Index(days)).ffill().to_numpy()
        return out

    monkeypatch.setattr(features, "macro_asof", by_date)
    assert not evaluate.leakage_stocks(closes, macro, [idx[200]])["passed"]


# ---------------------------------------------------------------------------- the stock model's shrinkage


def _noise_panel(sessions, signal):
    rng = np.random.default_rng(5)
    dates = [d.isoformat() for d in markets.trading_days(D(2015, 1, 2), D(2030, 1, 1))[:sessions]]
    n = len(dates) * 20
    x = rng.normal(size=(n, len(features.STOCK_FEATURES)))
    y = (rng.random(n) < 1 / (1 + np.exp(-signal * x[:, 0]))).astype(float)
    df = pd.DataFrame(x, columns=list(features.STOCK_FEATURES))
    return df.assign(date=np.repeat(dates, 20), symbol=np.tile([f"S{i}" for i in range(20)], len(dates)), y=y, ret=y)


def test_stock_model_is_shrunk_toward_its_base_rate_when_its_signal_is_noise() -> None:
    noise = _noise_panel(2 * models.CAL_SESSIONS, 0.0)
    m = models.fit("stock", noise, features.STOCK_FEATURES, "2016-01-01", l2=0.001)
    assert m.shrink < 1.0 and m.center == pytest.approx(np.log(noise["y"].mean() / (1 - noise["y"].mean())))
    raw = 1 / (1 + np.exp(-m.logit(noise)))
    assert np.std(m.prob(noise)) < np.std(raw)  # pulled toward the base rate
    back = models.Model.from_params(m.params())
    assert back.model_id == m.model_id and back.shrink == m.shrink
    strong = models.fit("stock", _noise_panel(2 * models.CAL_SESSIONS, 3.0), features.STOCK_FEATURES, "2016-01-01")
    assert (strong.shrink, strong.center) == (1.0, 0.0) and "shrink" not in strong.params()


def test_shrinkage_needs_a_long_enough_window_with_both_outcomes_early() -> None:
    short = _noise_panel(2 * models.CAL_SESSIONS - 1, 0.0)
    assert models.shrinkage(short, features.STOCK_FEATURES) == (1.0, 0.0)
    one_sided = _noise_panel(2 * models.CAL_SESSIONS, 0.0)
    cut = sorted(one_sided["date"].unique())[-models.CAL_SESSIONS]
    one_sided.loc[one_sided["date"] < cut, "y"] = 1.0
    assert models.shrinkage(one_sided, features.STOCK_FEATURES) == (1.0, 0.0)
    old = {"kind": "ipo", "features": ["a"], "mean": [0.0], "std": [1.0], "w_prob": [0.0, 1.0]}
    m = models.Model.from_params({**old, "w_value": [0.0, 0.0], "n_train": 30, "trained_through": "2025-01-01"})
    assert (m.shrink, m.center) == (1.0, 0.0)


# ---------------------------------------------------------------------------- the hit rate


def test_a_coin_flip_probability_is_half_a_hit() -> None:
    assert stats.hit_rate(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == 0.5
    assert stats.hit_rate(np.array([0.7, 0.2, 0.5, 0.6]), np.array([1, 0, 1, 0])) == (1 + 1 + 0.5 + 0) / 4
