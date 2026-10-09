"""The point-in-time harness's edges (bot_benchmark.py) and ipo-bot's own bots in it (real_setup.py), offline:
the IPO dataset, SPY and the stock panel are small fakes."""
import datetime as dt
import math
import pathlib
import random
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "evaluation" / "harness"))

import benchmark  # noqa: E402
import bot_benchmark as bb  # noqa: E402
import ipo_data  # noqa: E402
import ipo_eval  # noqa: E402
import real_setup as RS  # noqa: E402

T0 = pd.Timestamp("2020-01-01")


def small_store(dates=12, names=6, seed=0):
    """`names` stocks a day for `dates` days: a close each morning, the next day's return as the outcome."""
    rng = np.random.default_rng(seed)
    data, events = [], []
    for d in range(dates):
        t = T0 + pd.Timedelta(days=d)
        for k in range(names):
            ent = f"S{k}"
            data.append((t, ent, "close", 100 + d + k, "market"))
            eid = f"{ent}@{d}"
            data.append((t + pd.Timedelta(days=1), eid, "outcome", float(rng.normal(0.001 * k, 0.02)), "market"))
            events.append(bb.Event(eid, "market", ent, t + pd.Timedelta(hours=1), t + pd.Timedelta(days=1)))
    return bb.PointInTimeData(pd.DataFrame(data, columns=bb.STORE_COLUMNS)), events


# ----------------------------------------------------------------------------- bot_benchmark

def test_views_hold_only_the_past_and_say_so_when_asked_for_more():
    store, _ = small_store(dates=3)
    v = store.view(T0 + pd.Timedelta(days=1))
    assert v.frame["available_at"].max() <= v.as_of and len(store.frame) == 36
    assert set(v.subset(["S0"], ["close"])["entity"]) == {"S0"}
    assert v.value_at("S1", "close", T0) == 101.0 and math.isnan(v.value_at("nobody", "close", T0))
    with pytest.raises(bb.LookAheadError):
        v.value_at("S1", "close", T0 + pd.Timedelta(days=2))
    with pytest.raises(ValueError, match="missing columns"):
        bb.PointInTimeData(pd.DataFrame({"entity": []}))


def test_bad_bot_answers_are_refused():
    evs = [bb.Event("a", "market", "x", T0, T0 + pd.Timedelta(days=1)), bb.Event("b", "market", "x", T0, T0 + pd.Timedelta(days=1))]
    df = lambda ids, p: pd.DataFrame({"event_id": ids, "prob_up": p})   # noqa: E731
    for out, msg in ((None, "must return a DataFrame"), (df(["a", "a"], [0.5, 0.5]), "duplicate"), (df(["a"], [0.5]), "skipped"),
                     (df(["a", "b", "c"], [0.5] * 3), "not asked about"), (df(["a", "b"], [0.5, 1.5]), "outside")):
        with pytest.raises(ValueError, match=msg):
            bb._validate(out, evs, "bot")
    ok = bb._validate(df(["a", "b"], [0.2, 0.8]), evs, "bot")
    assert ok["expected_return"].isna().all()
    with pytest.raises(ValueError, match="resolves before"):
        bb.group_by_time([bb.Event("z", "market", "x", T0, T0)])


def test_cut_dates_outcomes_and_metrics_at_their_edges():
    store, events = small_store(dates=3)
    assert len(bb.pick_cut_dates(events, n=10)) == 2                  # more chunks than dates: empty ones skipped
    preds = bb.run_walk_forward(lambda s: bb.CoinFlipBot(), store, events)
    ghost = bb.Event("ghost", "market", "x", T0, T0 + pd.Timedelta(days=1))
    extra = pd.DataFrame([{"event_id": "ghost", "prob_up": 0.5, "expected_return": 0.0, "bot": "coin_flip", "as_of": T0}])
    with pytest.raises(ValueError, match="no outcome stored"):
        bb.attach_outcomes(pd.concat([preds, extra], ignore_index=True), store, events + [ghost])
    assert abs(bb.log_loss([0.5, 0.5], [1, 0]) - math.log(2)) < 1e-12
    assert bb.auc([0.1, 0.9], [0, 1]) == 1.0 and math.isnan(bb.auc([0.1, 0.9], [1, 1]))
    assert bb.oos_r2([0.1], [0.1], benchmark=[0.0]) == 1.0 and math.isnan(bb.oos_r2([0.1], [0.0]))
    assert bb.hit_rate([0.5, 0.9], [1, 1]) == 0.75
    few = pd.DataFrame({"as_of": [T0] * 5, "expected_return": [1, 2, 3, 4, 5], "outcome": [1, 2, 3, 4, 5]})
    assert bb.information_coefficient(few)[2] == 1 and math.isnan(bb.information_coefficient(few)[0])


def test_the_t_distribution_falls_back_to_the_normal_without_scipy(monkeypatch):
    monkeypatch.setattr(bb, "_sps", None)
    assert bb._t_cdf(0.0, 5) == 0.5 and abs(bb._t_cdf(1.96, 5) - 0.975) < 1e-3


def test_diebold_mariano_edge_cases():
    s = lambda xs: pd.Series(xs, index=range(len(xs)))   # noqa: E731
    with pytest.raises(ValueError, match="need at least 10"):
        bb.diebold_mariano(s([0.1] * 5), s([0.2] * 5))
    flat = bb.diebold_mariano(s([0.3] * 12), s([0.1] * 12))           # the same gap every date: no variance
    assert math.isnan(flat["dm_stat"]) and abs(flat["mean_diff"] - 0.2) < 1e-12
    out = bb.diebold_mariano(s([0.1, 0.3] * 6), s([0.2] * 12), lag=1)
    assert math.isfinite(out["dm_stat"]) and abs(out["mean_diff"]) < 1e-12 and 0 < out["p_value"] < 1


def test_score_tables_reports_slices_and_long_short_on_a_small_run():
    store, events = small_store(dates=12, names=6)
    preds = pd.concat([bb.run_walk_forward(f, store, events) for f in
                       (lambda s: bb.CoinFlipBot(), lambda s: bb.BaseRateBot(min_history=6))], ignore_index=True)
    joined = bb.attach_outcomes(preds, store, events)
    table = bb.score_table(joined)
    assert list(table["bot"]) == ["base_rate", "coin_flip"] and (table["n"] == 72).all()
    assert table.loc[table["bot"] == "coin_flip", "brier"].iloc[0] > 0
    sl = bb.slice_comparison(joined, "coin_flip", "base_rate", "year")
    assert list(sl["year"]) == [2020] and sl["diff"].iloc[0] == sl["candidate"].iloc[0] - sl["opponent"].iloc[0]
    none = bb.slice_comparison(joined, "coin_flip", "nobody", "year")
    assert none.empty and "diff" in none.columns
    assert len(bb.long_short_returns(joined, "base_rate")) == 12
    thin = joined[joined["event_id"].str.startswith(("S0", "S1"))]
    assert bb.long_short_returns(thin, "base_rate").empty          # under five names a date: no portfolio
    text = bb.format_report(joined, "base_rate", ["coin_flip"])
    assert "Head to head, base_rate vs others" in text and "By kind:" in text


# ----------------------------------------------------------------------------- real_setup: IPOs

def ipo_rows(n=160, seed=5):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        day = dt.date(2015, 1, 5) + dt.timedelta(days=int(i * 2000 / n))
        lo = rng.uniform(8, 30)
        rev = rng.gauss(0, 0.12)
        offer = round((lo + 2) * (1 + rev), 2)
        first = rng.uniform(0.25, 0.9) if rng.random() < 1 / (1 + math.exp(-(-1 + 20 * rev))) else rng.uniform(-0.2, 0.15)
        rows.append({"adsh": f"a{i}", "cik": str(i), "offer_price": offer, "shares": rng.choice([None, 5e6, 2e7]),
                     "lead_bank": rng.choice(["GS", "MS", None]), "foreign": i % 5 == 0, "sic": rng.choice(["7372", "2834", "x", None]),
                     "range": [lo, lo + 4], "prospectus_date": day.isoformat(),
                     "range_date": (day - dt.timedelta(days=9 if i % 7 else -1)).isoformat(),   # a few filed too late to use
                     "prices": {"listing_date": day.isoformat(), "open": offer * (1 + first), "close": offer * (1 + first),
                                "suspect": False}})
    rows.append(dict(rows[0], adsh="held", prices={**rows[0]["prices"], "listing_date": "2024-03-01"}))   # sealed
    return rows


@pytest.fixture
def ipo_world(monkeypatch):
    rows = ipo_rows()
    monkeypatch.setattr(ipo_data, "load", lambda: rows)
    spy = {(dt.date(2014, 6, 1) + dt.timedelta(days=k)).isoformat(): 100 + 0.05 * k for k in range(4000)}
    monkeypatch.setattr(ipo_eval, "spy_closes", lambda: spy)
    monkeypatch.setattr(RS, "MIN_IPO_HISTORY", 60)                     # a smaller market fits sooner
    return rows


def test_fit_logit_stopped_early_still_gives_probabilities():
    X = np.array([[x] for x in np.linspace(-2, 2, 40)])
    p = RS.fit_logit(X, (X[:, 0] > 0).astype(float), iters=1)(X)
    assert p[0] < 0.5 < p[-1] and ((0 < p) & (p < 1)).all()


def test_the_ipo_store_stamps_facts_before_the_prediction_and_leaves_out_the_holdout(ipo_world):
    store, events, banks = RS.ipo_store()
    assert banks == ["GS", "MS"] and len(events) == 160 and not any("held" in e.event_id for e in events)
    f = store.frame
    facts = f[f["field"] != "outcome"]
    first = events[0]
    assert facts[facts["entity"] == first.entity]["available_at"].max() < first.as_of < first.resolve_at
    late = f[(f["entity"] == "a0") & (f["field"].isin(["range_lo", "sic"]))]["value"].tolist()
    assert late[0] == -1.0                                            # a0's range was filed after it listed: not a fact
    assert (f[(f["field"] == "sic")]["value"] == -1.0).any()         # an unreadable SIC is -1


def test_the_ipo_bots_learn_only_from_their_own_log_and_refit_monthly(ipo_world):
    setup = RS.ipo_setup()
    store, events = setup["store"], setup["events"]
    for name, factory in [("ipo_model", setup["candidate"]), *setup["rivals"].items(), *setup["baselines"].items()]:
        preds = bb.run_walk_forward(factory, store, events)
        assert len(preds) == len(events) and preds["prob_up"].between(0.01, 0.99).all(), name
        if name in ("ipo_model", "revision_only"):
            early, late = preds["prob_up"].iloc[:RS.MIN_IPO_HISTORY], preds["prob_up"].iloc[-60:]
            assert early.iloc[0] == 0.5 and late.nunique() > 10      # base rate first, then a fitted model
    bot = setup["candidate"](store)
    cut = events[150].as_of
    bot.predict(store.view(cut), [e for e in events if e.as_of == cut])
    bot.predict(store.view(cut), [e for e in events if e.as_of == cut])   # the same month again: no refit
    assert bot.fitted_month == cut.strftime("%Y-%m") and bot.model is None   # nothing it logged has resolved yet


# ----------------------------------------------------------------------------- real_setup: stocks, and both

@pytest.fixture
def market_world(monkeypatch):
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    monkeypatch.setitem(benchmark.UNIVERSES, "US large caps", symbols)
    rng = random.Random(9)
    months = [f"{2015 + k // 12}-{k % 12 + 1:02d}" for k in range(75)]
    px = {s: [] for s in symbols}
    for s in symbols:
        p = 100.0
        for _ in months:
            p *= 1 + rng.gauss(0.005, 0.05)
            px[s].append(p)
    monkeypatch.setattr(benchmark, "panel", lambda syms: (months, px))
    monkeypatch.setattr(RS, "MIN_MARKET_HISTORY", 20)
    return symbols, months


def test_the_market_store_and_algorithm_bots(market_world):
    symbols, months = market_world
    store, events = RS.market_store()
    assert len(events) == (75 - 1 - benchmark.LOOKBACK) * 4 and RS._month_end("2016-02") == pd.Timestamp(2016, 2, 29)
    setup = RS.market_setup()
    assert set(setup["rivals"]) == set(benchmark.ALGORITHMS)
    for factory in [setup["candidate"], *setup["rivals"].values()]:
        preds = bb.run_walk_forward(factory, store, events)
        assert preds["prob_up"].between(0.01, 0.99).all() and preds["expected_return"].notna().all()
        first, last = preds.iloc[:4], preds.iloc[-4:]
        assert first["prob_up"].nunique() == 1 and first["prob_up"].iloc[0] == 0.5   # nothing published yet
        assert last["expected_return"].nunique() > 1                                # a fitted map from rank


def test_real_setup_answers_both_kinds_with_one_bot(ipo_world, market_world):
    setup = RS.real_setup()
    kinds = {e.kind for e in setup["events"]}
    assert kinds == {"ipo", "market"} and len(setup["rivals"]) == len(benchmark.ALGORITHMS)
    bot = setup["candidate"](setup["store"])
    assert bot.name == "revision_model+blend" and bot.knowledge_cutoff < setup["store"].frame["available_at"].min()
    ipo_only = [e for e in setup["events"] if e.kind == "ipo"][:1]
    out = bot.predict(setup["store"].view(ipo_only[0].as_of), ipo_only)
    assert list(out.columns) == ["event_id", "prob_up", "expected_return"] and out["expected_return"].isna().all()
    t = max(e.as_of for e in setup["events"] if e.kind == "market")
    month = [e for e in setup["events"] if e.kind == "market" and e.as_of == t]
    out = next(iter(setup["rivals"].values()))(setup["store"]).predict(setup["store"].view(t), month)
    assert len(out) == 4 and out["expected_return"].notna().all()
    for f in setup["baselines"].values():
        assert f(setup["store"]).name in ("base_rate", "coin_flip")
