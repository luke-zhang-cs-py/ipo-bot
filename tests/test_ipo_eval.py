"""The IPO dataset's parsers and the IPO evaluation's checks, offline, on a synthetic IPO market with a known signal."""
import datetime as dt
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import ipo_data as D  # noqa: E402
import ipo_eval as E  # noqa: E402


# ----------------------------------------------------------------------------- parsers

def test_offer_price_range_shares_and_bank_from_prospectus_text():
    cover = D.text_of(b"<p>200,000,000 Shares</p><p>Class A Common Stock</p><p>This is an initial public offering."
                      b" The initial public offering price is $17.00 per share.</p><p>Morgan Stanley&nbsp;Goldman Sachs &amp; Co.</p>")
    assert D.offer_price(cover) == 17.0
    assert D.shares_offered(cover) == 200_000_000
    assert D.lead_bank(cover) == "Morgan Stanley"
    assert D.price_range(D.text_of("We anticipate that the initial public offering price will be between $14.00 and $16.00 per share.")) == (14.0, 16.0)
    # Carvana's wording: no "per share", entities for the space and the quote
    assert D.price_range(D.text_of(b"the initial public offering price per share will be between $14.00&nbsp;and $16.00. Carvana Co&#146;s")) == (14.0, 16.0)
    assert D.price_range(D.text_of("an assumed initial public offering price of $5.00 per share")) == (5.0, 5.0)
    assert D.price_range(D.text_of("salaries between $100.00 and $200.00 a year")) is None       # no offering price nearby
    assert D.offer_price(D.text_of("no price here")) is None and D.lead_bank("nobody") is None
    assert D.lead_bank("Merrill Lynch, Pierce, Fenner & Smith") == "BofA Securities"            # the old name, read as the new
    # the symbol it listed under, from the prospectus: the SEC's record holds today's ticker, or none once delisted
    assert D.listing_symbol("approved for listing on the NYSE under the symbol “SNAP.”") == "SNAP"
    assert D.listing_symbol('under the symbol "FIT"') == "FIT" and D.listing_symbol("under the trading symbol “BRK.B”") == "BRK.B"
    assert D.listing_symbol("no symbol named") is None


def _chart(days, opens, closes, splits=None):
    ts = [int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()) for d in days]
    out = {"timestamp": ts, "indicators": {"quote": [{"open": opens, "close": closes}]}}
    if splits:
        out["events"] = {"splits": {str(i): {"date": int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()),
                                             "numerator": n, "denominator": m} for i, (d, n, m) in enumerate(splits)}}
    return out


def test_first_days_undo_later_splits_and_flag_what_cannot_be_a_listing():
    days = [(dt.date(2021, 10, 27) + dt.timedelta(days=i)).isoformat() for i in range(300)]
    # a 1-for-20 reverse split later: Yahoo shows $460 for a $23 open; multiplied back by 1/20
    c = _chart(days, [460.0] * 300, [385.8] * 300, splits=[("2023-05-01", 1, 20)])
    out = D.first_days(c, "2021-10-27", 21.0)
    assert out["open"] == 23.0 and abs(out["close"] - 19.29) < 1e-9 and out["suspect"] is False
    assert out["close_21"] and out["close_252"]
    # history that starts weeks away from the prospectus is not this IPO
    assert D.first_days(c, "2021-08-01", 21.0) is None
    # an open at a fifth of the offer with no split on record (Carvana on Yahoo) is kept, marked suspect
    assert D.first_days(_chart(days[:5], [2.7] * 5, [2.2] * 5), "2021-10-27", 15.0)["suspect"] is True


# ----------------------------------------------------------------------------- a synthetic IPO market

def market(n=900, seed=1, signal=2.5):
    """IPOs 2015-2025 whose pop odds rise with the range revision (and nothing else): the model should find it."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        day = dt.date(2015, 1, 5) + dt.timedelta(days=int(i * 4000 / n))
        lo = rng.uniform(8, 30)
        rev = rng.gauss(0, 0.12)
        offer = round((lo + 2) * (1 + rev), 2)
        p = 1 / (1 + math.exp(-(-1.6 + signal * rev / 0.12)))
        pop = rng.random() < p
        first = rng.uniform(0.25, 0.9) if pop else rng.uniform(-0.25, 0.15)
        open_ = offer * (1 + first * rng.uniform(0.6, 1.1))
        rows.append({"adsh": f"x{i}", "company": f"Co {i}", "ticker": f"C{i}", "sic": rng.choice(["7372", "2834", "6022", "5500"]),
                     "prospectus_date": (day + dt.timedelta(days=1)).isoformat(), "offer_price": offer, "shares": rng.randint(2, 30) * 1e6,
                     "lead_bank": rng.choice(["Goldman Sachs", "Morgan Stanley", "Maxim Group", "Boustead"]), "foreign": rng.random() < 0.2,
                     "range": [lo, lo + 4], "range_date": (day - dt.timedelta(days=9)).isoformat(),
                     "prices": {"listing_date": day.isoformat(), "open": open_, "close": offer * (1 + first),
                                "close_21": offer * (1 + first) * rng.uniform(0.8, 1.2), "close_252": None, "suspect": False}})
    return rows


ROWS = market()
SPY20 = {}
MARKET = E.market_features(ROWS, SPY20)


def test_the_holdout_is_sealed_until_final():
    dev, hold = E.split(ROWS)
    assert hold == [] and all(r["prices"]["listing_date"] < E.HOLDOUT_FROM for r in dev)
    dev2, hold2 = E.split(ROWS, final=True)
    assert hold2 and all(r["prices"]["listing_date"] >= E.HOLDOUT_FROM for r in hold2) and len(dev2) == len(dev)


def test_walk_forward_trains_only_on_earlier_years_and_finds_the_signal():
    dev, _ = E.split(ROWS)
    folds = E.walk_forward(dev, MARKET)
    assert [f["year"] for f in folds] == list(E.TEST_YEARS) == list(range(2018, 2024))
    for f in folds:
        assert f["train_last"] < f"{f['year']}-01-01"                                 # nothing from the test year or later
        assert all(r["prices"]["listing_date"][:4] == str(f["year"]) for r in f["rows"])
    s, y, _ = E.pooled(folds)
    base = sum(y) / len(y)
    assert E.average_precision(s, y) > base + 0.15 and E.roc_auc(s, y) > 0.7
    assert E.permutation_p(s, y, n=200) < 0.01


def test_a_market_with_no_signal_scores_like_chance():
    dev, _ = E.split(market(signal=0.0, seed=4))
    folds = E.walk_forward(dev, E.market_features(dev, {}))
    s, y, _ = E.pooled(folds)
    assert abs(E.roc_auc(s, y) - 0.5) < 0.08
    assert E.permutation_p(s, y, n=200) > 0.01


def test_metrics_that_respect_imbalance():
    ys = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    perfect = [0.9, 0.1, 0.2, 0.1, 0.8, 0.3, 0.1, 0.2, 0.1, 0.0]
    assert E.average_precision(perfect, ys) == 1.0 and E.roc_auc(perfect, ys) == 1.0
    assert abs(E.average_precision([0.5] * 10, ys) - 0.2) < 0.2       # ties: about the base rate
    card = E.scorecard([0.0] * 10, ys, 0.5, 0.2)                        # always no: 80% accurate, useless
    assert card["accuracy"] == 0.8 == card["accuracy_always_no"] and card["f1"] == 0.0 and card["recall"] == 0.0
    assert E.f1_at(perfect, ys, 0.5)[0] == 1.0


def test_the_bank_league_table_uses_only_the_training_years():
    train = [r for r in ROWS if r["prices"]["listing_date"] < "2018-01-01"]
    shares = E.bank_shares(train)
    assert abs(sum(shares.values()) - 1) < 1e-9
    changed = [dict(r, lead_bank="Brand New Bank") if r["prices"]["listing_date"] >= "2018-01-01" else r for r in ROWS]
    assert E.bank_shares([r for r in changed if r["prices"]["listing_date"] < "2018-01-01"]) == shares


def test_market_features_use_only_earlier_listings():
    by = sorted(ROWS, key=lambda r: r["prices"]["listing_date"])
    first = by[0]
    assert MARKET[first["adsh"]]["heat_30d"] == 0.0 and MARKET[first["adsh"]]["deals_30d"] == 0.0
    later = by[40]
    window = [x for x in by if (dt.date.fromisoformat(later["prices"]["listing_date"]) - dt.timedelta(days=30)).isoformat()
              <= x["prices"]["listing_date"] < later["prices"]["listing_date"]]
    assert abs(MARKET[later["adsh"]]["heat_30d"] - sum(E.first_day(x) for x in window) / len(window)) < 1e-12


def test_the_leakage_audit_passes_the_real_features_and_catches_a_planted_leak():
    dev = [r for r in ROWS if r["prices"]["listing_date"] < "2017-01-01"]
    audit = E.leakage_audit(dev)
    assert audit["leaks"] == [] and audit["range_after_listing"] == [] and audit["prospectus_late"] == []
    assert audit["canary_caught"] is True
    leaky = E.perturbation_leaks(dev, lambda rs: {r["adsh"]: {"x": r["prices"]["close"]} for r in rs}, samples=5)
    assert leaky, "a feature built from the listing day's close must be flagged"


def test_sensitivity_ranks_the_real_signal_first_and_noise_near_zero():
    dev, _ = E.split(ROWS)
    folds = E.walk_forward(dev, MARKET)
    base, perm = E.sensitivity(folds, repeats=5)
    assert max(perm, key=perm.get) == "revision" and perm["revision"] > 0.05
    assert all(abs(v) < 0.05 for k, v in perm.items() if k not in ("revision", "above_range", "below_range"))
    _, drop = E.drop_one(dev, MARKET, folds)
    assert drop["revision"] > 0.02 and abs(drop["noise (added)"]) < 0.03


def test_slippage_only_costs_and_scales_with_its_size():
    dev, _ = E.split(ROWS)
    folds = E.walk_forward(dev, MARKET)
    runs = [E.at_open(folds, s) for s in E.SLIPPAGE]
    for key in runs[0]:
        means = [r[key]["mean"] for r in runs]
        assert means == sorted(means, reverse=True), key
        assert means[0] - means[-1] > 0.03                    # 2% each way is about 4% a round trip


def test_the_winners_curse_fills_flops_in_full_and_hot_deals_thinly():
    assert E.allocation(-0.1, 5) == 1.0 and E.allocation(0.0, 5) == 1.0
    assert E.allocation(0.5, 5) < E.allocation(0.1, 5) < 1.0 and E.allocation(50, 10) == 0.02
    dev, _ = E.split(ROWS)
    out = E.winners_curse(E.walk_forward(dev, MARKET))
    for v in out.values():
        allocated = [v[f"allocated k={k}"] for k in E.CURSE_K]
        assert all(a < v["as asked"] for a in allocated)       # rationing the winners lowers the real return
        assert allocated == sorted(allocated, reverse=True)    # the steeper the rationing, the lower


def test_the_live_feed_never_crashes_and_stays_fast():
    dev, _ = E.split(ROWS)
    model, banks, _, _ = E.fit_year(dev, MARKET)
    st = E.stress(model, banks, n=400, burst=50)
    assert st["errors"] == 0, st["error_examples"]
    assert st["ok"] > 100 and st["unknown"] > 20               # broken feeds say "unknown"; good ones score
    assert st["p99_ms"] < st["budget_ms"]
    good = E.score_live({"prospectus": "The initial public offering price is $20.00 per share.", "opening": {"indication": 25.0}},
                        model, banks, {"spy_20d": 0, "heat_30d": 0, "deals_30d": 0})
    assert good["status"] == "ok" and abs(good["indicated_open_vs_offer"] - 0.25) < 1e-12 and 0 < good["p_pop"] < 1
    assert E.score_live({}, model, banks, {})["status"] == "unknown"


def test_reoptimising_each_year_looks_only_at_training_years():
    dev, _ = E.split(ROWS)
    seen = []
    choose = E.choose_settings(MARKET)

    def spy_on(year, earlier):
        seen.append((year, max(r["prices"]["listing_date"] for r in earlier)))
        return choose(year, earlier)
    folds = E.walk_forward(dev, MARKET, settings=spy_on)
    assert all(last < f"{year}-01-01" for year, last in seen)                 # the chooser never saw its test year
    assert all(f["l2"] in E.L2_GRID and f["window"] in E.WINDOWS for f in folds)
    s, y, _ = E.pooled(folds)
    assert E.average_precision(s, y) > sum(y) / len(y) + 0.1                 # still finds the signal
    win = E.walk_forward(dev, MARKET, window=3)
    assert all(min(r["prices"]["listing_date"] for r in f["rows"]) >= f"{f['year']}-01-01" for f in win)


def test_the_spy_regime_on_a_listing_day_reads_only_earlier_closes():
    days = [(dt.date(2019, 1, 1) + dt.timedelta(days=k)).isoformat() for k in range(800)]
    rising = {d: 100 * 1.001 ** k for k, d in enumerate(days)}
    assert E.spy_regime(rising, days[400]) == "bull"
    crash = {d: (v if d < days[500] else v * 0.7) for d, v in rising.items()}
    assert E.spy_regime(crash, days[500]) == "bull"          # the crash starts that day: not yet known
    assert E.spy_regime(crash, days[520]) == "drawdown"
    assert E.spy_regime(rising, days[100]) is None            # under a year of history: no label


def test_break_even_friction_is_where_the_open_stops_paying():
    dev, _ = E.split(ROWS)
    folds = E.walk_forward(dev, MARKET)
    be = E.break_even_friction(folds)
    assert be > 0
    assert E.at_open(folds, be * 0.9)[("every IPO", "day 1")]["mean"] > 0 > E.at_open(folds, be * 1.1)[("every IPO", "day 1")]["mean"]
