"""ipo-bot's own data and models, in the point-in-time harness (bot_benchmark.py).

ipo_setup()     the pre-listing IPO model from evaluation/ipo_eval.py (same features, same logistic regression)
                on the real IPO dataset. Each IPO's prospectus facts are stamped at pricing night (22:00 UTC the
                evening before it lists), it is predicted at 13:00 UTC on the listing day (before the 13:30 open),
                and its first-day return is published at 21:00 UTC (after the close). An event "pops" when its
                first day is ipo_eval.POP (20%) or more. Every listing from 2015 to 2023 is an event, so the walk-forward starts with
                nothing known, as it would have; 2024 on is the sealed holdout and is left out entirely.
market_setup()  the benchmark's stock algorithms (evaluation/benchmark.py) on its 30 US large caps: each
                month-end, next month's return. Closes are stamped at 21:00 UTC on the month-end, predictions at
                22:00 UTC, the outcome at the next month-end's close.

Every bot here learns only from the views it is handed: features it logged when it predicted, and outcomes
already published. Nothing is fitted on the full history first.
"""
from __future__ import annotations

import calendar
import math
import pathlib
import statistics
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL.parent / "src"))

import bot_benchmark as bb  # noqa: E402

MIN_IPO_HISTORY = 100           # IPOs with known outcomes before the model is fitted; until then, the base rate
MIN_MARKET_HISTORY = 300        # stock-months with known outcomes before an algorithm's map from score is fitted
PRICING_HOUR = -2               # IPO facts are public at 22:00 UTC the evening before the listing day
PREDICT_HOUR = 13               # IPO predictions at 13:00 UTC, before the 13:30 UTC (9:30 New York) open
CLOSE_HOUR = 21                 # first-day outcomes and month-end closes at 21:00 UTC, after the 20:00 UTC close
MONTH_PREDICT_HOUR = 22         # stock predictions an hour after the month-end close is stamped
FACTS = ("offer", "range_lo", "range_hi", "shares", "sic", "foreign", "bank", "spy_20d")   # an IPO's pre-listing facts


def fit_logit(X, y, l2=1.0, iters=25):
    """ipo_eval.Logit's fit in numpy: standardised inputs, an unpenalised intercept, L2 on the rest, Newton steps.
    Same solution to rounding error (checked in the harness README), a hundred times faster. Returns prob(X)."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], float(l2))
    pen[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -35, 35)))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen), Z.T @ (p - y) + pen * w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ w, -35, 35)))


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ----------------------------------------------------------------------------- IPOs

def ipo_threshold():
    """The IPO events' threshold: just under ipo_eval.POP, so the harness's outcome > threshold matches popped()."""
    import ipo_eval
    return ipo_eval.POP - ipo_eval.POP_TOL

def ipo_store():
    import ipo_data
    import ipo_eval
    rows = [r for r in ipo_eval.usable(ipo_data.load()) if r["prices"]["listing_date"] < ipo_eval.HOLDOUT_FROM]
    spy20 = ipo_eval.spy_returns(ipo_eval.spy_closes())
    banks = sorted({r["lead_bank"] for r in rows if r.get("lead_bank")})
    bank_id = {b: k for k, b in enumerate(banks)}
    data, events = [], []
    for r in rows:
        day = pd.Timestamp(r["prices"]["listing_date"])
        pricing, as_of, resolve = (day + pd.Timedelta(hours=PRICING_HOUR), day + pd.Timedelta(hours=PREDICT_HOUR),
                                   day + pd.Timedelta(hours=CLOSE_HOUR))
        ent = r["adsh"]
        rng = r.get("range") if r.get("range") and r.get("range_date") and r["range_date"] < r["prices"]["listing_date"] else None
        try:
            sic = float(int(r.get("sic")))
        except (TypeError, ValueError):
            sic = -1.0
        facts = {"offer": r["offer_price"], "range_lo": rng[0] if rng else -1.0, "range_hi": rng[1] if rng else -1.0,
                 "shares": r.get("shares") or -1.0, "sic": sic, "foreign": float(bool(r.get("foreign"))),
                 "bank": float(bank_id.get(r.get("lead_bank"), -1)), "spy_20d": spy20.get(r["prices"]["listing_date"], 0.0)}
        data += [(pricing, ent, k, v, "ipo") for k, v in facts.items()]
        eid = f"{ent}@{r['prices']['listing_date']}"
        data.append((resolve, eid, "outcome", ipo_eval.first_day(r), "ipo"))
        # The harness scores an event "up" when outcome > threshold; with the threshold a hair under POP that is
        # ipo_eval.popped (POP or more, a close of exactly 20% over the offer included), not "more than POP".
        events.append(bb.Event(eid, "ipo", ent, as_of, resolve, ipo_threshold()))
    store = bb.PointInTimeData(pd.DataFrame(data, columns=bb.STORE_COLUMNS))
    return store, events, banks


class IPOModelBot:
    """ipo_eval's logistic regression as an online, point-in-time bot. When asked about an IPO it logs that IPO's
    pre-listing facts and the market as it stood (SPY's last 20 days, and the first days of IPOs already
    published in the 30 days before); once a month it refits on the logged IPOs whose outcomes are now
    published, with a league table of banks built from those same IPOs. Until MIN_IPO_HISTORY are known it
    predicts their base rate. Learning only from its own log keeps it honest under the harness's masking."""

    def __init__(self, banks, name="ipo_model", features=None):
        import ipo_eval
        self.E, self.banks, self.name = ipo_eval, banks, name
        self.features = features or ipo_eval.FEATURES
        self.model, self.fitted_month, self.base = None, None, None
        self._log = {}                                  # event_id -> (row, market facts at the time)

    def _row(self, facts, ent, day):
        f = {k: facts[k].get(ent, -1.0) for k in facts}
        lo, hi = f["range_lo"], f["range_hi"]
        return {"adsh": ent, "offer_price": f["offer"], "shares": f["shares"] if f["shares"] > 0 else None,
                "range": [lo, hi] if lo > 0 and hi > 0 else None, "range_date": None,
                "sic": str(int(f["sic"])) if f["sic"] >= 0 else None, "foreign": f["foreign"] > 0.5,
                "lead_bank": self.banks[int(f["bank"])] if f["bank"] >= 0 else None,
                "prices": {"listing_date": day}}

    def _matrix(self, logged, banks):
        """The logged feature vectors, with bank_share from this refit's league table."""
        X = np.array([v for _, v in logged], float)
        if "bank_share" in self.features:
            X[:, self.features.index("bank_share")] = [banks.get(r.get("lead_bank"), 0.0) for r, _ in logged]
        return X

    def predict(self, view, events):
        # One pass over the view for just the IPOs asked about (training rows come from the bot's own log), instead
        # of a full-table filter per fact: same values as view.latest_many, about three times faster.
        sub = view.subset({e.entity for e in events}, FACTS).drop_duplicates(subset=["entity", "field"], keep="last")
        facts = {k: {} for k in FACTS}
        for ent, field, value in zip(sub["entity"], sub["field"], sub["value"]):
            facts[field][ent] = float(value)
        res = view.resolved("ipo")
        known = {e: y for e, y in zip(res["event_id"], res["outcome"]) if e in self._log}
        published = sorted((self._log[e][0]["prices"]["listing_date"], y) for e, y in known.items())
        for e in events:
            day = pd.Timestamp(e.as_of).date().isoformat()
            start = (pd.Timestamp(day) - pd.Timedelta(days=30)).date().isoformat()
            prior = [y for d, y in published if start <= d < day]
            mk = {"spy_20d": facts["spy_20d"].get(e.entity, 0.0), "heat_30d": statistics.fmean(prior) if prior else 0.0,
                  "deals_30d": math.log1p(len(prior))}
            row = self._row(facts, e.entity, day)
            base = self.E.features(row, {row["adsh"]: mk}, {})            # bank_share is set at each refit
            self._log[e.event_id] = (row, np.array([base[k] for k in self.features], float))
        month = view.as_of.strftime("%Y-%m")
        if month != self.fitted_month:
            train = [self._log[e] for e in known]
            ys = [int(self.E.popped(known[e])) for e in known]
            self.base = (sum(ys) / len(ys)) if ys else 0.5
            if len(train) >= MIN_IPO_HISTORY and 0 < sum(ys) < len(ys):
                banks = self.E.bank_shares([r for r, _ in train])
                self.model = (fit_logit(self._matrix(train, banks), np.array(ys, float)), banks)
            else:
                self.model = None
            self.fitted_month = month
        out = []
        for e in events:
            if self.model is None:
                p = self.base if self.base is not None else 0.5
            else:
                model, banks = self.model
                p = float(model(self._matrix([self._log[e.event_id]], banks))[0])
            out.append((e.event_id, min(max(p, 0.01), 0.99)))
        return pd.DataFrame(out, columns=["event_id", "prob_up"])


def ipo_setup():
    store, events, banks = ipo_store()
    return {"store": store, "events": events,
            "candidate": lambda s: IPOModelBot(banks, name="ipo_model"),
            "rivals": {"revision_only": lambda s: IPOModelBot(banks, name="revision_only",
                                                              features=["revision", "above_range", "below_range", "no_range"])},
            "baselines": {"base_rate": lambda s: bb.BaseRateBot(), "coin_flip": lambda s: bb.CoinFlipBot()}}


# ----------------------------------------------------------------------------- stocks

def _month_end(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return pd.Timestamp(y, m, calendar.monthrange(y, m)[1])


def market_store():
    import benchmark as bm
    symbols = bm.UNIVERSES["US large caps"]
    months, px = bm.panel(symbols)
    data, events = [], []
    for i, ym in enumerate(months):
        end = _month_end(ym)
        for s in symbols:
            data.append((end + pd.Timedelta(hours=CLOSE_HOUR), s, "close", px[s][i], "market"))
    for i in range(bm.LOOKBACK, len(months) - 1):
        as_of = _month_end(months[i]) + pd.Timedelta(hours=MONTH_PREDICT_HOUR)
        resolve = _month_end(months[i + 1]) + pd.Timedelta(hours=CLOSE_HOUR)
        for s in symbols:
            eid = f"{s}@{months[i]}"
            data.append((resolve, eid, "outcome", px[s][i + 1] / px[s][i] - 1, "market"))
            events.append(bb.Event(eid, "market", s, as_of, resolve, 0.0))
    return bb.PointInTimeData(pd.DataFrame(data, columns=bb.STORE_COLUMNS)), events


class AlgoBot:
    """One of benchmark.py's algorithms, scored point-in-time. Each month it computes the algorithm's score for
    every stock from the closes in the view, logs each stock's cross-sectional rank (0 to 1), and maps rank to
    an expected return and a probability of a rise with a line fitted on the ranks it logged earlier whose
    outcomes are now published (falling back to the base rate until MIN_MARKET_HISTORY are known)."""

    def __init__(self, algo, name):
        import benchmark as bm
        self.bm, self.algo, self.name = bm, algo, name
        self._log = {}

    def _scores(self, prices):
        bm = self.bm
        n = min(len(v) for v in prices.values())
        series = {s: v[-n:] for s, v in prices.items()}
        i = n - 1
        if self.algo == "blend":
            raw = {a: [bm.ALGORITHMS[a](series[s], i) for s in series] for a in bm.ALGORITHMS}
            return dict(zip(series, bm.blend(raw)))
        return {s: bm.ALGORITHMS[self.algo](series[s], i) for s in series}

    def predict(self, view, events):
        closes = view.rows("close")
        prices = {s: g["value"].tolist() for s, g in closes.groupby("entity", sort=False)}
        names = [e.entity for e in events]
        scores = self._scores({s: prices[s] for s in names})
        order = sorted(names, key=lambda s: (scores[s], s))
        rank = {s: k / max(1, len(order) - 1) for k, s in enumerate(order)}
        for e in events:
            self._log[e.event_id] = rank[e.entity]
        res = view.resolved("market")
        pairs = [(self._log[e], y) for e, y in zip(res["event_id"], res["outcome"]) if e in self._log]
        out = []
        if len(pairs) >= MIN_MARKET_HISTORY:
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            b, a = np.polyfit(x, y, 1)
            sigma = float(np.std(y - (a + b * x))) or 1e-6
            for e in events:
                mu = float(a + b * rank[e.entity])
                out.append((e.event_id, min(max(_phi(mu / sigma), 0.01), 0.99), mu))
        else:
            past = res["outcome"].to_numpy(dtype=float)
            p = float(np.mean(past > 0)) if len(past) else 0.5
            mu = float(np.mean(past)) if len(past) else 0.0
            out = [(e.event_id, min(max(p, 0.01), 0.99), mu) for e in events]
        return pd.DataFrame(out, columns=["event_id", "prob_up", "expected_return"])


def market_setup():
    import benchmark as bm
    store, events = market_store()
    rivals = {a: (lambda s, a=a: AlgoBot(a, a)) for a in bm.ALGORITHMS}
    return {"store": store, "events": events,
            "candidate": lambda s: AlgoBot("blend", "blend_mom_trend_lowvol"),
            "rivals": rivals,
            "baselines": {"base_rate": lambda s: bb.BaseRateBot(), "coin_flip": lambda s: bb.CoinFlipBot()}}


# ----------------------------------------------------------------------------- both together (the default)

REVISION = ["revision", "above_range", "below_range", "no_range"]


class CombinedBot:
    """An IPO bot and a stock bot answering as one, each for its own kind of event. knowledge_cutoff is true for
    these models: they are fitted from nothing inside the walk-forward, so they know no more than the day
    before the first row of data."""

    def __init__(self, ipo_bot, market_bot, name, cutoff):
        self.ipo, self.market, self.name, self.knowledge_cutoff = ipo_bot, market_bot, name, cutoff

    def predict(self, view, events):
        parts = []
        for kind, bot in (("ipo", self.ipo), ("market", self.market)):
            evs = [e for e in events if e.kind == kind]
            if evs:
                parts.append(bot.predict(view, evs))
        out = pd.concat(parts, ignore_index=True)
        if "expected_return" not in out.columns:
            out["expected_return"] = np.nan
        return out


def real_setup():
    """IPO and stock events in one store, so every test applies. Candidate: the revision-only IPO model (the
    better of ipo-bot's two on 2015-2023: lower Brier, calibrated) with the benchmark's blend for stocks.
    Rivals: the 14-feature IPO model paired with each single stock algorithm."""
    import benchmark as bm
    ipo_store_, ipo_events, banks = ipo_store()
    mkt_store, mkt_events = market_store()
    store = bb.PointInTimeData(pd.concat([ipo_store_.frame, mkt_store.frame], ignore_index=True))
    cutoff = store.frame["available_at"].min() - pd.Timedelta(days=1)
    events = ipo_events + mkt_events
    candidate = lambda s: CombinedBot(IPOModelBot(banks, name="ipo_revision", features=REVISION),
                                      AlgoBot("blend", "blend"), "revision_model+blend", cutoff)
    rivals = {f"full_model+{a}": (lambda s, a=a: CombinedBot(IPOModelBot(banks, name="ipo_full"), AlgoBot(a, a),
                                                              f"full_model+{a}", cutoff)) for a in bm.ALGORITHMS}
    return {"store": store, "events": events, "candidate": candidate, "rivals": rivals,
            "baselines": {"base_rate": lambda s: bb.BaseRateBot(), "coin_flip": lambda s: bb.CoinFlipBot()}}
