"""Synthetic market and IPO data plus example bots, so the test suite runs out of the box.

Replace this with your real data and bots (see get_setup() in test_bot_benchmark.py).

The demo world
--------------
* Market: 30 tickers. A daily signal published at the close predicts next-day return weakly.
* IPOs: on pricing night you know how far the offer price was revised versus the filed
  range, plus a market "heat" reading. First-day return depends mostly on the revision.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

from bot_benchmark import Event, PITView, PointInTimeData


def build_demo(seed: int = 7, n_tickers: int = 30, n_days: int = 420, every: int = 4,
               n_ipos: int = 160, ipo_threshold: float = 0.10):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = [f"TCK{i:02d}" for i in range(n_tickers)]
    rows, events = [], []

    signal = rng.normal(size=(n_days, n_tickers))
    shock = rng.normal(size=n_days)[:, None]          # market-wide move shared by all tickers
    noise = rng.normal(size=(n_days, n_tickers))
    next_ret = 0.004 * signal + 0.006 * shock + 0.010 * noise
    for t, day in enumerate(days):
        for j, tic in enumerate(tickers):
            rows.append((day, tic, "signal", signal[t, j], "market"))
    for t in range(0, n_days - 1, every):
        for j, tic in enumerate(tickers):
            eid = f"{tic}@{days[t].date()}"
            events.append(Event(eid, "market", tic, days[t], days[t + 1], 0.0))
            rows.append((days[t + 1], eid, "outcome", next_ret[t, j], "market"))

    ipo_days = np.sort(rng.choice(np.arange(20, n_days - 1), size=n_ipos, replace=False))
    for k, t in enumerate(ipo_days):
        ipo = f"IPO{k:03d}"
        revision = rng.normal(0.0, 0.10)               # offer price vs midpoint of filed range
        heat = rng.normal()
        first_day = 0.12 + 0.9 * revision + 0.03 * heat + rng.normal(0.0, 0.15)
        rows.append((days[t], ipo, "range_revision", revision, "ipo"))
        rows.append((days[t], ipo, "heat", heat, "ipo"))
        eid = f"{ipo}@{days[t].date()}"
        events.append(Event(eid, "ipo", ipo, days[t], days[t + 1], ipo_threshold))
        rows.append((days[t + 1], eid, "outcome", first_day, "ipo"))

    store = PointInTimeData(pd.DataFrame(rows, columns=["available_at", "entity", "field", "value", "kind"]))
    return store, events


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _det_noise(key: str) -> float:
    """Deterministic pseudo-noise (Python's hash() changes between runs, so it is not used)."""
    return float(np.random.default_rng(zlib.crc32(key.encode())).normal())


class LearningBot:
    """An honest online learner.

    It logs the features it saw when it made each prediction, and once an outcome has been
    published it uses that pair to refit a small linear model. It never looks past ``view``.
    """

    def __init__(self, name: str = "candidate", signal_noise: float = 0.0, use_revision: bool = True,
                 min_history: int = 30):
        self.name = name
        self.signal_noise = signal_noise
        self.use_revision = use_revision
        self.min_history = min_history
        self._log = {}

    def _fit(self, kind: str, resolved: pd.DataFrame):
        known = resolved[resolved["kind"] == kind]
        pairs = [(self._log[e][1], y) for e, y in zip(known["event_id"], known["outcome"]) if e in self._log]
        if len(pairs) < self.min_history:
            return None
        x = np.vstack([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        sigma = float(np.std(y - x @ beta)) or 1e-6
        return beta, sigma

    def predict(self, view: PITView, events):
        signal = view.latest_many("signal")
        revision = view.latest_many("range_revision")
        heat = view.latest_many("heat")
        for e in events:
            if e.kind == "market":
                s = signal.get(e.entity, 0.0)
                if self.signal_noise:
                    s += self.signal_noise * _det_noise(e.event_id)
                self._log[e.event_id] = ("market", np.array([1.0, s]))
            else:
                r = revision.get(e.entity, 0.0) if self.use_revision else 0.0
                self._log[e.event_id] = ("ipo", np.array([1.0, r, heat.get(e.entity, 0.0)]))

        resolved = view.resolved()
        models = {k: self._fit(k, resolved) for k in ("market", "ipo")}
        out = []
        for e in events:
            kind, x = self._log[e.event_id]
            model = models[kind]
            if model is None:
                past = resolved.loc[resolved["kind"] == kind, "outcome"].to_numpy()
                p = float(np.mean(past > e.threshold)) if len(past) else 0.5
                mu = float(np.mean(past)) if len(past) else 0.0
            else:
                beta, sigma = model
                mu = float(x @ beta)
                p = _phi((mu - e.threshold) / sigma)
            out.append((e.event_id, min(max(p, 0.01), 0.99), mu))
        return pd.DataFrame(out, columns=["event_id", "prob_up", "expected_return"])


class CheaterBot:
    """Leaky on purpose: memorises every outcome, including future ones, when it is built."""

    name = "cheater"

    def __init__(self, store: PointInTimeData):
        self._answers = store.outcomes()

    def predict(self, view, events):
        p = [0.99 if self._answers[e.event_id] > e.threshold else 0.01 for e in events]
        return pd.DataFrame({"event_id": [e.event_id for e in events], "prob_up": p})


class LeakyScalerBot:
    """A subtler, very common leak: z-scoring a feature with full-history mean and spread."""

    name = "leaky_scaler"

    def __init__(self, store: PointInTimeData):
        s = store.frame
        s = s.loc[s["field"] == "signal", "value"]
        self._mean, self._std = float(s.mean()), float(s.std())

    def predict(self, view, events):
        signal = view.latest_many("signal")
        rows = []
        for e in events:
            z = (signal.get(e.entity, self._mean) - self._mean) / self._std
            rows.append((e.event_id, min(max(_phi(0.3 * z), 0.01), 0.99), 0.004 * z))
        return pd.DataFrame(rows, columns=["event_id", "prob_up", "expected_return"])
