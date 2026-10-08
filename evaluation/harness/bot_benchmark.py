"""Point-in-time backtesting and head-to-head accuracy tests for market and IPO prediction bots.

How it works
------------
* Every row of data carries an ``available_at`` timestamp: the moment it became public
  (a filing's release time, not the quarter it describes). Use naive UTC timestamps.
* Bots never touch the raw store. At each prediction time they get a ``PITView`` that
  physically contains only rows available at or before that time.
* Every bot predicts the same events at the same times, so every comparison is paired.
* Losses are averaged per prediction date before significance testing, because predictions
  made on the same day share the same market shock and are not independent.

Bot contract
------------
A *bot factory* is ``factory(store) -> bot``. The bot has a ``name`` and
``predict(view, events) -> DataFrame`` with columns ``event_id``, ``prob_up`` (probability the
outcome ends above the event's threshold) and, optionally, ``expected_return``.
Honest bots learn only from the views they are given. The factory receives the full store on
purpose: the leakage tests use it to catch bots that peek at future data during setup
(for example, fitting a scaler or a model on the whole history before the backtest starts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

try:  # SciPy gives exact t-distribution p-values; fall back to a normal approximation without it.
    from scipy import stats as _sps
except ImportError:  # pragma: no cover
    _sps = None

STORE_COLUMNS = ["available_at", "entity", "field", "value", "kind"]


class LookAheadError(RuntimeError):
    """Raised when code asks a view for data stamped after its prediction time."""


@dataclass(frozen=True)
class Event:
    """One thing to predict.

    event_id   unique id; its outcome is stored as entity=event_id, field="outcome".
    kind       "market", "ipo" or any label you want to slice results by.
    entity     the ticker or IPO the event is about.
    as_of      information cutoff: predictions may use data available at or before this time.
    resolve_at when the outcome becomes known; must be later than as_of.
    threshold  the event resolves "up" when outcome > threshold
               (0 for direction, 0.10 for "pops more than 10% on day one", and so on).
    """

    event_id: str
    kind: str
    entity: str
    as_of: pd.Timestamp
    resolve_at: pd.Timestamp
    threshold: float = 0.0


def _ns(ts) -> int:
    return int(pd.Timestamp(ts).as_unit("ns").value)


class PITView:
    """Everything that was public at ``as_of``, and nothing after it."""

    def __init__(self, frame: pd.DataFrame, as_of: pd.Timestamp):
        self.as_of = pd.Timestamp(as_of)
        self._frame = frame

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def subset(self, entities: Iterable[str], fields: Iterable[str]) -> pd.DataFrame:
        """Rows for these entities and fields only, without copying the whole view (read-only)."""
        f = self._frame
        return f[f["entity"].isin(set(entities)) & f["field"].isin(set(fields))]

    def rows(self, field: str, kind: Optional[str] = None) -> pd.DataFrame:
        f = self._frame[self._frame["field"] == field]
        return f if kind is None else f[f["kind"] == kind]

    def latest_many(self, field: str) -> Dict[str, float]:
        """Most recent value of ``field`` for every entity that has one."""
        f = self.rows(field).drop_duplicates(subset="entity", keep="last")
        return dict(zip(f["entity"], f["value"].astype(float)))

    def value_at(self, entity: str, field: str, when) -> float:
        when = pd.Timestamp(when)
        if when > self.as_of:
            raise LookAheadError(f"asked for {entity}/{field} at {when}, but the view ends at {self.as_of}")
        f = self._frame
        f = f[(f["entity"] == entity) & (f["field"] == field) & (f["available_at"] <= when)]
        return float(f["value"].iloc[-1]) if len(f) else float("nan")

    def resolved(self, kind: Optional[str] = None) -> pd.DataFrame:
        """Outcomes that were already known at ``as_of`` (safe to train on)."""
        f = self.rows("outcome", kind)
        return f.rename(columns={"entity": "event_id", "value": "outcome"})[
            ["available_at", "event_id", "kind", "outcome"]]


class PointInTimeData:
    """A long-format store of timestamped observations: available_at, entity, field, value, kind."""

    def __init__(self, frame: pd.DataFrame):
        missing = set(STORE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"store is missing columns: {sorted(missing)}")
        f = frame[STORE_COLUMNS].copy()
        f["available_at"] = pd.to_datetime(f["available_at"])
        f["value"] = f["value"].astype(float)
        # Plain Python strings, not pandas' default Arrow strings: identical values, and filtering and iterating them
        # in every view is several times faster (pandas 3 made Arrow strings the default).
        f["entity"] = f["entity"].astype(str).astype(object)
        f["field"] = f["field"].astype(str).astype(object)
        f["kind"] = f["kind"].astype(str).astype(object)
        self._frame = f.sort_values("available_at", kind="mergesort").reset_index(drop=True)
        self._times = self._frame["available_at"].astype("datetime64[ns]").to_numpy().view("i8")

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def view(self, as_of) -> PITView:
        idx = int(np.searchsorted(self._times, _ns(as_of), side="right"))
        return PITView(self._frame.iloc[:idx], as_of)

    def outcomes(self) -> pd.Series:
        o = self._frame[self._frame["field"] == "outcome"]
        return o.set_index("entity")["value"]

    def perturb_after(self, t, seed: int = 0) -> "PointInTimeData":
        """Copy of the store with every row after ``t`` shuffled and jittered.

        A bot that only uses information available at ``t`` cannot tell the difference.
        """
        rng = np.random.default_rng(seed)
        f = self._frame.copy()
        future = f["available_at"] > pd.Timestamp(t)
        for _, idx in f[future].groupby("field").groups.items():
            vals = f.loc[idx, "value"].to_numpy(dtype=float)
            center, scale = float(np.nanmean(vals)), float(np.nanstd(vals)) or 1.0
            # Shuffle, mirror around the mean and jitter: big changes that flip most up/down labels.
            f.loc[idx, "value"] = 2 * center - rng.permutation(vals) + rng.normal(0.0, 0.5 * scale, len(vals))
        return PointInTimeData(f)


class Bot(Protocol):
    name: str

    def predict(self, view: PITView, events: Sequence[Event]) -> pd.DataFrame: ...


BotFactory = Callable[[PointInTimeData], Bot]


# ---------------------------------------------------------------------------
# Baselines every serious bot must beat
# ---------------------------------------------------------------------------
class CoinFlipBot:
    """No skill at all: 50/50 on direction, zero expected return (a random walk)."""

    name = "coin_flip"

    def predict(self, view: PITView, events: Sequence[Event]) -> pd.DataFrame:
        return pd.DataFrame({"event_id": [e.event_id for e in events],
                             "prob_up": 0.5, "expected_return": 0.0})


class BaseRateBot:
    """Predicts the historical frequency and mean outcome of past events of the same kind.

    For IPOs this is "the average recent IPO", which is harder to beat than it sounds.
    """

    name = "base_rate"

    def __init__(self, min_history: int = 10):
        self.min_history = min_history

    def predict(self, view: PITView, events: Sequence[Event]) -> pd.DataFrame:
        res = view.resolved()
        rows = []
        for e in events:
            past = res.loc[res["kind"] == e.kind, "outcome"].to_numpy(dtype=float)
            if len(past) < self.min_history:
                p, mu = 0.5, 0.0
            else:
                p, mu = float(np.mean(past > e.threshold)), float(np.mean(past))
            rows.append((e.event_id, min(max(p, 0.01), 0.99), mu))
        return pd.DataFrame(rows, columns=["event_id", "prob_up", "expected_return"])


# ---------------------------------------------------------------------------
# Running bots
# ---------------------------------------------------------------------------
def _validate(out, events: Sequence[Event], name: str) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or not {"event_id", "prob_up"} <= set(out.columns):
        raise ValueError(f"{name} must return a DataFrame with event_id and prob_up columns")
    expected = {e.event_id for e in events}
    got = list(out["event_id"])
    if len(got) != len(set(got)):
        raise ValueError(f"{name} returned duplicate event_ids")
    missing, extra = expected - set(got), set(got) - expected
    if missing:
        raise ValueError(f"{name} skipped {len(missing)} events, e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"{name} predicted events it was not asked about, e.g. {sorted(extra)[:3]}")
    p = out["prob_up"].to_numpy(dtype=float)
    if not np.all(np.isfinite(p)) or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name} returned probabilities outside [0, 1] or non-finite values")
    out = out.copy()
    if "expected_return" not in out.columns:
        out["expected_return"] = np.nan
    return out[["event_id", "prob_up", "expected_return"]]


def group_by_time(events: Iterable[Event]) -> Dict[pd.Timestamp, List[Event]]:
    groups: Dict[pd.Timestamp, List[Event]] = {}
    for e in events:
        if not pd.Timestamp(e.resolve_at) > pd.Timestamp(e.as_of):
            raise ValueError(f"event {e.event_id} resolves before or at its prediction time")
        groups.setdefault(pd.Timestamp(e.as_of), []).append(e)
    return dict(sorted(groups.items()))


def run_walk_forward(factory: BotFactory, store: PointInTimeData, events: Sequence[Event],
                     until=None) -> pd.DataFrame:
    """Ask one bot for every event, in time order, giving it only the data public at each as_of."""
    bot = factory(store)
    name = getattr(bot, "name", type(bot).__name__)
    frames = []
    for t, evs in group_by_time(events).items():
        if until is not None and t > pd.Timestamp(until):
            break
        out = _validate(bot.predict(store.view(t), evs), evs, name)
        frames.append(out.assign(bot=name, as_of=t))
    return pd.concat(frames, ignore_index=True)


def pick_cut_dates(events: Sequence[Event], n: int = 3, lo: float = 0.3, hi: float = 0.9) -> List[pd.Timestamp]:
    """Choose cut dates for the leak test where the most predictions are still unresolved.

    Those are the moments a leaky bot has the most to gain from peeking, so leaks show up there.
    """
    times = sorted({pd.Timestamp(e.as_of) for e in events})
    as_of = np.array([_ns(e.as_of) for e in events])
    resolve = np.array([_ns(e.resolve_at) for e in events])
    window = times[int((len(times) - 1) * lo): int((len(times) - 1) * hi) + 1]
    picks = []
    for chunk in np.array_split(np.array(window, dtype=object), n):
        if len(chunk):
            pending = [int(np.sum((as_of <= _ns(t)) & (resolve > _ns(t)))) for t in chunk]
            picks.append(pd.Timestamp(chunk[int(np.argmax(pending))]))
    return picks


def future_invariance(factory: BotFactory, store: PointInTimeData, events: Sequence[Event],
                      cut_dates: Sequence, tol: float = 1e-9, seed: int = 0) -> pd.DataFrame:
    """Leak detector. For each cut date, scramble everything after it and re-run the bot up to it.

    Returns the predictions that changed. An honest bot returns an empty frame.
    """
    changed = []
    for i, t in enumerate(cut_dates):
        base = run_walk_forward(factory, store, events, until=t)
        pert = run_walk_forward(factory, store.perturb_after(t, seed=seed + i), events, until=t)
        m = base.merge(pert, on="event_id", suffixes=("", "_perturbed"))
        dp = (m["prob_up"] - m["prob_up_perturbed"]).abs()
        dr = (m["expected_return"] - m["expected_return_perturbed"]).abs().fillna(0.0)
        bad = m[(dp > tol) | (dr > tol)]
        if len(bad):
            changed.append(bad.assign(cut_date=pd.Timestamp(t)))
    cols = ["cut_date", "event_id", "prob_up", "prob_up_perturbed"]
    return pd.concat(changed, ignore_index=True)[cols] if changed else pd.DataFrame(columns=cols)


def mask_identities(store: PointInTimeData, events: Sequence[Event], seed: int = 0):
    """Replace tickers, IPO names and event ids with random codes.

    A numbers-only bot should score the same. If an LLM-based bot gets much worse, it was
    probably recalling what happened to those companies from its training data.
    Returns (masked_store, masked_events, map_back_to_original_event_ids).
    """
    rng = np.random.default_rng(seed)
    entities = sorted({e.entity for e in events})
    ids = [e.event_id for e in events]
    ent_map = {ent: f"ENT{i:05d}" for i, ent in enumerate(rng.permutation(entities))}
    ev_map = {eid: f"EVT{i:06d}" for i, eid in enumerate(rng.permutation(ids))}
    f = store.frame
    f["entity"] = [ev_map.get(x, ent_map.get(x, x)) for x in f["entity"]]
    masked = [replace(e, event_id=ev_map[e.event_id], entity=ent_map[e.entity]) for e in events]
    return PointInTimeData(f), masked, {v: k for k, v in ev_map.items()}


def attach_outcomes(preds: pd.DataFrame, store: PointInTimeData, events: Sequence[Event]) -> pd.DataFrame:
    meta = pd.DataFrame([{"event_id": e.event_id, "kind": e.kind, "entity": e.entity,
                          "threshold": e.threshold} for e in events])
    outcomes = store.outcomes().rename("outcome")
    j = preds.merge(meta, on="event_id", how="left").merge(
        outcomes, left_on="event_id", right_index=True, how="left")
    if j["outcome"].isna().any():
        missing = j.loc[j["outcome"].isna(), "event_id"].unique()[:3]
        raise ValueError(f"no outcome stored for some events, e.g. {list(missing)}")
    j["y"] = (j["outcome"] > j["threshold"]).astype(int)
    j["year"] = pd.to_datetime(j["as_of"]).dt.year
    return j


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency, weighted by bin size."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    gap = sum(np.sum(idx == b) * abs(p[idx == b].mean() - y[idx == b].mean())
              for b in range(bins) if np.any(idx == b))
    return float(gap / len(p))


def auc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hit_rate(p, y) -> float:
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) == 1)))


def oos_r2(pred, actual, benchmark=None) -> float:
    """Out-of-sample R squared against a benchmark forecast (zero return by default).

    Positive means the bot's return forecasts have smaller squared errors than the benchmark's.
    """
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    bench = np.zeros_like(actual) if benchmark is None else np.asarray(benchmark, float)
    denom = np.sum((actual - bench) ** 2)
    return float(1 - np.sum((actual - pred) ** 2) / denom) if denom > 0 else float("nan")


def information_coefficient(joined: pd.DataFrame, min_names: int = 5):
    """Mean cross-sectional Spearman correlation between expected and realised returns.

    Returns (mean IC, t-statistic, number of dates). Measures ranking skill, which is what a
    long-short strategy actually monetises.
    """
    ics = []
    for _, g in joined.dropna(subset=["expected_return"]).groupby("as_of"):
        if len(g) >= min_names and g["expected_return"].nunique() > 1:
            ics.append(g["expected_return"].rank().corr(g["outcome"].rank()))
    ics = np.asarray([x for x in ics if np.isfinite(x)])
    if len(ics) < 2:
        return float("nan"), float("nan"), len(ics)
    return float(ics.mean()), float(ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))), len(ics)


def score_table(joined: pd.DataFrame, by: Sequence[str] = ("bot",)) -> pd.DataFrame:
    rows = []
    for key, g in joined.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        r = g.dropna(subset=["expected_return"])
        ic, ic_t, _ = information_coefficient(g)
        rows.append(dict(zip(by, key), n=len(g), brier=brier(g["prob_up"], g["y"]),
                         log_loss=log_loss(g["prob_up"], g["y"]),
                         ece=expected_calibration_error(g["prob_up"], g["y"]),
                         auc=auc(g["prob_up"], g["y"]), hit_rate=hit_rate(g["prob_up"], g["y"]),
                         mae=float(np.mean(np.abs(r["expected_return"] - r["outcome"]))) if len(r) else np.nan,
                         rmse=float(np.sqrt(np.mean((r["expected_return"] - r["outcome"]) ** 2))) if len(r) else np.nan,
                         oos_r2=oos_r2(r["expected_return"], r["outcome"]) if len(r) else np.nan,
                         ic=ic, ic_t=ic_t))
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Head-to-head statistics
# ---------------------------------------------------------------------------
LOSSES = {
    "brier": lambda d: (d["prob_up"] - d["y"]) ** 2,
    "log": lambda d: -(d["y"] * np.log(np.clip(d["prob_up"], 1e-6, 1 - 1e-6))
                       + (1 - d["y"]) * np.log(np.clip(1 - d["prob_up"], 1e-6, 1 - 1e-6))),
    "abs_return": lambda d: (d["expected_return"] - d["outcome"]).abs(),
}


def per_date_loss(joined: pd.DataFrame, bot: str, loss: str = "brier") -> pd.Series:
    d = joined[joined["bot"] == bot]
    return LOSSES[loss](d).groupby(d["as_of"]).mean().sort_index()


def _t_cdf(x: float, df: int) -> float:
    if _sps is not None:
        return float(_sps.t.cdf(x, df))
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lag: int = 0) -> dict:
    """Diebold-Mariano test (with the Harvey-Leybourne-Newbold correction) on per-date losses.

    One-sided: a small p-value means bot A has lower expected loss than bot B.
    Set ``lag`` to (forecast horizon in prediction steps - 1) when horizons overlap,
    for example weekly predictions of 90-day IPO returns.
    """
    a, b = loss_a.align(loss_b, join="inner")
    d = (a - b).to_numpy(dtype=float)
    n = len(d)
    if n < 10:
        raise ValueError(f"only {n} shared prediction dates; need at least 10")
    dbar = float(d.mean())
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / n)
    lrv = gamma0 + sum(2 * (1 - k / (lag + 1)) * float(np.dot(dc[k:], dc[:-k]) / n)
                       for k in range(1, lag + 1))
    if lrv <= 0:
        lrv = gamma0
    if lrv == 0:
        return {"n_dates": n, "mean_diff": dbar, "dm_stat": float("nan"), "p_value": float("nan")}
    h = lag + 1
    stat = dbar / math.sqrt(lrv / n) * math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return {"n_dates": n, "mean_diff": dbar, "dm_stat": float(stat), "p_value": _t_cdf(stat, n - 1)}


def block_bootstrap_diff(loss_a: pd.Series, loss_b: pd.Series, block: Optional[int] = None,
                         n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Moving-block bootstrap confidence interval for mean(loss_a - loss_b) over dates."""
    a, b = loss_a.align(loss_b, join="inner")
    d = (a - b).to_numpy(dtype=float)
    n = len(d)
    block = block or max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjustment, so beating ten rivals is not ten lucky coin flips."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def head_to_head(joined: pd.DataFrame, candidate: str, opponents: Sequence[str], loss: str = "brier",
                 lag: int = 0, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Paired comparison of the candidate against each opponent on the same events and dates.

    mean_diff < 0 means the candidate's loss is lower. ``wins`` requires a Holm-adjusted DM
    p-value below alpha *and* a bootstrap confidence interval entirely below zero.
    """
    la = per_date_loss(joined, candidate, loss)
    rows = []
    for opp in opponents:
        lb = per_date_loss(joined, opp, loss)
        rows.append({"opponent": opp, **diebold_mariano(la, lb, lag),
                     **{k: v for k, v in block_bootstrap_diff(la, lb, seed=seed).items() if k != "mean_diff"}})
    out = pd.DataFrame(rows)
    adj = holm(dict(zip(out["opponent"], out["p_value"].fillna(1.0))))
    out["p_holm"] = out["opponent"].map(adj)
    out["wins"] = (out["p_holm"] < alpha) & (out["ci_high"] < 0)
    return out


def slice_comparison(joined: pd.DataFrame, candidate: str, opponent: str, by: str) -> pd.DataFrame:
    """Brier score of candidate vs opponent within each slice (year, kind, sector, regime...)."""
    rows = []
    for key, g in joined.groupby(by):
        c, o = g[g["bot"] == candidate], g[g["bot"] == opponent]
        if len(c) and len(o):
            rows.append({by: key, "n": len(c), "candidate": brier(c["prob_up"], c["y"]),
                         "opponent": brier(o["prob_up"], o["y"])})
    out = pd.DataFrame(rows)
    out["diff"] = out["candidate"] - out["opponent"]
    return out


def long_short_returns(joined: pd.DataFrame, bot: str, kind: str = "market", quantile: float = 0.2,
                       cost_bps: float = 10.0) -> pd.Series:
    """Daily return of going long the top quantile and short the bottom quantile by expected return.

    Subtracts ``cost_bps`` for each leg every period (a simple full-turnover assumption).
    Accuracy is not profit; this checks whether the bot's edge survives trading costs.
    """
    d = joined[(joined["bot"] == bot) & (joined["kind"] == kind)].dropna(subset=["expected_return"])
    out = {}
    for t, g in d.groupby("as_of"):
        if len(g) < 5:
            continue
        k = max(1, int(len(g) * quantile))
        g = g.sort_values(["expected_return", "event_id"])
        out[t] = g["outcome"].tail(k).mean() - g["outcome"].head(k).mean() - 2 * cost_bps / 1e4
    return pd.Series(out, dtype=float).sort_index()


def format_report(joined: pd.DataFrame, candidate: str, opponents: Sequence[str]) -> str:
    pd.set_option("display.width", 160)
    parts = ["Scores (lower is better for brier, log_loss, ece, mae, rmse):",
             score_table(joined).round(4).to_string(index=False), "",
             "By kind:", score_table(joined, by=("kind", "bot")).round(4).to_string(index=False), "",
             f"Head to head, {candidate} vs others (Brier, per-date):",
             head_to_head(joined, candidate, opponents).round(4).to_string(index=False)]
    return "\n".join(parts)
