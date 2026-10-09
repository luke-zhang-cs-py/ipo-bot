"""Live predictions: made once, never revised, scored when they resolve, and watched.

Every run (idempotent: a prediction already made is not made again):
  1. score: each prediction whose event has resolved gets its outcome (a withdrawn IPO resolves as void);
  2. predict: the next session's direction for every member with the last session's close (only between that
     close, plus the price delay, and the next open: a run during a session makes no stock predictions), and
     each IPO with an EFFECT notice that is not trading yet, by the adopted model and by each baseline, so live
     comparisons use the same rows;
  3. watch: per period (a week of stock targets, a quarter of IPOs), the model's Brier against the best
     baseline's, and its calibration. Three periods in a row below the best baseline, or calibration error
     above calibration_drift, is a warning in the health report.

The ledgers live in SQLite (append-only) and are mirrored to JSONL files that are committed to git, so the
record survives even if the database cache is lost.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from bot import collect, data, edgar, features, markets, models, stats
from bot.context import Ctx
from bot.evaluate import STOCK_TRAIN_DAYS
from bot.log import event
from bot.store import Store

BASELINES = {"stock": ("base_rate", "random_walk"), "ipo": ("base_rate", "recent_ipos")}
MIN_IPOS = 60  # resolved IPOs before a retrain is attempted
MIN_HOLDOUT_DAYS = 15  # out-of-sample sessions a stock retrain is judged on, at least
MIN_HOLDOUT_IPOS = 10


# ---------------------------------------------------------------------------- the model registry


def current_model(store: Store, kind: str) -> Optional[models.Model]:
    rows = store.query(
        "SELECT params FROM models WHERE kind = ? AND adopted = 1 ORDER BY trained_at DESC, model_id DESC " "LIMIT 1",
        (kind,),
    )
    return models.Model.from_params(json.loads(rows[0]["params"])) if rows else None


def register(ctx: Ctx, m: models.Model, metrics: Dict[str, Any], adopted: bool) -> None:
    ctx.store.record(
        "models",
        [
            {
                "model_id": m.model_id,
                "kind": m.kind,
                "trained_at": ctx.at,
                "params": m.params(),
                "metrics": metrics,
                "adopted": int(adopted),
                "run_id": ctx.run_id,
            }
        ],
    )


# ---------------------------------------------------------------------------- inputs as of now


def macro_rows(store: Store) -> pd.DataFrame:
    """Every version of every macro value (features align them by available_at)."""
    rows = store.query("SELECT series, date, value, available_at FROM macro ORDER BY rowid")
    return pd.DataFrame(rows, columns=["series", "date", "value", "available_at"])


def stock_inputs(store: Store, at: Optional[str] = None) -> pd.DataFrame:
    """The stock panel as known at `at`: a (date, symbol) row only where the symbol was an index member that day
    (point-in-time membership, so stocks that joined later are not in the earlier years and leavers are kept)."""
    syms = data.universe(store, at)
    closes = data.closes(store, at, [*syms, features.INDEX])
    panel = features.stock_panel(closes, macro_rows(store), syms)
    if panel.empty:
        return panel
    return features.members_only(panel, data.membership(store, list(closes.index), at))


def _recent(df: pd.DataFrame) -> pd.DataFrame:
    """The last STOCK_TRAIN_DAYS sessions of a stock panel: the window the backtest fits each model on."""
    dates = sorted(df["date"].unique())
    return df[df["date"] >= dates[-STOCK_TRAIN_DAYS]] if len(dates) > STOCK_TRAIN_DAYS else df


def ipo_inputs(store: Store, cfg_pop: float, at: Optional[str] = None) -> pd.DataFrame:
    return features.ipo_rows(edgar.deals_asof(store, at), macro_rows(store), cfg_pop)


# ---------------------------------------------------------------------------- predict


def predict(ctx: Ctx) -> Dict[str, int]:
    out = {"stock": predict_stocks(ctx), "ipo": predict_ipos(ctx)}
    mirror(ctx)
    return out


def _ensure_model(ctx: Ctx, kind: str, df: pd.DataFrame, feats: Sequence[str]) -> Optional[models.Model]:
    """The adopted model; on a fresh database, a first one fitted the way the backtest fits (stocks: the last
    STOCK_TRAIN_DAYS sessions known now; IPOs: every resolved IPO). A model shrunk all the way (s = 0, every
    prediction the base rate) is a warning each time it is used."""
    m = current_model(ctx.store, kind)
    if m is None:
        try:
            m = models.fit(kind, _recent(df) if kind == "stock" else df, feats, ctx.now.date().isoformat())
        except ValueError as e:
            ctx.warn("no_model", kind=kind, reason=str(e))
            return None
        note = "first model, fitted at the first run on " + (
            f"the last {STOCK_TRAIN_DAYS} sessions" if kind == "stock" else "all resolved IPOs"
        )
        register(ctx, m, {"note": note, "shrink": m.shrink}, adopted=True)
    if m.shrink == 0.0:  # every prediction is the training base rate
        ctx.warn("constant_model", kind=kind, model=m.model_id, shrink=m.shrink)
    return m


def predict_stocks(ctx: Ctx) -> int:
    """Predictions for the session after the last closed one, made only between that close (plus the price
    delay) and the next open. Outside that window (a run during a session) nothing is predicted: the next
    session would already be under way. That is noted in the log, not warned about."""
    day = markets.last_closed_session(ctx.now)
    nxt = markets.next_trading_day(day)
    if not markets.close_utc(day) + collect.PRICE_DELAY <= ctx.now < markets.open_utc(nxt):
        event(
            ctx.log,
            "stock_predictions_skipped",
            logging.INFO,
            reason="market open or close not yet final",
            day=str(day),
        )
        return 0
    panel = stock_inputs(ctx.store)
    if panel.empty:
        ctx.warn("no_stock_predictions", reason="no prices")
        return 0
    known = panel[panel["date"] < day.isoformat()]
    m = _ensure_model(ctx, "stock", known, features.STOCK_FEATURES)
    # today's members, less any known delisted (a monthly snapshot can lag a delisting by weeks)
    current = set(data.members(ctx.store))
    today = panel[(panel["date"] == day.isoformat()) & panel["symbol"].isin(current)].reset_index(drop=True)
    if m is None or today.empty:
        if m is not None:
            ctx.warn("no_stock_predictions", reason=f"no closes for {day}")
        return 0
    target = nxt.isoformat()
    hist = panel[panel["y"].notna()]
    rate = hist.groupby("symbol")["y"].agg(["mean", "count"])
    pooled = float(hist["y"].mean()) if len(hist) else 0.5
    p_base = [
        float(rate.loc[s, "mean"]) if s in rate.index and rate.loc[s, "count"] >= 20 else pooled
        for s in today["symbol"]
    ]
    forecasts = {
        m.model_id: (m.prob(today), m.value(today)),
        "base_rate": (np.array(p_base), np.zeros(len(today))),
        "random_walk": (np.full(len(today), 0.5), np.zeros(len(today))),
    }
    rows = []
    for name, (p, v) in forecasts.items():
        for i, sym in enumerate(today["symbol"]):
            rows.append(
                {
                    "pred_id": f"stock:{sym}:{day}:{name}",
                    "model": name,
                    "kind": "stock",
                    "subject": sym,
                    "event": "close_up",
                    "target": target,
                    "prob": float(p[i]),
                    "value": float(v[i]),
                    "made_at": ctx.at,
                    "data": {"date": day.isoformat()},
                    "run_id": ctx.run_id,
                }
            )
    _check_outputs(rows, set(today["symbol"]), len(forecasts))
    n = ctx.store.record("predictions", rows)
    ctx.count("stock_predictions", n)
    return n


def predict_ipos(ctx: Ctx) -> int:
    rows_df = ipo_inputs(ctx.store, ctx.cfg.ipo_pop)
    if rows_df.empty:
        return 0
    deals = {d.cik: d for d in edgar.deals_asof(ctx.store)}
    # only deals with an EFFECT notice and no first trade: a deal known only by its final prospectus (424B4) may
    # already be trading, since that is often filed on or after the listing day. ipo_rows drops those deals too,
    # so the backtest scores the same population; the check here is a guard
    pending = rows_df[
        [
            deals[c].effective is not None and deals[c].status(ctx.now.date()) in ("effective", "priced")
            for c in rows_df["cik"]
        ]
    ]
    if pending.empty:
        return 0
    trained = rows_df[rows_df["trade_date"].notna()]
    m = _ensure_model(ctx, "ipo", trained, features.IPO_FEATURES)
    if m is None:
        return 0
    known = trained.sort_values("trade_date")
    pooled = float(known["y"].mean()) if len(known) else models.PRIOR_POP
    recent = known.tail(20)
    forecasts = {
        m.model_id: (m.prob(pending), m.value(pending)),
        "base_rate": (
            np.full(len(pending), pooled),
            np.full(len(pending), float(known["ret"].mean()) if len(known) else models.PRIOR_FIRST_DAY),
        ),
        "recent_ipos": (
            np.full(len(pending), float(recent["y"].mean()) if len(recent) else models.PRIOR_POP),
            np.full(len(pending), float(recent["ret"].mean()) if len(recent) else models.PRIOR_FIRST_DAY),
        ),
    }
    rows = []
    for name, (p, v) in forecasts.items():
        for i, (cik, company) in enumerate(zip(pending["cik"], pending["company"])):
            rows.append(
                {
                    "pred_id": f"ipo:{cik}:{name}",
                    "model": name,
                    "kind": "ipo",
                    "subject": cik,
                    "event": "first_day_pop",
                    "target": "first_trade",
                    "prob": float(p[i]),
                    "value": float(v[i]),
                    "made_at": ctx.at,
                    "data": {"company": company, "moment": pending["moment"].iloc[i]},
                    "run_id": ctx.run_id,
                }
            )
    _check_outputs(rows, set(pending["cik"]), len(forecasts))
    n = ctx.store.record("predictions", rows)
    ctx.count("ipo_predictions", n)
    return n


class PredictionError(Exception):
    """A forecast that fails the output checks: never recorded."""


def _check_outputs(rows: List[Dict[str, Any]], subjects: Set[str], forecasters: int) -> None:
    """Every output finite, every probability in [0, 1], every requested subject answered by every forecaster."""
    for r in rows:
        if not (np.isfinite(r["prob"]) and np.isfinite(r["value"])):
            raise PredictionError(f"{r['pred_id']}: non-finite output")
        if not 0.0 <= r["prob"] <= 1.0:
            raise PredictionError(f"{r['pred_id']}: probability {r['prob']}")
    answered = {r["subject"] for r in rows}
    if answered != subjects or len(rows) != len(subjects) * forecasters:
        raise PredictionError(f"answered {len(rows)} of {len(subjects) * forecasters} requested forecasts")


# ---------------------------------------------------------------------------- score


def score(ctx: Ctx) -> int:
    """Outcomes for predictions that have resolved and have none yet."""
    open_ = ctx.store.query(
        "SELECT p.* FROM predictions p LEFT JOIN outcomes o ON o.pred_id = p.pred_id " "WHERE o.pred_id IS NULL"
    )
    if not open_:
        return 0
    stock = [r for r in open_ if r["kind"] == "stock"]
    rows = []
    if stock:
        syms = sorted({r["subject"] for r in stock})
        closes = data.closes(ctx.store, None, syms)
        # a target day still without a close two sessions later is void (a halt, a delisting), not pending forever
        last = markets.last_closed_session(ctx.now)
        void_before = markets.previous_trading_day(markets.previous_trading_day(last)).isoformat()
        for r in stock:
            day, target, sym = json.loads(r["data"])["date"], r["target"], r["subject"]
            c0 = closes.at[day, sym] if sym in closes.columns and day in closes.index else np.nan
            c1 = closes.at[target, sym] if sym in closes.columns and target in closes.index else np.nan
            if pd.notna(c0) and pd.notna(c1):
                rows.append(
                    {
                        "pred_id": r["pred_id"],
                        "outcome": float(c1 > c0),
                        "value": float(np.log(c1 / c0)),
                        "resolved_at": ctx.at,
                        "run_id": ctx.run_id,
                    }
                )
            elif target < void_before:
                rows.append(
                    {
                        "pred_id": r["pred_id"],
                        "outcome": None,
                        "value": None,
                        "resolved_at": ctx.at,
                        "run_id": ctx.run_id,
                    }
                )
    ipo = [r for r in open_ if r["kind"] == "ipo"]
    if ipo:
        deals = {d.cik: d for d in edgar.deals_asof(ctx.store)}
        for r in ipo:
            d = deals.get(r["subject"])
            if d is None:
                continue
            ret = d.first_day_return
            if ret is not None:
                rows.append(
                    {
                        "pred_id": r["pred_id"],
                        "outcome": float(features.popped(ret, ctx.cfg.ipo_pop)),
                        "value": ret,
                        "resolved_at": ctx.at,
                        "run_id": ctx.run_id,
                    }
                )
            elif d.status(ctx.now.date()) == "withdrawn":
                rows.append(
                    {
                        "pred_id": r["pred_id"],
                        "outcome": None,
                        "value": None,
                        "resolved_at": ctx.at,
                        "run_id": ctx.run_id,
                    }
                )
    n = ctx.store.record("outcomes", rows)
    ctx.count("scored", n)
    return n


# ---------------------------------------------------------------------------- watch


def resolved(store: Store, kind: str) -> pd.DataFrame:
    rows = store.query(
        "SELECT p.pred_id, p.model, p.subject, p.target, p.prob, p.value AS forecast, p.made_at, p.data, "
        "o.outcome, o.value AS actual FROM predictions p JOIN outcomes o ON o.pred_id = p.pred_id "
        "WHERE p.kind = ? AND o.outcome IS NOT NULL ORDER BY p.made_at",
        (kind,),
    )
    return pd.DataFrame(rows)


def period_of(kind: str, row: pd.Series) -> str:
    if kind == "stock":
        y, w, _ = dt.date.fromisoformat(row["target"]).isocalendar()
        return f"{y}-W{w:02d}"
    return markets.quarter_label(dt.date.fromisoformat(row["made_at"][:10]))


def watch(store: Store, kind: str, warn_periods: int, drift: float) -> Dict[str, Any]:
    """Rolling live metrics per period and the warnings they raise."""
    df = resolved(store, kind)
    if df.empty:
        return {"periods": [], "warnings": []}
    df["period"] = [period_of(kind, r) for _, r in df.iterrows()]
    df["family"] = [m if m in BASELINES[kind] else "model" for m in df["model"]]
    periods = []
    for p, g in df.groupby("period"):
        row: Dict[str, Any] = {"period": p}
        for fam, h in g.groupby("family"):
            row[fam] = {
                "n": len(h),
                "brier": float(stats.brier(h["prob"], h["outcome"]).mean()),
                "ece": float(stats.calibration(h["prob"].to_numpy(), h["outcome"].to_numpy())["ece"]),
            }
        periods.append(row)
    warnings = []
    # only periods where the model and a baseline were both scored count (else "worse in every one" holds of none)
    scored = [p for p in periods if "model" in p and any(b in p for b in BASELINES[kind])]
    recent = scored[-warn_periods:]
    if len(recent) == warn_periods and all(
        p["model"]["brier"] > min(p[b]["brier"] for b in BASELINES[kind] if b in p) for p in recent
    ):
        warnings.append(
            f"{kind}: the model's Brier was worse than the best baseline's in each of the last {warn_periods} periods"
        )
    last = [p for p in periods if "model" in p][-warn_periods:]
    if last:
        model_rows = df[(df["family"] == "model") & df["period"].isin([p["period"] for p in last])]
        ece = float(stats.calibration(model_rows["prob"].to_numpy(), model_rows["outcome"].to_numpy())["ece"])
        if len(model_rows) >= 50 and ece > drift:
            warnings.append(
                f"{kind}: calibration drift: expected calibration error {ece:.3f} over the last "
                f"{len(last)} periods (limit {drift:.2f})"
            )
    return {"periods": periods, "warnings": warnings}


def mirror(ctx: Ctx, folder: Optional[pathlib.Path] = None) -> None:
    """Append ledger rows not yet in the committed JSONL files (tracking/bot/)."""
    folder = folder or ctx.cfg.tracking_dir or ctx.cfg.data_dir / "tracking"
    folder.mkdir(parents=True, exist_ok=True)
    for table, key in (("predictions", "pred_id"), ("outcomes", "pred_id"), ("models", "model_id")):
        path = folder / f"{table}.jsonl"
        have = set()
        if path.exists():
            have = {json.loads(line)[key] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        new = [r for r in ctx.store.query(f"SELECT * FROM {table} ORDER BY rowid") if r[key] not in have]
        if new:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                for r in new:
                    f.write(json.dumps(r, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------- retrain


def retrain(ctx: Ctx, kind: str, holdout_days: int = 126) -> Dict[str, Any]:
    """Refit the model on everything known now and adopt it only if it beats the current model out of sample.

    The holdout is the outcomes after the current model's training data (never ones it was fitted on), at most
    the last holdout_days sessions (stocks) or sixth of the IPOs. A candidate is fitted on what came before the
    holdout and both are scored on it (Brier). With too few outcomes since the current model, nothing changes
    yet. The comparison is recorded either way."""
    feats: Sequence[str]
    current = current_model(ctx.store, kind)
    since = current.trained_through if current else ""
    if kind == "stock":
        df = stock_inputs(ctx.store)
        df = df[df["y"].notna()]
        dates = sorted(df["date"].unique())
        if len(dates) <= holdout_days + 60:
            return {"adopted": False, "reason": "not enough history"}
        cut = max(dates[-holdout_days], min((d for d in dates if d > since), default="9999"))
        feats, period, need = features.STOCK_FEATURES, "date", MIN_HOLDOUT_DAYS
        held = len([d for d in dates if d >= cut])
    else:
        df = ipo_inputs(ctx.store, ctx.cfg.ipo_pop)
        df = df[df["y"].notna()]
        if len(df) < MIN_IPOS:
            return {"adopted": False, "reason": "not enough IPOs"}
        trades = sorted(df["trade_date"])
        cut = max(trades[-max(20, len(df) // 6)], min((d for d in trades if d > since), default="9999"))
        feats, period, need = features.IPO_FEATURES, "trade_date", MIN_HOLDOUT_IPOS
        held = len([d for d in trades if d >= cut])
    if held < need:
        return {"adopted": False, "reason": f"{held} outcomes since the current model was fitted; waiting for {need}"}
    train, hold = df[df[period] < cut], df[df[period] >= cut]
    if kind == "stock":  # fitted on the backtest's rolling window, as the first model was
        train, full = _recent(train), _recent(df)
    else:
        full = df
    try:
        cand_hold = models.fit(kind, train, feats, str(cut))
        cand = models.fit(kind, full, feats, ctx.now.date().isoformat())
    except ValueError as e:
        return {"adopted": False, "reason": str(e)}
    y = hold["y"].to_numpy()
    b_cand = float(stats.brier(cand_hold.prob(hold), y).mean())
    b_cur = float(stats.brier(current.prob(hold), y).mean()) if current else float("inf")
    adopt = b_cand < b_cur
    keys = hold[period].astype(str).tolist()
    test = (
        stats.diebold_mariano(
            stats.by_period(keys, stats.brier(cand_hold.prob(hold), y).tolist()),
            stats.by_period(keys, stats.brier(current.prob(hold), y).tolist()),
        )
        if current
        else {}
    )
    metrics = {
        "holdout_from": str(cut),
        "holdout_n": len(hold),
        "brier_candidate": b_cand,
        "brier_current": b_cur if current else None,
        "current": current.model_id if current else None,
        "dm": test,
        "shrink": cand.shrink,
    }
    register(ctx, cand, metrics, adopted=adopt)
    if adopt and cand.shrink == 0.0:
        ctx.warn("constant_model", kind=kind, model=cand.model_id, shrink=cand.shrink)
    mirror(ctx)
    return {"adopted": adopt, **metrics, "model_id": cand.model_id}
