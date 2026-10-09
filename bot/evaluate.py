"""Walk-forward backtests, with baselines and significance tests, and the scrambled-future leakage test.

Stocks: every 21 sessions the model is refitted on the previous three years of (stock, day) rows whose
outcome was known at the decision time, then predicts the next 21 sessions. IPOs: refitted each quarter on
the IPOs that had traded before it. Each forecast is scored against the baselines on the same rows:

  stocks  base rate (each stock's up-day rate so far) and the random walk (p = 0.5, return 0)
  IPOs    base rate (the pop rate so far) and the average recent IPO (the last 20 IPOs' pop rate and return)

The tests compare losses on the model's rows only: Diebold-Mariano and a block bootstrap on the daily mean
loss (stocks) or the per-IPO loss in time order (IPOs), Holm-corrected within each domain.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from bot import features, ipos, markets, models, stats

STOCK_TRAIN_DAYS = 756
STOCK_REFIT_EVERY = 21
STOCK_WARMUP = 252  # sessions of history before the first walk-forward prediction
IPO_MIN_TRAIN = 40


def _next_session(date: str) -> str:
    return markets.next_trading_day(dt.date.fromisoformat(date)).isoformat()


def stock_walkforward(
    panel: pd.DataFrame,
    train_days: int = STOCK_TRAIN_DAYS,
    refit_every: int = STOCK_REFIT_EVERY,
    warmup: int = STOCK_WARMUP,
    l2: float = 1.0,
) -> pd.DataFrame:
    """Out-of-sample predictions: one row per (date, symbol) with p_model, v_model, p_base, v_base, p_rw, v_rw,
    y and ret."""
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = sorted(panel["date"].unique())
    out = []
    for b in range(warmup, len(dates), refit_every):
        train = panel[(panel["date"] >= dates[max(0, b - train_days)]) & (panel["date"] < dates[b])]
        test = panel[(panel["date"] >= dates[b]) & (panel["date"] <= dates[min(b + refit_every, len(dates)) - 1])]
        try:
            m = models.fit("stock", train, features.STOCK_FEATURES, dates[b - 1], l2=l2)
        except ValueError:
            continue
        t = test[["date", "symbol", "y", "ret"]].copy()
        t["p_model"], t["v_model"] = m.prob(test), m.value(test)
        out.append(t)
    if not out:
        return pd.DataFrame(
            columns=["date", "symbol", "y", "ret", "p_model", "v_model", "p_base", "v_base", "p_rw", "v_rw"]
        )
    res = pd.concat(out, ignore_index=True)
    # the base rate is computed over the whole panel (so its history starts at the panel's start), then joined
    known = [_next_session(d) if not np.isnan(y) else None for d, y in zip(panel["date"], panel["y"])]
    panel = panel.assign(
        p_base=models.base_rate(panel["y"], panel["symbol"], panel["date"], known, prior=models.PRIOR_UP)
    )
    res = res.merge(panel[["date", "symbol", "p_base"]], on=["date", "symbol"], how="left")
    res["v_base"] = 0.0
    res["p_rw"], res["v_rw"] = 0.5, 0.0
    return res


def ipo_walkforward(rows: pd.DataFrame, min_train: int = IPO_MIN_TRAIN, l2: float = 1.0) -> pd.DataFrame:
    """Out-of-sample IPO predictions: per deal p_model, v_model, p_base, v_base, p_recent, v_recent, y, ret."""
    if rows.empty:
        return pd.DataFrame(
            columns=["cik", "moment", "y", "ret", "p_model", "v_model", "p_base", "v_base", "p_recent", "v_recent"]
        )
    rows = rows.sort_values(["moment", "cik"]).reset_index(drop=True)
    quarters = sorted({markets.quarter_start(dt.date.fromisoformat(m)).isoformat() for m in rows["moment"]})
    out = []
    for q in quarters:
        # traded before the quarter, and predicted before it too: a deal in the quarter's test rows is never in
        # its training rows (every test row's moment is on or after q, so every training outcome came before it)
        train = rows[rows["trade_date"].notna() & (rows["trade_date"] < q) & (rows["moment"] < q)]
        nxt = markets.next_quarter(dt.date.fromisoformat(q)).isoformat()
        test = rows[(rows["moment"] >= q) & (rows["moment"] < nxt)]
        if len(train) < min_train or test.empty:
            continue
        try:
            m = models.fit("ipo", train, features.IPO_FEATURES, q, l2=l2)
        except ValueError:
            continue
        t = test[["cik", "moment", "y", "ret"]].copy()
        t["p_model"], t["v_model"] = m.prob(test), m.value(test)
        out.append(t)
    known = rows["trade_date"].tolist()
    base = rows.assign(
        p_base=models.base_rate(rows["y"], ["all"] * len(rows), rows["moment"], known, prior=models.PRIOR_POP),
        v_base=models.recent_mean(rows["ret"], rows["moment"], known, prior=models.PRIOR_FIRST_DAY, window=10_000),
        p_recent=models.recent_mean(rows["y"], rows["moment"], known, prior=models.PRIOR_POP),
        v_recent=models.recent_mean(rows["ret"], rows["moment"], known, prior=models.PRIOR_FIRST_DAY),
    )
    if not out:
        return pd.DataFrame(
            columns=["cik", "moment", "y", "ret", "p_model", "v_model", "p_base", "v_base", "p_recent", "v_recent"]
        )
    res = pd.concat(out, ignore_index=True)
    return res.merge(
        base[["cik", "moment", "p_base", "v_base", "p_recent", "v_recent"]], on=["cik", "moment"], how="left"
    )


def report(preds: pd.DataFrame, period: str, forecasters: Sequence[str], value_target: str = "ret") -> Dict[str, Any]:
    """Scores for each forecaster on the resolved rows, and the model against each baseline: DM and bootstrap
    on Brier, log loss and absolute error, Holm-adjusted."""
    r = preds[preds["y"].notna() & preds[value_target].notna()].sort_values(period)
    if r.empty:
        return {"n": 0, "scores": {}, "tests": {}}
    out: Dict[str, Any] = {"n": len(r), "first": str(r[period].iloc[0]), "last": str(r[period].iloc[-1]), "scores": {}}
    for f in forecasters:
        out["scores"][f] = stats.scores(
            r[f"p_{f}"].to_numpy(), r["y"].to_numpy(), r[f"v_{f}"].to_numpy(), r[value_target].to_numpy()
        )
    out["calibration"] = stats.calibration(r["p_model"].to_numpy(), r["y"].to_numpy())
    keys = r[period].astype(str).tolist()
    y, ret = r["y"].to_numpy(), r[value_target].to_numpy()
    losses = {
        f: {
            "brier": stats.brier(r[f"p_{f}"].to_numpy(), y),
            "log_loss": stats.log_loss(r[f"p_{f}"].to_numpy(), y),
            "abs_error": np.abs(r[f"v_{f}"].to_numpy() - ret),
        }
        for f in forecasters
    }
    tests: Dict[str, Dict[str, Any]] = {}
    pvals: Dict[str, float] = {}
    for f in forecasters:
        if f == "model":
            continue
        for metric in ("brier", "log_loss", "abs_error"):
            a = stats.by_period(keys, losses["model"][metric].tolist())
            b = stats.by_period(keys, losses[f][metric].tolist())
            dm = stats.diebold_mariano(a, b)
            boot = stats.block_bootstrap(a, b)
            name = f"model vs {f}: {metric}"
            tests[name] = {"dm": dm, "bootstrap": boot, "better": dm["mean_diff"] < 0}
            pvals[name] = dm["p"]
    for name, adj in stats.holm(pvals).items():
        tests[name]["dm_p_holm"] = adj
    out["tests"] = tests
    best = min((f for f in forecasters if f != "model"), key=lambda f: out["scores"][f]["brier"])
    out["best_baseline"] = best
    out["beats_best_baseline"] = bool(
        out["scores"]["model"]["brier"] < out["scores"][best]["brier"]
        and tests[f"model vs {best}: brier"]["dm_p_holm"] < 0.05
    )
    return out


# ---------------------------------------------------------------------------- the leakage test


def _perturb_macro(macro_rows: pd.DataFrame, after: dt.datetime, rng: np.random.Generator) -> pd.DataFrame:
    """Scramble every macro value published after `after` (by available_at, not by its date: a value dated the
    cutoff day can be published after the decision)."""
    m = macro_rows.copy()
    if not m.empty:
        fut = (pd.to_datetime(m["available_at"], utc=True) > after).to_numpy()
        m.loc[fut, "value"] = rng.permutation(m.loc[fut, "value"].to_numpy()) * rng.uniform(0.5, 2.0, int(fut.sum()))
    return m


def _scramble_after(
    closes: pd.DataFrame, macro_rows: pd.DataFrame, cutoff: str, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c = closes.copy()
    future = c.index > cutoff
    vals = c.loc[future].to_numpy()
    c.loc[future] = rng.permutation(vals.reshape(-1)).reshape(vals.shape) * rng.uniform(0.5, 2.0, size=vals.shape)
    return c, _perturb_macro(macro_rows, features.decision_time(dt.date.fromisoformat(cutoff)), rng)


def _same(a: pd.DataFrame, b: pd.DataFrame, cols: Sequence[str]) -> float:
    """The largest difference between two frames' columns (NaN in the same place counts as equal)."""
    x, y = a[list(cols)].to_numpy(dtype=float), b[list(cols)].to_numpy(dtype=float)
    if x.shape != y.shape:
        return float("inf")
    both = np.isnan(x) & np.isnan(y)
    if (np.isnan(x) != np.isnan(y)).any():
        return float("inf")
    d = np.abs(np.where(both, 0.0, x - y))
    return float(d.max()) if d.size else 0.0


def leakage_stocks(
    closes: pd.DataFrame, macro_rows: pd.DataFrame, cutoffs: Sequence[str], seed: int = 7
) -> Dict[str, Any]:
    """Replace every close after each cutoff, and every macro value published after its decision time (22:00
    UTC), with scrambled values; the features up to the cutoff, and a model fitted on what was known at the
    cutoff, must not change at all."""
    rng = np.random.default_rng(seed)
    base = features.stock_panel(closes, macro_rows)
    worst, checks = 0.0, []
    for c in cutoffs:
        sc, sm = _scramble_after(closes, macro_rows, c, rng)
        alt = features.stock_panel(sc, sm)
        a, b = base[base["date"] <= c].reset_index(drop=True), alt[alt["date"] <= c].reset_index(drop=True)
        diff = _same(a, b, features.STOCK_FEATURES)
        # outcomes known at the cutoff's decision time: targets of sessions before it
        tr_a, tr_b = a[a["date"] < c], b[b["date"] < c]
        pred_diff = 0.0
        try:
            ma = models.fit("stock", tr_a, features.STOCK_FEATURES, c)
            mb = models.fit("stock", tr_b, features.STOCK_FEATURES, c)
            today_a, today_b = a[a["date"] == c], b[b["date"] == c]
            pred_diff = float(np.max(np.abs(ma.prob(today_a) - mb.prob(today_b)), initial=0.0))
        except ValueError:
            pass
        checks.append({"cutoff": c, "feature_diff": diff, "prediction_diff": pred_diff})
        worst = max(worst, diff, pred_diff)
    return {"passed": worst == 0.0, "worst_diff": worst, "checks": checks}


def _other_bank(lead: Any) -> str:
    return "Small Bank LLC" if lead in ipos.TOP_BANKS else ipos.TOP_BANKS[0]


def _perturb_deal(d: ipos.Deal, cutoff: str, rng: np.random.Generator) -> ipos.Deal:
    """A copy of the deal with every fact published after the cutoff's evening changed: ranges and documents
    (shares, lead, offer) filed after the cutoff day, and first trades on or after it (a first close is read
    the session after, so a listing on the cutoff day is not known that evening either). The latest-value
    fields (shares, lead, symbol, offer) change too whenever a filing after the cutoff could have set them."""
    e = copy.deepcopy(d)
    if e.first_trade and e.first_trade["date"] >= cutoff:
        e.first_trade["close"] = float(e.first_trade["close"]) * float(rng.uniform(0.2, 5.0))
    e.ranges = [
        r if r[0] <= cutoff else (r[0], r[1] * float(rng.uniform(0.5, 2)), r[2] * float(rng.uniform(0.5, 2)))
        for r in e.ranges
    ]
    for _, _, doc in (x for x in e.docs if x[0] > cutoff):
        if doc.get("shares"):
            doc["shares"] = float(doc["shares"]) * float(rng.uniform(0.2, 5.0))
        doc["lead"] = _other_bank(doc.get("lead"))
        if doc.get("offer"):
            doc["offer"] = float(doc["offer"]) * float(rng.uniform(0.5, 2.0))
    if any(x[0] > cutoff for x in e.docs) or (e.priced or "") > cutoff:
        e.shares = e.shares * float(rng.uniform(0.2, 5.0)) if e.shares else e.shares
        e.lead, e.symbol = _other_bank(e.lead), f"{e.symbol}X"
    if e.offer and (e.priced or "") > cutoff:
        e.offer = e.offer * float(rng.uniform(0.5, 2.0))
    return e


def leakage_ipos(
    deals: Sequence[ipos.Deal], macro_rows: pd.DataFrame, cutoffs: Sequence[str], pop: float, seed: int = 7
) -> Dict[str, Any]:
    """Change every IPO fact published after each cutoff's prediction time (the close plus FILING_DELAY on the
    cutoff day): later ranges and documents (including the final prospectus's shares, lead and offer), first
    trades on or after the cutoff day, and macro values by available_at. The features of deals predicted by the
    cutoff must not change. Every row's prediction moment must also come before its first trade."""
    rng = np.random.default_rng(seed)
    base = features.ipo_rows(deals, macro_rows, pop)
    traded = base[base["trade_date"].notna()]
    late = int((traded["moment"] >= traded["trade_date"]).sum())
    worst, checks = (float("inf") if late else 0.0), []
    for c in cutoffs:
        alt_deals = [_perturb_deal(d, c, rng) for d in deals]
        when = markets.close_utc(dt.date.fromisoformat(c)) + markets.FILING_DELAY
        alt = features.ipo_rows(alt_deals, _perturb_macro(macro_rows, when, rng), pop)
        a = base[base["moment"] <= c].reset_index(drop=True)
        b = alt[alt["moment"] <= c].reset_index(drop=True)
        diff = _same(a, b, features.IPO_FEATURES)
        checks.append({"cutoff": c, "feature_diff": diff, "deals": len(a)})
        worst = max(worst, diff)
    return {"passed": worst == 0.0, "worst_diff": worst, "checks": checks, "moment_not_before_trade": late}


def to_json(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore[arg-type]
        raise TypeError(type(o))

    return json.dumps(obj, default=default, indent=2, sort_keys=True, allow_nan=True)


def markdown(rep: Dict[str, Any], title: str, forecasters: Sequence[str]) -> List[str]:
    """One domain's section of the backtest report."""
    lines = [f"## {title}", ""]
    if not rep.get("n"):
        return [*lines, "No resolved out-of-sample predictions yet.", ""]
    lines += [
        f"{rep['n']:,} out-of-sample predictions, {rep['first']} to {rep['last']}.",
        "",
        "| forecaster | Brier | log loss | hit rate | ECE | MAE | RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in forecasters:
        s = rep["scores"][f]
        lines.append(
            f"| {f} | {s['brier']:.4f} | {s['log_loss']:.4f} | {s['hit_rate']:.1%} | {s['ece']:.3f} | "
            f"{s['mae']:.4f} | {s['rmse']:.4f} |"
        )
    lines += [
        "",
        "| comparison | mean loss difference | DM p | DM p (Holm) | bootstrap 95% CI | bootstrap p |",
        "|---|---|---|---|---|---|",
    ]
    for name, t in rep["tests"].items():
        lines.append(
            f"| {name} | {t['dm']['mean_diff']:+.5f} | {t['dm']['p']:.3g} | {t['dm_p_holm']:.3g} | "
            f"[{t['bootstrap']['lo']:+.5f}, {t['bootstrap']['hi']:+.5f}] | {t['bootstrap']['p']:.3g} |"
        )
    verdict = "beats" if rep["beats_best_baseline"] else "does not beat"
    lines += [
        "",
        f"The model {verdict} the best baseline ({rep['best_baseline']}) on Brier at the 5% level after Holm.",
        "A negative difference means the model's loss is lower.",
        "",
    ]
    return lines
