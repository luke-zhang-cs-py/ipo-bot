"""The jobs `python -m bot` runs. Each holds the update lock, records a run row (start, end, status, summary),
and ends by writing the health report.

  daily    universe, prices, macro, reconciliation, scoring, stock and IPO predictions   weekdays 22:00 UTC
  edgar    EDGAR filings, companies, documents, first trades; IPO scoring and predictions every 3 h, weekdays
  weekly   the walk-forward backtest and the leakage test, written to reports/backtest.md   Saturdays 06:00 UTC
  monthly  retrain both models; a new one is adopted only if it beats the current one      the 1st, 08:00 UTC
  all      daily then edgar: what a fresh clone runs (`python -m bot update`)

A run whose sources fail finishes as "partial" with warnings; one that crashes finishes as "failed" (and
re-raises). Running a job twice does nothing new the second time.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional

from bot import __version__, checks, collect, data, edgar, evaluate, features, health, markets, tracking
from bot.config import Settings
from bot.context import Ctx
from bot.http import Http
from bot.log import event, setup
from bot.store import Store, lock, utc_now

JOBS = ("daily", "edgar", "weekly", "monthly", "all")


def run(job: str, cfg: Settings, now: Optional[dt.datetime] = None, http: Optional[Http] = None) -> Dict[str, Any]:
    """Run one job; returns its summary (with "status" and the health report's path)."""
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; one of {', '.join(JOBS)}")
    now = now or utc_now()
    log = setup(cfg.data_dir)
    with lock(cfg.data_dir / "update.lock", cfg.lock_stale_hours, now) as stale:
        store = Store(cfg.data_dir / "bot.sqlite")
        run_id = f"{now:%Y%m%dT%H%M%SZ}-{job}-{uuid.uuid4().hex[:6]}"
        ctx = Ctx(cfg, http or Http(cfg), store, log, run_id, now)
        if stale:
            ctx.warn("stale_lock", detail=stale)
        store.start_run(run_id, job, __version__, now)
        event(log, "run_started", job=job, run_id=run_id, replay=bool(cfg.replay_dir))
        summary: Dict[str, Any] = {}
        try:
            steps: Dict[str, Callable[[Ctx], Dict[str, Any]]] = {
                "daily": daily,
                "edgar": edgar_job,
                "weekly": weekly,
                "monthly": monthly,
            }
            for name in ("daily", "edgar") if job == "all" else (job,):
                summary[name] = steps[name](ctx)
            report = health.build(ctx, summary.get("daily", {}).get("reconciled"))
            summary["health"] = str(health.write(ctx, report))
            summary["health_status"] = report["status"]
            summary["warnings"] = len(ctx.warnings)
            summary["counts"] = ctx.counts
            status = "partial" if ctx.warnings else "ok"
        except Exception as e:
            store.finish_run(run_id, "failed", {"error": f"{type(e).__name__}: {e}", **_jsonable(summary)})
            event(log, "run_failed", logging.ERROR, job=job, run_id=run_id, error=str(e))
            store.close()
            raise
        store.finish_run(run_id, status, _jsonable(summary))
        event(
            log,
            "run_finished",
            job=job,
            run_id=run_id,
            status=status,
            warnings=len(ctx.warnings),
            requests=ctx.http.requests,
        )
        store.close()
        return {"run_id": run_id, "status": status, **summary}


def _jsonable(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = json.loads(json.dumps(summary, default=str))
    for v in out.values():
        if isinstance(v, dict):
            v.pop("reconciled", None)
    return out


def daily(ctx: Ctx) -> Dict[str, Any]:
    symbols = collect.update_universe(ctx)
    prices = collect.update_prices(ctx, symbols)
    leavers = collect.backfill_leavers(ctx, symbols)
    macro = collect.update_macro(ctx)
    day = markets.last_closed_session(ctx.now)
    bars = data.bars(ctx.store, symbols=symbols)
    recent = bars[bars["date"] >= markets.previous_trading_day(markets.previous_trading_day(day)).isoformat()]
    # symbols whose newest moves look wrong are reconciled today whatever the rotation says
    moved = checks.moves(recent, data.splits(ctx.store), data.dividends(ctx.store), ctx.cfg.max_daily_move)
    flagged = {i["subject"].split()[0] for i in moved}
    reconciled = collect.reconcile(ctx, [*collect.rotation(symbols, day), *sorted(flagged)])
    scored = tracking.score(ctx)
    made = tracking.predict(ctx)
    return {
        "symbols": len(symbols),
        "price_sources": _tally(prices),
        "leavers_read": leavers,
        "macro": macro,
        "reconciled": reconciled,
        "reconcile_failures": sum(not r["ok"] for r in reconciled),
        "scored": scored,
        "predictions": made,
    }


def _tally(sources: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in sources.values():
        out[s] = out.get(s, 0) + 1
    return out


def edgar_job(ctx: Ctx) -> Dict[str, Any]:
    out = edgar.update(ctx)
    out["scored"] = tracking.score(ctx)
    out["predictions"] = {"ipo": tracking.predict_ipos(ctx)}
    tracking.mirror(ctx)
    return out


def weekly(ctx: Ctx) -> Dict[str, Any]:
    """The walk-forward backtest of both models against their baselines, and the leakage test."""
    st = ctx.store
    macro = tracking.macro_rows(st)
    panel = tracking.stock_inputs(st)
    stock = evaluate.stock_walkforward(panel)
    rep_s = evaluate.report(stock, "date", ("model", "base", "rw"))
    ipo_rows = tracking.ipo_inputs(st, ctx.cfg.ipo_pop)
    ipo = evaluate.ipo_walkforward(ipo_rows)
    rep_i = evaluate.report(ipo, "moment", ("model", "base", "recent"))
    syms = data.universe(st)  # every past and present member: the panel's symbols
    closes = data.closes(st, None, [*syms, features.INDEX])
    dates = list(closes.index)
    cut_s = [dates[int(len(dates) * f)] for f in (0.5, 0.75, 0.95)] if len(dates) > 300 else dates[-3:-2]
    moments = sorted(ipo_rows["moment"]) if not ipo_rows.empty else []
    cut_i = [moments[int(len(moments) * f)] for f in (0.5, 0.9)] if len(moments) > 20 else moments[-1:]
    leak = {
        "stock": evaluate.leakage_stocks(closes, macro, cut_s),
        "ipo": evaluate.leakage_ipos(edgar.deals_asof(st), macro, cut_i, ctx.cfg.ipo_pop),
    }
    result = {"stock": rep_s, "ipo": rep_i, "leakage": leak, "at": ctx.at, "run_id": ctx.run_id}
    folder = ctx.cfg.data_dir / "backtest"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "latest.json").write_text(evaluate.to_json(result), encoding="utf-8")
    lines = [
        "# Backtest",
        "",
        f"Walk-forward, out of sample, run `{ctx.run_id}` at {ctx.at}. Stocks: next-session direction (close up) of "
        "S&P 500 members; IPOs: a first close 20% or more above the offer. Base rate, random walk and average "
        "recent IPO are the baselines; DM is Diebold-Mariano (HLN-corrected), the bootstrap is a moving-block one.",
        "",
        *evaluate.markdown(rep_s, "Stocks: next-day direction", ("model", "base", "rw")),
        *evaluate.markdown(rep_i, "IPOs: first-day pop", ("model", "base", "recent")),
        "## Leakage test (scrambled future)",
        "",
        f"- stocks: {'passed' if leak['stock']['passed'] else 'FAILED'} at cutoffs "
        + ", ".join(c["cutoff"] for c in leak["stock"]["checks"])
        + f" (largest change {leak['stock']['worst_diff']:.3g})",
        f"- IPOs: {'passed' if leak['ipo']['passed'] else 'FAILED'} at cutoffs "
        + ", ".join(c["cutoff"] for c in leak["ipo"]["checks"])
        + f" (largest change {leak['ipo']['worst_diff']:.3g})",
        "",
    ]
    reports = ctx.cfg.reports_dir or ctx.cfg.data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "backtest.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    for k in ("stock", "ipo"):
        if not leak[k]["passed"]:
            ctx.warn("leakage_test_failed", kind=k, worst=leak[k]["worst_diff"])
    return {
        "stock_n": rep_s["n"],
        "ipo_n": rep_i["n"],
        "leakage_passed": all(v["passed"] for v in leak.values()),
        "report": str(reports / "backtest.md"),
    }


def monthly(ctx: Ctx) -> Dict[str, Any]:
    return {k: tracking.retrain(ctx, k) for k in ("stock", "ipo")}
