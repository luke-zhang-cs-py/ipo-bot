"""The health report every run writes: data coverage of the whole universe (missing tickers, dates and fields),
every check's issues, source problems the run worked around, IPO pipeline counts, and live tracking.

Written to data/health/<run>.json and data/health/latest.json, and as Markdown to reports/health.md (committed,
so a glance at the repository shows whether the bot is healthy).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from bot import checks, collect, data, edgar, ipos, markets, tracking
from bot.context import Ctx

MISSING_FIELD_DAYS = 252  # missing fields are counted over the last year of bars
LIST_LIMIT = 25  # how many examples a list in the report shows


def build(ctx: Ctx, reconciled: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    st, cfg = ctx.store, ctx.cfg
    through = markets.last_closed_session(ctx.now)
    members = data.members(st)
    bars = data.bars(st, symbols=[*members, collect.INDEX])
    issues: List[Dict[str, Any]] = []
    issues += checks.prices(bars)
    issues += checks.moves(bars, data.splits(st), data.dividends(st), cfg.max_daily_move)
    macro = data.macro(st)
    macro_newest = {s: (macro[s].last_valid_index() if s in macro else None) for s in collect.MACRO_SERIES}
    issues += checks.staleness(bars, [*members, collect.INDEX], macro_newest, ctx.now, cfg)
    issues += checks.missed_runs(st, ctx.now)
    issues += checks.point_in_time(st)
    issues += checks.reconcile(reconciled or [], cfg.reconcile_tolerance)
    deals = edgar.deals_asof(st)
    issues += checks.ipo(deals)

    have = set(bars["symbol"]) if not bars.empty else set()
    gaps = checks.missing_days(bars, through) if not bars.empty else {}
    issues += checks.gap_issues(gaps)
    recent = bars[bars["date"] >= _ago(through, MISSING_FIELD_DAYS)] if not bars.empty else bars
    fields = {
        c: int(recent[c].isna().sum())
        for c in ("open", "high", "low", "close", "volume")
        if c in recent and len(recent)
    }
    stale = sorted(i["subject"] for i in issues if i["check"] == "staleness" and not i["subject"].startswith("job "))
    coverage = {
        "universe": len(members),
        "with_prices": len(have & set(members)),
        "current": len(members) - len([s for s in stale if s in members]),
        "missing_tickers": sorted(set(members) - have),
        "stale": stale[:LIST_LIMIT],
        "missing_dates": {
            "symbols": len(gaps),
            "days": sum(len(v) for v in gaps.values()),
            "examples": {s: v[:5] for s, v in sorted(gaps.items(), key=lambda kv: -len(kv[1]))[:LIST_LIMIT]},
        },
        "missing_fields_last_year": fields,
        "delisted_kept": len([r for r in st.asof("universe", where={"source": "nasdaqtrader"}) if r["listed"] == 0]),
        "membership_history_from": _first_snapshot(st),
    }
    status_counts = ipos.counts(deals, ctx.now.date())
    errors = [i for i in issues if i["severity"] == "error"]
    watch = {k: tracking.watch(st, k, cfg.warn_periods, cfg.calibration_drift) for k in ("stock", "ipo")}
    alerts = [w for v in watch.values() for w in v["warnings"]]
    status = "failing" if errors else "degraded" if (ctx.warnings or alerts or issues) else "ok"
    return {
        "run_id": ctx.run_id,
        "at": ctx.at,
        "through": through.isoformat(),
        "status": status,
        "coverage": coverage,
        "issues": {"total": len(issues), "by_check": _count(issues), "errors": errors[:100], "all": issues[:500]},
        "run_warnings": ctx.warnings,
        "counts": ctx.counts,
        "ipo_pipeline": {
            "deals": len(deals),
            "first_time_ipos": sum(d.ipo for d in deals),
            "by_status": status_counts,
            "with_offer": sum(1 for d in deals if d.ipo and d.offer),
            "with_first_trade": sum(1 for d in deals if d.ipo and d.first_trade),
            "edgar_backfill_reached": st.cursor("edgar_backfill"),
            "edgar_daily_through": st.cursor("edgar_daily"),
        },
        "tracking": {k: {"periods": v["periods"][-8:], "warnings": v["warnings"]} for k, v in watch.items()},
        "alerts": alerts,
    }


def _ago(day: Any, n: int) -> str:
    d = day
    for _ in range(n):
        d = markets.previous_trading_day(d)
    return str(d.isoformat())


def _first_snapshot(st: Any) -> Optional[str]:
    r = st.query("SELECT min(available_at) AS t FROM universe WHERE source IN ('wikipedia', 'config')")
    return r[0]["t"] if r and r[0]["t"] else None


def _count(issues: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in issues:
        out[i["check"]] = out.get(i["check"], 0) + 1
    return dict(sorted(out.items()))


def write(ctx: Ctx, report: Dict[str, Any]) -> pathlib.Path:
    folder = ctx.cfg.data_dir / "health"
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    (folder / f"{ctx.run_id}.json").write_text(text, encoding="utf-8")
    (folder / "latest.json").write_text(text, encoding="utf-8")
    reports = ctx.cfg.reports_dir or ctx.cfg.data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    md = reports / "health.md"
    md.write_text("\n".join(markdown(report)) + "\n", encoding="utf-8", newline="\n")
    return md


def markdown(r: Mapping[str, Any]) -> List[str]:
    c = r["coverage"]
    lines = [
        "# Bot health",
        "",
        f"Run `{r['run_id']}` at {r['at']}, data through the {r['through']} session. **Status: {r['status']}.**",
        "",
        "## Coverage",
        "",
        f"- Universe: {c['universe']} symbols; {c['with_prices']} with prices; {c['current']} current.",
        f"- Missing tickers (no prices at all): {', '.join(c['missing_tickers'][:LIST_LIMIT]) or 'none'}",
        f"- Stale: {', '.join(c['stale']) or 'none'}",
        f"- Missing trading days inside histories: {c['missing_dates']['days']} days over "
        f"{c['missing_dates']['symbols']} symbols",
        "- Missing fields in the last year of bars: "
        + (", ".join(f"{k} {v}" for k, v in c["missing_fields_last_year"].items() if v) or "none"),
        f"- Delisted symbols kept: {c['delisted_kept']}; membership history recorded from "
        f"{c['membership_history_from'] or 'not yet'} (earlier backtests use the membership seen then).",
        "",
        "## Checks",
        "",
    ]
    by = r["issues"]["by_check"]
    lines += [f"- {k}: {v}" for k, v in by.items()] or ["- no issues"]
    if r["issues"]["errors"]:
        lines += ["", "Errors (first 20):", ""]
        lines += [f"- {i['check']} {i['subject']}: {i['detail']}" for i in r["issues"]["errors"][:20]]
    lines += ["", "## Sources this run", ""]
    lines += [
        f"- {w['what']}: " + ", ".join(f"{k}={v}" for k, v in w.items() if k != "what") for w in r["run_warnings"][:30]
    ]
    if not r["run_warnings"]:
        lines.append("- every source answered")
    p = r["ipo_pipeline"]
    lines += [
        "",
        "## IPO pipeline",
        "",
        f"- {p['first_time_ipos']} first-time IPO deals of {p['deals']} registrants; by status: "
        + (", ".join(f"{k} {v}" for k, v in sorted(p["by_status"].items())) or "none"),
        f"- {p['with_offer']} with an offer price read, {p['with_first_trade']} with a first trade.",
        f"- EDGAR daily index read through {p['edgar_daily_through'] or 'not yet'}; backfill reached "
        f"{p['edgar_backfill_reached'] or 'not started'}.",
        "",
        "## Live tracking",
        "",
    ]
    for kind, t in r["tracking"].items():
        if not t["periods"]:
            lines.append(f"- {kind}: no resolved predictions yet")
            continue
        last = t["periods"][-1]
        parts = [f"{fam} Brier {v['brier']:.4f} (n={v['n']})" for fam, v in last.items() if fam != "period"]
        lines.append(f"- {kind}, {last['period']}: " + "; ".join(parts))
    lines += [f"- **{a}**" for a in r["alerts"]]
    return lines
