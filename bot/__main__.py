"""Command line: python -m bot <command>.

update [--job daily|edgar|weekly|monthly|all] [--now 2026-10-08T22:00:00Z]
                      run a job (default: all, what a fresh clone needs)
backtest              the weekly job: walk-forward backtest and the leakage test
retrain               the monthly job: refit, adopt only if better
health                print the latest health report's summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import List, Optional

from bot.config import settings
from bot.store import LockedError, parse_iso
from bot.update import JOBS, run


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update", help="run a job")
    up.add_argument("--job", choices=JOBS, default="all")
    up.add_argument("--now", help="the run's clock, UTC (for replaying a past day against recorded responses)")
    sub.add_parser("backtest", help="walk-forward backtest and leakage test")
    sub.add_parser("retrain", help="refit the models; adopt only if better")
    sub.add_parser("health", help="print the latest health summary")
    args = ap.parse_args(argv)
    cfg = settings()
    if args.cmd == "health":
        path = cfg.data_dir / "health" / "latest.json"
        if not path.exists():
            print("no health report yet: run `python -m bot update` first")
            return 1
        r = json.loads(path.read_text(encoding="utf-8"))
        print(
            json.dumps({k: r[k] for k in ("run_id", "status", "through", "coverage", "alerts")}, indent=2, default=str)[
                :4000
            ]
        )
        return 0 if r["status"] != "failing" else 2
    job = {"update": getattr(args, "job", "all"), "backtest": "weekly", "retrain": "monthly"}[args.cmd]
    now = parse_iso(args.now) if getattr(args, "now", None) else dt.datetime.now(dt.UTC)
    try:
        out = run(job, cfg, now)
    except LockedError as e:
        print(f"skipped: {e}", file=sys.stderr)
        return 3
    print(
        f"{job}: {out['status']} ({out.get('warnings', 0)} warnings); health: {out.get('health_status')} -> {out.get('health')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
