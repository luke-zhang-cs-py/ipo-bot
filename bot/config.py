"""Every setting the bot runs on, in one place: paths, sources' rate limits, check tolerances and schedule.

Overridable from the environment (none is required): BOT_DATA_DIR, SEC_USER_AGENT, BOT_REPLAY (a directory of
recorded responses: the offline mode tests use), BOT_RECORD (save every response there, to make fixtures),
BOT_SYMBOLS (a comma-separated universe instead of the S&P 500, for a small run), BOT_HISTORY_START,
BOT_SEC_BUDGET (EDGAR requests a run may make).

SEC_USER_AGENT is not a key: EDGAR's fair-access policy asks every caller to name itself with a contact address
("Jane Doe jane@example.com") and refuses requests that don't. Without it the bot runs everything else and skips
EDGAR with a warning; it never sends a made-up contact.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _env_file(path: pathlib.Path) -> Dict[str, str]:
    """KEY=value lines of a .env file, if there is one (comments and blank lines skipped)."""
    out: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = v.strip("\"'")
    return out


@dataclass(frozen=True)
class Settings:
    data_dir: pathlib.Path
    sec_user_agent: Optional[str]
    replay_dir: Optional[pathlib.Path]
    record_dir: Optional[pathlib.Path]
    symbols: Optional[Tuple[str, ...]] = None  # None: the S&P 500
    tracking_dir: Optional[pathlib.Path] = None  # committed JSONL ledgers (None: data_dir/tracking)
    reports_dir: Optional[pathlib.Path] = None  # committed health and backtest reports (None: data_dir/reports)
    user_agent: str = "ipo-bot/1.0 (+https://github.com/luke-zhang-cs-py/ipo-bot)"
    # sources: requests a second per host (SEC's published limit is 10; Yahoo and Nasdaq publish none)
    rate_limits: Mapping[str, float] = field(
        default_factory=lambda: {"www.sec.gov": 8.0, "efts.sec.gov": 8.0, "data.sec.gov": 8.0}
    )
    default_rate: float = 2.0
    timeout_s: float = 30.0
    retries: int = 4
    max_retry_after_s: float = 60.0  # a server asking to wait longer is treated as down for this run
    backoff_s: float = 1.0  # 1, 2, 4, 8 seconds between tries (or Retry-After when given)
    # checks
    reconcile_tolerance: float = 0.005  # two price sources must agree within 0.5%
    max_daily_move: float = 0.5  # a daily move beyond 50% needs a split or dividend to explain it
    stale_trading_days: int = 1  # the newest price may lag the last finished session by this much
    warn_periods: int = 3  # rolling periods below the best baseline before a warning
    calibration_drift: float = 0.10  # rolling calibration error that counts as drift
    # history
    history_start: str = "2016-01-04"
    reconcile_days: int = 20  # recent trading days checked against the second price source
    refetch_days: int = 5  # trading days re-read before the newest stored, to catch revisions
    min_members: int = 400  # fewer parsed from Wikipedia means its table changed shape
    sec_budget: int = 1500  # EDGAR requests a run may spend (the backfill takes what is left)
    ipo_pop: float = 0.20  # an IPO "pops" when its first close is 20% or more above the offer
    lock_stale_hours: float = 6.0  # a lock older than this is from a run that died


def settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """The settings, from the environment and .env (an explicit env mapping is used as is, for tests)."""
    if env is None:
        env = {**_env_file(ROOT / ".env"), **os.environ}
    custom = env.get("BOT_DATA_DIR")
    data = pathlib.Path(custom) if custom else ROOT / "data"
    ua = (env.get("SEC_USER_AGENT") or "").strip()
    replay = env.get("BOT_REPLAY")
    record = env.get("BOT_RECORD")
    symbols = tuple(s.strip().upper() for s in env.get("BOT_SYMBOLS", "").split(",") if s.strip()) or None
    return Settings(
        data_dir=data,
        sec_user_agent=ua if "@" in ua else None,
        replay_dir=pathlib.Path(replay) if replay else None,
        record_dir=pathlib.Path(record) if record else None,
        symbols=symbols,
        tracking_dir=None if custom else ROOT / "tracking" / "bot",
        reports_dir=None if custom else ROOT / "reports",
        history_start=env.get("BOT_HISTORY_START") or Settings.history_start,
        sec_budget=int(env.get("BOT_SEC_BUDGET") or Settings.sec_budget),
    )
