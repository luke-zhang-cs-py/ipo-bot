"""What one run carries around: settings, the HTTP client, the store, the logger, the run's id and clock, and
the warnings it collects for the health report."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from bot.config import Settings
from bot.http import Http
from bot.log import event
from bot.store import Store, iso


@dataclass
class Ctx:
    cfg: Settings
    http: Http
    store: Store
    log: logging.Logger
    run_id: str
    now: dt.datetime
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def at(self) -> str:
        """The run's clock as stored text."""
        return iso(self.now)

    def warn(self, what: str, **fields: Any) -> None:
        """A problem the run worked around: logged now and listed in the health report."""
        self.warnings.append({"what": what, **fields})
        event(self.log, what, logging.WARNING, **fields)

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n
