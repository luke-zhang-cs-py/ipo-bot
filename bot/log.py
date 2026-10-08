"""Structured logs: one JSON object a line, to stderr and to data/logs/bot.jsonl."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Any

from bot.store import iso, utc_now


class _Json(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"at": iso(utc_now()), "level": record.levelname.lower(), "event": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        return json.dumps(out, default=str, sort_keys=True)


def setup(data_dir: pathlib.Path, level: int = logging.INFO) -> logging.Logger:
    """The bot's logger, writing JSON lines (idempotent: a second call does not add handlers)."""
    log = logging.getLogger("bot")
    log.setLevel(level)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    for h in (
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(data_dir / "logs" / "bot.jsonl", encoding="utf-8"),
    ):
        h.setFormatter(_Json())
        log.addHandler(h)
    return log


def event(log: logging.Logger, name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log an event with fields: event(log, "fetched", source="yahoo", rows=10)."""
    log.log(level, name, extra={"fields": fields})
