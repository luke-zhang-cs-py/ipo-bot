"""Shared fixtures for the bot's offline tests: no network (a socket guard fails any real connection), no keys."""

from __future__ import annotations

import datetime as dt
import gzip
import logging
import pathlib
import socket
import sys
from typing import Any, Callable, Dict, Iterator, Optional

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from world import World, at  # noqa: E402

from bot.config import Settings  # noqa: E402
from bot.context import Ctx  # noqa: E402
from bot.http import Http  # noqa: E402
from bot.store import Store  # noqa: E402

RECORDED = HERE / "fixtures" / "recorded"
UA = "Test Bot test@example.com"


def recorded(name: str) -> bytes:
    """A real response recorded from its source (trimmed), stored gzipped."""
    return gzip.decompress((RECORDED / f"{name}.gz").read_bytes())


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to open a real connection fails the test."""

    def guard(*a: Any, **k: Any) -> None:
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def make_cfg(tmp: pathlib.Path, **kw: Any) -> Settings:
    base: Dict[str, Any] = dict(
        data_dir=tmp,
        sec_user_agent=UA,
        replay_dir=None,
        record_dir=None,
        history_start="2023-01-03",
        min_members=3,
        sec_budget=5000,
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def cfg(tmp_path: pathlib.Path) -> Settings:
    return make_cfg(tmp_path)


class Opener:
    """A fake network: url -> bytes, or an exception to raise, or a callable(url) giving either."""

    def __init__(self, routes: Optional[Dict[str, Any]] = None):
        self.routes = routes or {}
        self.calls: list = []

    def __call__(self, req: Any, timeout: float) -> bytes:
        url = req.full_url
        self.calls.append(url)
        for key, what in self.routes.items():
            if key in url:
                if callable(what) and not isinstance(what, type):
                    what = what(url)
                if isinstance(what, BaseException):
                    raise what
                return bytes(what)
        raise AssertionError(f"unexpected request {url}")


def http(cfg: Settings, opener: Callable[..., bytes]) -> Http:
    return Http(cfg, opener=opener, sleep=lambda s: None)


def make_ctx(cfg: Settings, opener: Callable[..., bytes], now: dt.datetime, store: Optional[Store] = None) -> Ctx:
    log = logging.getLogger("bot.test")
    st = store or Store(cfg.data_dir / "bot.sqlite")
    OPEN.append(st)
    return Ctx(cfg, http(cfg, opener), st, log, "run-test", now)


OPEN: list = []  # stores made by make_ctx, closed after each test


@pytest.fixture(autouse=True)
def fresh_calendar() -> None:
    """The holiday table is cached per year; start each test without it, so a test always runs the real rules
    (the mutation run depends on this: a cached year would hide a mutated rule)."""
    from bot import markets

    markets.holidays.cache_clear()


@pytest.fixture(autouse=True)
def close_stores() -> Iterator[None]:
    yield
    while OPEN:
        OPEN.pop().close()


@pytest.fixture
def world() -> World:
    return World(now=at(dt.date(2025, 10, 8)))


@pytest.fixture
def store(tmp_path: pathlib.Path) -> Iterator[Store]:
    s = Store(tmp_path / "t.sqlite")
    yield s
    s.close()
