"""Every network read goes through here: timeouts, retries with exponential backoff, HTTP 429 handling, a rate
limit per host, a raw snapshot of each response, and a replay mode for offline tests.

Replay (settings.replay_dir): responses are read from recorded files keyed by the URL, never from the network;
an unrecorded URL raises SourceError("not recorded"), like a source that is down. Record (settings.record_dir):
every response is also written there, to make those files.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import pathlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Mapping, Optional

from bot.config import Settings

try:  # the OS certificate store, needed behind traffic-inspecting networks
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover  (truststore is in requirements; this keeps a bare install importable)
    pass


class SourceError(Exception):
    """A source could not be read. kind: timeout, rate_limited, http, blocked, not_found, not_configured,
    not_recorded, unreachable, schema."""

    def __init__(self, kind: str, message: str, status: Optional[int] = None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.status = kind, status


def key(url: str) -> str:
    """The file name a URL's response is recorded under."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


Opener = Callable[[urllib.request.Request, float], bytes]


def _open(req: urllib.request.Request, timeout: float) -> bytes:  # pragma: no cover  (the real network; tests inject)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return bytes(r.read())


class Http:
    """A polite client. sleep and opener are injectable so tests run instantly and offline."""

    def __init__(
        self,
        cfg: Settings,
        opener: Opener = _open,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg, self.opener, self.sleep, self.clock = cfg, opener, sleep, clock
        self._next: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.requests = 0

    def _wait_turn(self, host: str) -> None:
        rate = self.cfg.rate_limits.get(host, self.cfg.default_rate)
        with self._lock:
            now = self.clock()
            start = max(now, self._next.get(host, now))
            self._next[host] = start + 1.0 / rate
        if start > now:
            self.sleep(start - now)

    def get(self, url: str, headers: Optional[Mapping[str, str]] = None, limit: Optional[int] = None) -> bytes:
        """The body at url (its first `limit` bytes, when given). Raises SourceError when every try fails."""
        host = urllib.parse.urlsplit(url).netloc
        agent = self.cfg.user_agent
        if host.endswith("sec.gov"):
            if not self.cfg.sec_user_agent:
                raise SourceError(
                    "not_configured", 'EDGAR needs SEC_USER_AGENT="Your Name you@example.com" (fair-access policy)'
                )
            agent = self.cfg.sec_user_agent
        if self.cfg.replay_dir is not None:
            body = self._replay(url)
            return body[:limit] if limit else body
        hdrs = {"User-Agent": agent, **(headers or {})}
        last: Optional[SourceError] = None
        for attempt in range(self.cfg.retries):
            self._wait_turn(host)
            self.requests += 1
            try:
                body = self.opener(urllib.request.Request(url, headers=hdrs), self.cfg.timeout_s)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    last = SourceError("rate_limited", f"429 from {host}")
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    wait = (
                        float(retry_after) if retry_after and retry_after.isdigit() else self.cfg.backoff_s * 2**attempt
                    )
                    self.sleep(wait)
                    continue
                if e.code in (500, 502, 503, 504):
                    last = SourceError("http", f"{e.code} from {host}")
                    self.sleep(self.cfg.backoff_s * 2**attempt)
                    continue
                kind = "blocked" if e.code in (401, 403) else "not_found" if e.code == 404 else "http"
                if self.cfg.record_dir is not None:  # a refusal is part of what a replay must reproduce
                    record_failure(self.cfg.record_dir, url, kind, e.code)
                raise SourceError(kind, f"{e.code} from {host}", e.code) from e
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
                timed_out = isinstance(e, TimeoutError) or "timed out" in str(e)
                last = SourceError("timeout" if timed_out else "unreachable", f"{host}: {e}")
                self.sleep(self.cfg.backoff_s * 2**attempt)
                continue
            body = body[:limit] if limit else body
            self._snapshot(url, body)
            return body
        assert last is not None
        raise last

    def _replay(self, url: str) -> bytes:
        assert self.cfg.replay_dir is not None
        path = self.cfg.replay_dir / key(url)
        if not path.exists():
            raise SourceError("not_recorded", url)
        meta = path.with_suffix(".json")
        if meta.exists():  # a recorded failure: replay it as that failure
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("error"):
                raise SourceError(m["error"], url, m.get("status"))
        return path.read_bytes()

    def _snapshot(self, url: str, body: bytes) -> None:
        """Keep the raw response: under data/raw/<date>/<host>/, never overwritten, and in record_dir when set."""
        stamp = dt.datetime.now(dt.UTC)
        host = urllib.parse.urlsplit(url).netloc
        folder = self.cfg.data_dir / "raw" / stamp.strftime("%Y-%m-%d") / host
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{stamp.strftime('%H%M%S%f')}-{key(url)}"
        (folder / f"{name}.gz").write_bytes(gzip.compress(body))
        (folder / f"{name}.json").write_text(json.dumps({"url": url, "fetched": stamp.isoformat()}), encoding="utf-8")
        if self.cfg.record_dir is not None:
            record(self.cfg.record_dir, url, body)


def record(folder: pathlib.Path, url: str, body: bytes) -> None:
    """Write a fixture that replays as this response."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / key(url)).write_bytes(body)
    (folder / f"{key(url)}.url").write_text(url, encoding="utf-8")


def record_failure(folder: pathlib.Path, url: str, kind: str, status: Optional[int] = None) -> None:
    """Write a fixture that replays as a failed request (for tests of error paths)."""
    record(folder, url, b"")
    (folder / f"{key(url)}.json").write_text(json.dumps({"error": kind, "status": status}), encoding="utf-8")
