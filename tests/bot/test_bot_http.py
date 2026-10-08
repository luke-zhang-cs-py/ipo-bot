"""The HTTP client: retries with backoff, 429s, time-outs, error kinds, rate limits, snapshots, record and replay."""

import gzip
import io
import json
import urllib.error
from email.message import Message
from typing import List

import pytest
from conftest import UA, Opener, make_cfg

from bot import http as bhttp
from bot.http import Http, SourceError


def err(code: int, retry_after: str = "") -> urllib.error.HTTPError:
    h = Message()
    if retry_after:
        h["Retry-After"] = retry_after
    return urllib.error.HTTPError("u", code, "x", h, io.BytesIO(b""))


def client(tmp_path, routes, **kw):
    sleeps: List[float] = []
    cfg = make_cfg(tmp_path, **kw)
    return Http(cfg, opener=Opener(routes), sleep=sleeps.append, clock=lambda: 0.0), sleeps


def test_retries_then_succeeds_with_exponential_backoff(tmp_path) -> None:
    answers = iter([err(503), TimeoutError("timed out"), b"ok"])
    h, sleeps = client(tmp_path, {"x.test": lambda u: next(answers)}, rate_limits={"x.test": 1e9})
    assert h.get("https://x.test/a") == b"ok"
    assert [s for s in sleeps if s >= 0.01] == [1.0, 2.0]  # 1 s after the 503, 2 s after the time-out
    assert h.requests == 3


def test_429_waits_for_retry_after(tmp_path) -> None:
    answers = iter([err(429, "7"), err(429), b"ok"])
    h, sleeps = client(tmp_path, {"x.test": lambda u: next(answers)}, rate_limits={"x.test": 1e9})
    assert h.get("https://x.test/a") == b"ok"
    assert [s for s in sleeps if s >= 0.01] == [7.0, 2.0]  # Retry-After first, then backoff (1 * 2**1)


def test_every_try_failing_raises_the_last_error(tmp_path) -> None:
    h, _ = client(tmp_path, {"x.test": err(429)}, retries=2)
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/a")
    assert e.value.kind == "rate_limited"
    h, _ = client(tmp_path, {"x.test": ConnectionResetError("reset")}, retries=2)
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/a")
    assert e.value.kind == "unreachable"
    h, _ = client(tmp_path, {"x.test": urllib.error.URLError("timed out")}, retries=1)
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/a")
    assert e.value.kind == "timeout"


@pytest.mark.parametrize("code,kind", [(403, "blocked"), (401, "blocked"), (404, "not_found"), (400, "http")])
def test_client_errors_are_not_retried(tmp_path, code, kind) -> None:
    h, _ = client(tmp_path, {"x.test": err(code)})
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/a")
    assert (e.value.kind, e.value.status, h.requests) == (kind, code, 1)


def test_rate_limit_spaces_requests_per_host(tmp_path) -> None:
    sleeps: List[float] = []
    cfg = make_cfg(tmp_path, rate_limits={"a.test": 4.0})
    h = Http(cfg, opener=Opener({"test": b"x"}), sleep=sleeps.append, clock=lambda: 100.0)
    for _ in range(3):
        h.get("https://a.test/")
    h.get("https://b.test/")  # another host: its own pace
    assert sleeps == [0.25, 0.5]


def test_snapshots_and_recording(tmp_path) -> None:
    rec = tmp_path / "rec"
    h, _ = client(tmp_path, {"x.test": b"body-bytes"}, record_dir=rec)
    assert h.get("https://x.test/a?b=1", limit=4) == b"body"
    snaps = list((tmp_path / "raw").rglob("*.gz"))
    assert len(snaps) == 1 and gzip.decompress(snaps[0].read_bytes()) == b"body"
    meta = json.loads(snaps[0].with_suffix(".json").read_text())
    assert meta["url"] == "https://x.test/a?b=1"
    assert (rec / bhttp.key("https://x.test/a?b=1")).read_bytes() == b"body"


def test_replay_never_touches_the_network(tmp_path) -> None:
    rep = tmp_path / "rep"
    bhttp.record(rep, "https://x.test/a", b"hello")
    bhttp.record_failure(rep, "https://x.test/down", "timeout")
    h, _ = client(tmp_path, {}, replay_dir=rep)
    assert h.get("https://x.test/a") == b"hello" and h.get("https://x.test/a", limit=2) == b"he"
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/down")
    assert e.value.kind == "timeout"
    with pytest.raises(SourceError) as e:
        h.get("https://x.test/never")
    assert e.value.kind == "not_recorded"
    bhttp.record(rep, "https://x.test/meta", b"x")
    (rep / f"{bhttp.key('https://x.test/meta')}.json").write_text("{}")
    assert h.get("https://x.test/meta") == b"x"  # a metadata file without an error replays the body


def test_sec_needs_a_named_user_agent(tmp_path) -> None:
    op = Opener({"sec.gov": b"ok"})
    h = Http(make_cfg(tmp_path, sec_user_agent=None), opener=op, sleep=lambda s: None)
    with pytest.raises(SourceError) as e:
        h.get("https://www.sec.gov/x")
    assert e.value.kind == "not_configured" and not op.calls
    seen = []
    h = Http(
        make_cfg(tmp_path),
        opener=lambda req, t: seen.append(req.get_header("User-agent")) or b"ok",
        sleep=lambda s: None,
    )
    h.get("https://data.sec.gov/x")
    h.get("https://example.test/x")
    assert seen[0] == UA and "ipo-bot" in seen[1]


def test_recorded_refusals_replay_as_refusals(tmp_path) -> None:
    rec = tmp_path / "rec"
    h, _ = client(tmp_path, {"x.test": err(403)}, record_dir=rec)
    with pytest.raises(SourceError):
        h.get("https://x.test/missing")
    replay, _ = client(tmp_path, {}, replay_dir=rec)
    with pytest.raises(SourceError) as e:
        replay.get("https://x.test/missing")
    assert (e.value.kind, e.value.status) == ("blocked", 403)
