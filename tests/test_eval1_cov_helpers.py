"""Shared fakes for the test_eval1_cov_* files: a Yahoo chart response served without the network, and a
stdout that has no reconfigure(). No tests here."""
import datetime as dt
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for sub in ("src", "evaluation"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))


def ts(day):
    """A Yahoo timestamp for an ISO day (mid-session, so it maps back to the same UTC date)."""
    d = dt.date.fromisoformat(day)
    return int(dt.datetime(d.year, d.month, d.day, 14, 30, tzinfo=dt.timezone.utc).timestamp())


def chart(days, opens, closes, adj, volume=None, dividends=None, splits=None):
    """A Yahoo v8 chart payload."""
    events = {}
    if dividends:
        events["dividends"] = {str(k): {"date": ts(d), "amount": a} for k, (d, a) in enumerate(dividends.items())}
    if splits:
        events["splits"] = {str(k): {"date": ts(d), "numerator": n, "denominator": m}
                            for k, (d, (n, m)) in enumerate(splits.items())}
    return {"chart": {"result": [{"timestamp": [ts(d) for d in days],
                                  "indicators": {"quote": [{"open": opens, "close": closes,
                                                            "volume": volume or [1000] * len(days)}],
                                                 "adjclose": [{"adjclose": adj}]},
                                  **({"events": events} if events else {})}]}}


class Served:
    """A fake urllib.request.urlopen: hands back `payload` (or payload(url)) and records the URLs asked for."""

    def __init__(self, payload):
        self.payload, self.urls = payload, []

    def __call__(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        self.urls.append(url)
        body = self.payload(url) if callable(self.payload) else self.payload
        return io.BytesIO(json.dumps(body).encode())


def no_network(*a, **k):
    raise AssertionError("the network was used")


def plain_streams(monkeypatch):
    """stdout and stderr without reconfigure() (the except branch every main() has); returns stdout."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out
