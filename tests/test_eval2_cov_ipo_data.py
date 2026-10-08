"""ipo_data's fetching and building, offline: the SEC and Yahoo are fakes, every file lives in tmp_path."""
import datetime as dt
import importlib
import io
import json
import pathlib
import sys
import types
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import ipo_data as D  # noqa: E402


def ts(day):
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())


def chart(start, n, open_=10.0, close=12.0, splits=None):
    days = [(dt.date.fromisoformat(start) + dt.timedelta(days=i)).isoformat() for i in range(n)]
    out = {"timestamp": [ts(d) for d in days], "indicators": {"quote": [{"open": [open_] * n, "close": [close] * n}]}}
    if splits is not None:
        out["events"] = {"splits": splits}
    return out


PROSPECTUS = ("<p>PROSPECTUS</p><p>5,000,000 Shares</p><p>Class A Common Stock</p><p>This is an initial public offering. "
              "The initial public offering price is $10.00 per share. We have applied to list on the Nasdaq under the "
              "symbol \"NEWC.\"</p><p>Goldman Sachs &amp; Co. LLC</p>").encode()
REGISTRATION = b"<p>We expect the initial public offering price to be between $9.00 and $11.00 per share.</p>"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)


# ----------------------------------------------------------------------------- first_days

def test_first_days_rejects_empty_charts_and_keeps_only_later_valid_splits():
    assert D.first_days({}, "2020-01-02", 10.0) is None
    assert D.first_days({"timestamp": [ts("2020-01-02")], "indicators": {"quote": [{"close": []}]}}, "2020-01-02", 10.0) is None
    splits = {"a": {"date": ts("2019-06-01"), "numerator": 1, "denominator": 10},     # before the listing: ignored
              "b": {"date": ts("2020-06-01"), "numerator": 2, "denominator": 1},      # later: undone
              "c": {"date": ts("2020-07-01"), "numerator": 0, "denominator": 1}}      # malformed: ignored
    c = chart("2020-01-02", 5, open_=5.0, close=6.0, splits=splits)
    c["indicators"]["quote"][0]["open"][0] = 0.0                                       # a zero open reads as missing
    out = D.first_days(c, "2020-01-02", None)
    assert out["split_factor"] == 2.0 and out["close"] == 12.0 and out["open"] is None
    assert out["close_21"] is None and out["close_252"] is None and out["suspect"] is False


# ----------------------------------------------------------------------------- the SEC client

def make_sec(monkeypatch, get):
    monkeypatch.setitem(sys.modules, "ipo_bot", types.SimpleNamespace(load_env=lambda: None))
    monkeypatch.setitem(sys.modules, "tools", types.SimpleNamespace(_sec_headers=lambda: {"User-Agent": "t t@x"}, _get=get))
    monkeypatch.setattr(D, "SEC_INTERVAL", 0.0)
    return D.Sec()


def test_sec_json_retries_then_gives_up(monkeypatch):
    calls = []

    def get(url, headers):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("flaky")
        return '{"ok": 1}'
    sec = make_sec(monkeypatch, get)
    assert sec.json("u") == {"ok": 1} and len(calls) == 2
    sec = make_sec(monkeypatch, lambda url, headers: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="three times"):
        sec.json("u")


def test_sec_head_retries_a_busy_server_and_raises_anything_else(monkeypatch):
    sec = make_sec(monkeypatch, None)
    script = []

    def urlopen(req, timeout):
        code = script.pop(0)
        if code:
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
        return io.BytesIO(b"0123456789")
    monkeypatch.setattr(D.urllib.request, "urlopen", urlopen)
    script[:] = [503, 429, 0]
    assert sec.head("http://x/f", limit=4) == b"0123"
    script[:] = [404]
    with pytest.raises(urllib.error.HTTPError):
        sec.head("http://x/f")
    script[:] = [503, 503, 503, 503]
    with pytest.raises(urllib.error.HTTPError):
        sec.head("http://x/f")
    assert script == []


class FakeSec:
    """json() and head() answered from dicts; anything else is an error, as a real 404 would be."""

    def __init__(self, pages=None, heads=None):
        self.pages, self.heads, self.urls = pages or {}, heads or {}, []

    def json(self, url):
        self.urls.append(url)
        for key, v in self.pages.items():
            if key in url:
                return v
        raise RuntimeError(f"no page for {url}")

    def head(self, url, limit=D.HEAD_BYTES):
        for key, v in self.heads.items():
            if url.endswith(key):
                return v
        raise RuntimeError(f"no filing at {url}")


def hit_source(i, name="Newco Inc  (NEWC)  (CIK 0000000042)", **kw):
    return {"_id": f"0000000042-20-{i:06d}:doc{i}.htm",
            "_source": {"adsh": f"0000000042-20-{i:06d}", "ciks": ["0000000042"], "display_names": [name],
                        "file_date": "2020-03-01", "sics": ["7372"], **kw}}


def test_prospectus_hits_pages_through_both_halves_and_parses_names():
    full = {"hits": {"hits": [hit_source(i) for i in range(100)]}}
    last = {"hits": {"hits": [hit_source(500, name="Odd Name Without Ticker (CIK 0000000042)", sics=None, ciks=None,
                                         display_names=None)]}}
    sec = FakeSec({"startdt=2020-01-01&enddt=2020-06-30&from=0": full, "startdt=2020-01-01&enddt=2020-06-30&from=100": last,
                   "startdt=2020-07-01": {}})
    out = D.prospectus_hits(sec, 2020)
    assert len(out) == 101
    assert out[0] == {"adsh": "0000000042-20-000000", "cik": "42", "name": "Newco Inc", "ticker": "NEWC",
                      "file": "doc0.htm", "file_date": "2020-03-01", "sic": "7372"}
    assert out[-1]["name"] == "" and out[-1]["ticker"] == "" and out[-1]["sic"] is None and out[-1]["cik"] == ""


def sub(forms, dates, files=()):
    return {"sic": "7372", "filings": {"recent": {"form": forms, "filingDate": dates,
                                                  "accessionNumber": [f"acc-{i}" for i in range(len(forms))],
                                                  "primaryDocument": [f"d{i}.htm" for i in range(len(forms))]},
                                       "files": [{"name": f} for f in files]}}


def test_filings_of_reads_older_pages_only_when_the_recent_ones_cannot_settle_it():
    older = {"form": ["S-1"], "filingDate": ["2019-01-01"], "accessionNumber": ["old"], "primaryDocument": ["o.htm"]}
    sec = FakeSec({"CIK0000000042.json": sub(["424B4"], ["2020-03-01"], files=["older-1.json"]), "older-1.json": older})
    rows, sic = D.filings_of(sec, "42", before="2020-03-01")      # recent page starts at the prospectus: settled
    assert rows == [("424B4", "2020-03-01", "acc-0", "d0.htm")] and sic == "7372" and len(sec.urls) == 1
    rows, _ = D.filings_of(sec, "42")                                # no cut-off: every page
    assert rows[0] == ("S-1", "2019-01-01", "old", "o.htm")
    sec = FakeSec({"CIK0000000042.json": sub(["424B4"], ["2020-05-01"])})
    rows, _ = D.filings_of(sec, "42", before="2020-03-01")          # starts after: would read older pages, has none
    assert len(rows) == 1
    sec = FakeSec({"CIK0000000042.json": {"filings": {}}})
    assert D.filings_of(sec, "42", before="2020-03-01") == ([], "")   # nothing at all, no SIC


def test_yahoo_chart_returns_the_result_none_when_missing_and_retries_failures(monkeypatch):
    script = []

    def urlopen(req, timeout):
        assert "BRK-B" in req.full_url and "events=split" in req.full_url
        x = script.pop(0)
        if isinstance(x, Exception):
            raise x
        return io.BytesIO(json.dumps(x).encode())
    monkeypatch.setattr(D.urllib.request, "urlopen", urlopen)
    script[:] = [{"chart": {"result": [{"timestamp": [1]}]}}]
    assert D.yahoo_chart("BRK.B", "2020-03-01") == {"timestamp": [1]}
    script[:] = [urllib.error.HTTPError("u", 404, "x", {}, None)]
    assert D.yahoo_chart("BRK.B", "2020-03-01") is None
    script[:] = [urllib.error.HTTPError("u", 500, "x", {}, None), urllib.error.URLError("down"), {"chart": {"result": []}}]
    assert D.yahoo_chart("BRK.B", "2020-03-01") is None and script == []


# ----------------------------------------------------------------------------- one row

HIT = {"adsh": "0000000042-20-000001", "cik": "42", "name": "Newco Inc", "ticker": "OLDT", "file": "p.htm",
       "file_date": "2020-03-01", "sic": "7372"}


def row_sec(forms, dates, sic="7372", heads=None):
    s = sub(forms, dates)
    s["sic"] = sic
    return FakeSec({"CIK0000000042.json": s}, heads)


def test_build_row_leaves_out_spacs_follow_ons_and_later_prospectuses():
    assert D.build_row(FakeSec(), {**HIT, "sic": "6770"}) is None
    assert D.build_row(FakeSec(), {**HIT, "name": "Big Acquisition Corp"}) is None
    assert D.build_row(row_sec(["424B4"], ["2020-03-01"], sic="6770"), HIT) is None
    assert D.build_row(row_sec(["10-K", "424B4"], ["2019-03-01", "2020-03-01"]), HIT) is None
    assert D.build_row(row_sec(["424B4", "424B4"], ["2019-03-01", "2020-03-01"]), HIT) is None


def test_build_row_reads_the_cover_the_latest_range_and_the_first_symbol_yahoo_has(monkeypatch):
    forms = ["F-1", "S-1/A", "S-1/A", "424B4"]
    dates = ["2020-01-01", "2020-02-01", "2020-02-20", "2020-03-01"]
    heads = {"/000000004220000001/p.htm": PROSPECTUS, "/acc2/d2.htm": b"no range in this one", "/acc1/d1.htm": REGISTRATION}
    asked = []

    def yahoo(symbol, around):
        asked.append(symbol)
        return chart("2020-03-02", 30) if symbol == "OLDT" else (chart("2015-01-01", 5) if symbol == "NEWC" else None)
    monkeypatch.setattr(D, "yahoo_chart", yahoo)
    row = D.build_row(row_sec(forms, dates, heads=heads), HIT)
    assert asked == ["NEWC", "OLDT"]                       # the prospectus's own symbol first; its history is elsewhere
    assert row["range"] == [9.0, 11.0] and row["range_date"] == "2020-02-01" and row["sources"]["range"] == "2020-02-01"
    assert row["offer_price"] == 10.0 and row["shares"] == 5_000_000 and row["foreign"] is True
    assert row["listing_symbol"] == "NEWC" and row["price_symbol"] == "OLDT" and row["prices"]["close"] == 12.0
    monkeypatch.setattr(D, "yahoo_chart", lambda s, a: None)
    heads["/acc1/d1.htm"] = heads["/acc0/d0.htm"] = b"nothing"
    row = D.build_row(row_sec(forms, dates, heads=heads), HIT)
    assert row["prices"] is None and "price_symbol" not in row and row["range"] is None


# ----------------------------------------------------------------------------- the build, the refresh, the cache

@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path / "ipo" / "ipos.jsonl")
    monkeypatch.setattr(D, "INITIAL", tmp_path / "ipo" / "initial_ranges.json")
    monkeypatch.setattr(D, "Sec", lambda: FakeSec())
    return tmp_path


def test_build_resumes_skipping_settled_filings_and_records_errors(paths, monkeypatch, capsys):
    D.OUT.parent.mkdir(parents=True)
    D.OUT.write_text("\n".join(json.dumps(x) for x in [
        {"adsh": "done-skip", "skip": "not an IPO"}, {"adsh": "retry-error", "skip": "error: boom"},
        {"adsh": "done-priced", "prices": {"close": 1}}, {"adsh": "done-tried", "listing_symbol": None},
        {"adsh": "retry-untried", "prices": None}]) + "\n\n", encoding="utf-8")
    monkeypatch.setattr(D, "FIRST_YEAR", dt.date.today().year)
    names = ["done-skip", "retry-error", "done-priced", "done-tried", "retry-untried"] + [f"new-{i}" for i in range(50)]
    monkeypatch.setattr(D, "prospectus_hits", lambda sec, year: [{"adsh": a} for a in names])

    def build_row(sec, hit):
        if hit["adsh"] == "retry-error":
            raise ValueError("bad filing")
        return None if hit["adsh"] == "new-0" else {"x": 1}
    monkeypatch.setattr(D, "build_row", build_row)
    D.build()
    lines = [json.loads(x) for x in D.OUT.read_text(encoding="utf-8").splitlines() if x.strip()][5:]
    assert [x["adsh"] for x in lines] == ["retry-error", "retry-untried"] + [f"new-{i}" for i in range(50)]
    assert lines[0]["skip"] == "error: bad filing" and lines[2]["skip"] == "not an IPO" and lines[3]["x"] == 1
    assert "52 prospectuses to read" in capsys.readouterr().out
    D.OUT.unlink()
    D.build(years=[2020])                                       # a first build, no file yet
    assert len(D.OUT.read_text(encoding="utf-8").splitlines()) == 55


def ipo(adsh, cik, date, **kw):
    return {"adsh": adsh, "cik": cik, "company": "c", "ticker": "T" + adsh, "prospectus_date": date, "file": "old.htm",
            "range": [9.0, 11.0], "prices": None, **kw}


def test_load_dedupes_companies_and_merges_initial_ranges(paths):
    D.OUT.parent.mkdir(parents=True)
    rows = [ipo("a", "1", "2020-02-01"), ipo("b", "1", "2020-02-01"), ipo("c", "2", "2019-01-01"),
            {"adsh": "s", "skip": "not an IPO"}, ipo("a", "1", "2020-02-01", offer_price=12.0)]
    D.OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert [r["adsh"] for r in D.load()] == ["c", "a"]
    D.INITIAL.write_text(json.dumps({"a": {"range": [8.0, 10.0], "date": "2019-12-01"}}), encoding="utf-8")
    every = D.load(dedupe=False)
    assert [r["adsh"] for r in every] == ["c", "a", "b"] and every[1]["offer_price"] == 12.0
    assert every[1]["initial_range"] == [8.0, 10.0] and "initial_range" not in every[0]


def test_refresh_rereads_covers_and_prices_keeping_the_range(paths, monkeypatch, capsys):
    D.OUT.parent.mkdir(parents=True)
    rows = [ipo("a", "42", "2020-03-01", price_symbol="X"), ipo("gone", "43", "2020-03-01"), ipo("bad", "44", "2021-03-01")]
    rows += [ipo(f"n{i}", str(100 + i), "2021-05-01") for i in range(97)]
    D.OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    hits = {"a": {"adsh": "0000000042-20-000001", "file": "p.htm", "ticker": "OLDT"},
            "bad": {"adsh": "0000000044-21-000001", "file": "missing.htm", "ticker": ""}}
    hits.update({f"n{i}": {"adsh": f"0000000099-21-{i:06d}", "file": "p.htm", "ticker": ""} for i in range(97)})
    monkeypatch.setattr(D, "prospectus_hits", lambda sec, year: [{**h, "adsh": a} for a, h in hits.items()])
    heads = {"/p.htm": PROSPECTUS}
    monkeypatch.setattr(D, "Sec", lambda: FakeSec(heads=heads))
    monkeypatch.setattr(D, "yahoo_chart", lambda s, a: chart("2020-03-02", 30) if s == "NEWC" else None)
    D.refresh()
    new = [json.loads(x) for x in D.OUT.read_text(encoding="utf-8").splitlines()][len(rows):]
    by = {r["adsh"]: r for r in new}
    assert "gone" not in by and len(new) == 99                    # no prospectus found for it: left as it was
    assert by["a"]["offer_price"] == 10.0 and by["a"]["range"] == [9.0, 11.0] and by["a"]["price_symbol"] == "NEWC"
    assert "no filing" in by["bad"]["refresh_error"]
    assert "refreshing 100 IPO rows" in capsys.readouterr().out


def test_add_initial_ranges_caches_the_earliest_stated_range(paths, monkeypatch, capsys):
    D.OUT.parent.mkdir(parents=True)
    rows = [ipo("a", "42", "2020-03-01"), ipo("b", "43", "2020-03-01"), ipo("c", "44", "2020-03-01"),
            ipo("norange", "45", "2020-03-01", range=None), ipo("cached", "46", "2020-03-01")]
    D.OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    D.INITIAL.write_text(json.dumps({"cached": {"range": None, "date": None}}), encoding="utf-8")
    pages = {"CIK0000000042.json": sub(["S-1", "S-1/A", "424B4"], ["2019-12-01", "2020-02-01", "2020-03-01"]),
             "CIK0000000043.json": sub(["S-1", "424B4"], ["2019-12-01", "2020-03-01"])}
    heads = {"/acc0/d0.htm": b"no range yet", "/acc1/d1.htm": REGISTRATION}
    monkeypatch.setattr(D, "Sec", lambda: FakeSec(pages, heads))
    D.add_initial_ranges()
    done = json.loads(D.INITIAL.read_text(encoding="utf-8"))
    assert done["a"] == {"range": [9.0, 11.0], "date": "2020-02-01"}
    assert done["b"] == {"range": None, "date": None}            # its only registration states none
    assert "c" not in done and "norange" not in done             # c's lookup failed: tried again next time
    assert "3 IPOs to look up" in capsys.readouterr().out
    D.INITIAL.unlink()
    monkeypatch.setattr(D, "load", lambda: [])
    D.add_initial_ranges()                                      # nothing to do: no cache written
    assert not D.INITIAL.exists()


def test_summary_and_the_command_line(paths, monkeypatch, capsys):
    D.OUT.parent.mkdir(parents=True)
    rows = [ipo("a", "1", "2020-02-01", prices={"close": 1}), ipo("b", "2", "2020-05-01", range=None)]
    D.OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    D.main(["--summary"])
    assert "2020     2     1 (50%)     1 (50%)" in capsys.readouterr().out
    called = []
    for name in ("refresh", "add_initial_ranges", "build"):
        monkeypatch.setattr(D, name, lambda name=name: called.append(name))
    D.main(["--refresh"])
    D.main(["--initial-ranges"])
    D.main([])
    assert called == ["refresh", "add_initial_ranges", "build"]


def test_the_module_imports_without_truststore(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)        # not installed: the import is skipped
    importlib.reload(D)
    monkeypatch.undo()                                          # the real one back, and the module as it was
    importlib.reload(D)
    assert D.OUT.name == "ipos.jsonl"
