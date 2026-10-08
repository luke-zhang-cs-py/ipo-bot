"""Every adapter on a real response recorded from its source, and on malformed, empty and changed ones."""

import datetime as dt
import gzip
import json
import logging

import pytest
from conftest import Opener, http, make_cfg, recorded

from bot.adapters import (
    base,
    cboe,
    cboe_vix,
    fred,
    nasdaqtrader,
    sec_company,
    sec_doc,
    sec_index,
    sec_search,
    treasury,
    wikipedia,
    yahoo,
)
from bot.adapters.base import Result, SchemaError, conform, first_ok, read
from bot.http import SourceError

D = dt.date


def test_yahoo_bars_splits_dividends() -> None:
    bars, extra = yahoo.parse(recorded("yahoo_AAPL_2024-06"), symbol="AAPL")
    assert len(bars) == 21 and bars[0]["date"] == "2024-06-03" and bars[0]["close"] == pytest.approx(194.03, abs=0.01)
    assert extra["actions"] == []
    _, extra = yahoo.parse(recorded("yahoo_NVDA_split"), symbol="NVDA")
    assert {"symbol": "NVDA", "date": "2024-06-10", "kind": "split", "value": 10.0} in extra["actions"]
    _, extra = yahoo.parse(recorded("yahoo_KO_dividend"), symbol="KO")
    assert any(a["kind"] == "dividend" and a["value"] == pytest.approx(0.485) for a in extra["actions"])
    with pytest.raises(SourceError) as e:
        yahoo.parse(recorded("yahoo_not_found"), symbol="ZZZZQX")
    assert e.value.kind == "not_found"


def test_yahoo_urls_and_odd_responses() -> None:
    assert "BRK-B?" in yahoo.url("BRK.B", D(2024, 1, 2), D(2024, 1, 3)) and "%5EGSPC" in yahoo.url(
        "SPX", D(2024, 1, 2), D(2024, 1, 3)
    )
    with pytest.raises(SchemaError):
        yahoo.parse(b'{"x": 1}', symbol="A")
    with pytest.raises(SchemaError):
        yahoo.parse(b'{"chart": {"result": [], "error": null}}', symbol="A")
    with pytest.raises(SourceError) as e:
        yahoo.parse(b'{"chart": {"result": null, "error": "Bad Request"}}', symbol="A")
    assert e.value.kind == "http"
    with pytest.raises(SchemaError):
        yahoo.parse(
            b'{"chart": {"result": [{"meta": {}, "timestamp": [1], "indicators": {"quote": [{"close": [1]}]}}]}}',
            symbol="A",
        )
    empty, extra = yahoo.parse(b'{"chart": {"result": [{"meta": {"gmtoffset": -18000}}], "error": null}}', symbol="A")
    assert empty == [] and extra == {"actions": []}
    body = {
        "chart": {
            "result": [
                {
                    "meta": {},
                    "timestamp": [1704205800, 1704292200],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1, None],
                                "high": [2, None],
                                "low": [1, None],
                                "close": [1.5, None],
                                "volume": [10, None],
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars, _ = yahoo.parse(json.dumps(body).encode(), symbol="A")
    assert [b["date"] for b in bars] == ["2024-01-02"]  # the day with no close is left missing, not invented


def test_cboe() -> None:
    bars = cboe.parse(recorded("cboe_AAPL"), symbol="AAPL")
    assert len(bars) == 60 and bars[-1]["date"] == "2026-10-07"
    some = cboe.parse(recorded("cboe_AAPL"), symbol="AAPL", start=D(2026, 10, 1), end=D(2026, 10, 6))
    assert some[0]["date"] == "2026-10-01" and some[-1]["date"] == "2026-10-06"
    spx = cboe.parse(recorded("cboe_SPX"), symbol="SPX")
    assert spx[-1]["open"] is None or spx[-1]["open"] > 0
    assert cboe.url("SPX").endswith("/_SPX.json")
    with pytest.raises(SchemaError):
        cboe.parse(b"[1, 2]", symbol="A")
    zero = json.dumps(
        {
            "data": [
                {"date": "2024-01-02", "open": "0", "high": "", "low": None, "close": "0", "volume": "1"},
                {"date": "2024-01-03", "open": "0", "close": "5"},
            ]
        }
    ).encode()
    assert [b["open"] for b in cboe.parse(zero, symbol="A")] == [None]


def test_vix_treasury_fred() -> None:
    vix = cboe_vix.parse(recorded("cboe_vix"))
    assert len(vix) == 300 and vix[-1] == {"series": "VIX", "date": "2026-10-07", "value": 15.08}
    with pytest.raises(SchemaError):
        cboe_vix.parse(b"Date,Value\n")
    ust = treasury.parse(recorded("treasury_2026"))
    assert {r["series"] for r in ust} == {"UST3M", "UST2Y", "UST10Y"} and ust[0]["date"] == "2026-01-02"
    assert "2026/all" in treasury.url(year=2026)
    with pytest.raises(SchemaError):
        treasury.parse(b'"Date","1 Mo"\n01/02/2026,4\n')
    body = b"observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n"
    assert fred.parse(body, series="UST10Y") == [{"series": "UST10Y", "date": "2024-01-02", "value": 3.95}]
    with pytest.raises(SchemaError):
        fred.parse(b"DATE,VIXCLS\n", series="UST10Y")
    with pytest.raises(SchemaError):
        fred.parse(b"", series="VIX")
    assert fred.url(series="VIX").endswith("VIXCLS")


def test_nasdaqtrader() -> None:
    nas = nasdaqtrader.parse(recorded("nasdaqlisted"), which="nasdaq")
    other = nasdaqtrader.parse(recorded("otherlisted"), which="other")
    syms = {r["symbol"] for r in nas + other}
    assert {"AAPL", "MSFT", "BRK.B", "KO"} <= syms and "SPY" not in syms and "QQQ" not in syms  # ETFs left out
    assert next(r for r in other if r["symbol"] == "KO")["exchange"] == "NYSE"
    with pytest.raises(SchemaError):
        nasdaqtrader.parse(b"", which="nasdaq")
    with pytest.raises(SchemaError):
        nasdaqtrader.parse(b"Ticker|Name\nA|B\nFile Creation Time: x", which="nasdaq")
    head = recorded("nasdaqlisted").split(b"\r\n")[0]
    with pytest.raises(SchemaError, match="cut short"):
        nasdaqtrader.parse(head + b"\r\nAAPL|Apple|Q|N|N|100|N|N\r\n", which="nasdaq")
    assert nasdaqtrader.url(which="other").endswith("otherlisted.txt")


def test_wikipedia() -> None:
    rows = wikipedia.parse(recorded("wikipedia_sp500"))
    assert len(rows) >= 500
    brk = next(r for r in rows if r["symbol"] == "BRK.B")
    assert brk["cik"] == "1067983" and brk["added"] == "2010-02-16"
    with pytest.raises(SchemaError, match="no constituents"):
        wikipedia.parse(b'{"parse": {"text": "<p>x</p>"}}')
    with pytest.raises(SchemaError, match="columns"):
        wikipedia.parse(
            json.dumps({"parse": {"text": '<table id="constituents"><tr><th>Ticker</th></tr></table>'}}).encode()
        )
    with pytest.raises(SchemaError, match="only"):
        wikipedia.parse(recorded("wikipedia_sp500"), min_members=10_000)
    small = (
        '<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th>'
        "<th>CIK</th></tr><tr><td>A</td><td>B</td></tr><tr><td>X</td><td>Y</td><td></td><td></td><td>000</td></tr></table>"
    )
    assert wikipedia.parse(json.dumps({"parse": {"text": small}}).encode(), min_members=1) == [
        {"symbol": "X", "name": "Y", "cik": None, "sector": None, "added": None}
    ]


def test_sec_search_and_indexes() -> None:
    rows, extra = sec_search.parse(recorded("efts_2026-09-29"))
    assert extra["total"] == 11 and len(rows) == 11 and rows[0]["file"] == "forms-1a.htm"
    assert sec_search._company("Snap Inc.  (SNAP)  (CIK 0001564408)") == ("Snap Inc.", "SNAP")
    only, _ = sec_search.parse(recorded("efts_2026-09-29"), forms=("F-1/A",))
    assert {r["form"] for r in only} == {"F-1/A"}
    with pytest.raises(SchemaError):
        sec_search.parse(b'{"hits": []}')
    _, extra = sec_search.parse(b'{"hits": {"total": 3, "hits": []}}')
    assert extra["total"] == 3
    daily = sec_index.parse(recorded("daily_index_2026-09-30"))
    assert {r["form"] for r in daily} <= set(sec_search.FORMS) and any(r["form"] == "EFFECT" for r in daily)
    full = sec_index.parse(recorded("full_index_2025Q2"))  # gzipped, as EDGAR serves it
    assert len(full) == 400 and all(r["filed"][:4] == "2025" for r in full)
    with pytest.raises(SchemaError):
        sec_index.parse(b"<html>maintenance</html>")
    assert sec_index.url(day=D(2026, 9, 30)).endswith("QTR3/form.20260930.idx")
    assert sec_index.url(year=2025, qtr=2).endswith("full-index/2025/QTR2/form.gz")


def test_sec_company() -> None:
    info = sec_company.parse(recorded("submissions_2113605"), cik="2113605")
    assert info["first_report"] is None and info["pages"] == [] and info["name"]
    nvda = sec_company.parse(recorded("submissions_nvda_trimmed"), cik="1045810")
    assert nvda["first_report"] is not None and nvda["sic"] == "3674"
    page = sec_company.parse(b'{"form": ["10-K"], "filingDate": ["2001-01-01"]}', cik="1", page="CIK1-001.json")
    assert page == {"cik": "1", "first_report": "2001-01-01", "oldest": "2001-01-01"}
    assert sec_company.url("5", page="x.json").endswith("/x.json") and sec_company.url("5").endswith(
        "CIK0000000005.json"
    )
    with pytest.raises(SchemaError):
        sec_company.parse(b"{}", cik="1")
    assert sec_company.parse(b'{"form": [], "filingDate": []}', cik="1", page="p")["oldest"] is None


def test_sec_documents() -> None:
    s1a = sec_doc.parse(recorded("s1a_with_range"), form="S-1/A")
    assert s1a["range"] == [4.0, 5.0] and s1a["shares"] == 4_000_000 and "offer" not in s1a
    final = sec_doc.parse(recorded("424b4_newgenivf"), form="424B4")
    assert final["offer"] == 1.0 and final["symbol"] == "NIVF"
    assert sec_doc.parse(recorded("424b4_newgenivf"), form="424B4", marketed=[50, 60])["offer"] is None
    assert sec_doc.url("1981662", "0001213900-26-104917", "0001213900-26-104917.txt").endswith(
        "/1981662/0001213900-26-104917.txt"
    )
    assert sec_doc.url("5", "0000000005-26-000001", "a.htm").endswith("/5/000000000526000001/a.htm")


def test_conform() -> None:
    assert conform([{"series": "X", "date": "d", "value": 3}], "macro")[0]["value"] == 3.0
    for bad in (
        {"series": "X", "date": "d"},
        {"series": "X", "date": "d", "value": "3"},
        {"series": "X", "date": "d", "value": float("nan")},
        {"series": "X", "date": "d", "value": True},
    ):
        with pytest.raises(SchemaError):
            conform([bad], "macro")
    assert (
        conform(
            [{"symbol": "A", "date": "d", "close": 1.0, "open": None, "high": None, "low": None, "volume": None}], "bar"
        )[0]["open"]
        is None
    )


def test_read_and_fallback(tmp_path, caplog) -> None:
    cfg = make_cfg(tmp_path)
    vix_csv = recorded("cboe_vix")
    ok = http(cfg, Opener({"VIX_History": vix_csv, "fredgraph": b"observation_date,VIXCLS\n2024-01-02,13\n"}))
    res = first_ok(ok, [(cboe_vix, {}), (fred, {"series": "VIX"})])
    assert res.ok and not res.degraded and res.source == "cboe_vix"
    down = http(
        cfg,
        Opener({"VIX_History": SourceError("http", "500"), "fredgraph": b"observation_date,VIXCLS\n2024-01-02,13\n"}),
    )
    log = logging.getLogger("bot.test.fallback")
    with caplog.at_level(logging.WARNING):
        res = first_ok(down, [(cboe_vix, {}), (fred, {"series": "VIX"})], log, what="VIX")
    assert res.degraded and res.source == "fred" and res.failures[0][0] == "cboe_vix"
    assert any("fallback_used" in r.getMessage() for r in caplog.records)
    dead = http(cfg, Opener({"cdn.cboe.com": b"not,a,vix\n", "fredgraph": b"<html>"}))
    res = first_ok(dead, [(cboe_vix, {}), (fred, {"series": "VIX"})], log)
    assert not res.ok and len(res.failures) == 2 and all("schema" in e for _, e in res.failures)
    res = first_ok(dead, [(cboe_vix, {})])
    assert not res.ok
    broken = http(cfg, Opener({"query1": b"not json"}))
    res = read(broken, yahoo, symbol="A", start=D(2024, 1, 2), end=D(2024, 1, 3))
    assert not res.ok and res.failures[0][1].startswith("schema:")
    assert not Result().ok and not Result().degraded


def test_gzip_fixture_is_double_wrapped() -> None:
    # the recorded quarterly index is EDGAR's own gzip file; the test fixture wraps it once more
    assert (
        recorded("full_index_2025Q2")[:2] == b"\x1f\x8b"
        and gzip.decompress(recorded("full_index_2025Q2"))[:11] == b"Description"
    )


def test_every_adapter_names_its_terms() -> None:
    for a in (
        cboe,
        cboe_vix,
        fred,
        nasdaqtrader,
        sec_company,
        sec_doc,
        sec_index,
        sec_search,
        treasury,
        wikipedia,
        yahoo,
    ):
        assert a.NAME and a.TERMS
    assert base.SCHEMAS.keys() >= {"bar", "macro", "filing"}
