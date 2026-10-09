"""The bot's data tools (src/tools.py), offline: HTTP is faked at urlopen or at tools._get; files live in tmp_path."""
import io
import json
import pathlib
import sys
import urllib.error

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import tools  # noqa: E402

UA = "Test Person test@example.com"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setitem(tools.AS_OF, "date", None)
    monkeypatch.setitem(tools.PORTFOLIO, "path", None)
    for k in ("SEC_USER_AGENT", "FRED_API_KEY", "FMP_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def fake_get(monkeypatch, pages):
    """tools._get answering from {url substring: object}; the calls are kept."""
    calls = []

    def get(url, headers=None):
        calls.append((url, headers))
        for part, obj in pages.items():
            if part in url:
                return obj if isinstance(obj, str) else json.dumps(obj)
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(tools, "_get", get)
    return calls


# ----------------------------------------------------------------------------- HTTP and results

class Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_get_reads_utf8_and_names_the_host_on_each_kind_of_failure(monkeypatch):
    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda req, timeout: Resp("café".encode()))
    assert tools._get("https://example.com/x", {"A": "b"}) == "café"

    def http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))
    monkeypatch.setattr(tools.urllib.request, "urlopen", http_error)
    with pytest.raises(tools.ToolError, match="HTTP 404 from example.com"):
        tools._get("https://example.com/x")

    def url_error(req, timeout):
        raise urllib.error.URLError("dns failure")
    monkeypatch.setattr(tools.urllib.request, "urlopen", url_error)
    with pytest.raises(tools.ToolError, match="could not reach example.com: dns failure"):
        tools._get("https://example.com/x")


def test_a_small_result_is_returned_whole_with_its_note():
    out = json.loads(tools._result("src", {"a": 1}, "a note"))
    assert out["data"] == {"a": 1} and out["note"] == "a note" and "truncated" not in out


def test_a_long_list_result_is_cut_to_valid_json_and_says_so(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 2_000)
    rows = [{"date": f"2026-01-{i % 28 + 1:02d}", "close": i} for i in range(500)]
    text = tools._result("prices", rows, "first note")
    out = json.loads(text)                                             # valid JSON, not cut mid-string
    assert len(text) <= 2_000 and out["truncated"] is True
    assert out["note"] == "first note; truncated: ask for a narrower request"
    assert 0 < len(out["data"]) < 500 and out["data"] == rows[:len(out["data"])]   # the first rows, whole


def test_long_filing_text_is_cut_from_its_end_and_short_fields_stay_whole(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 1_000)
    out = json.loads(tools._result("doc", {"part": 1, "of_parts": 3, "text": "x" * 5_000, "when": tools.dt.date(2026, 1, 2)}))
    assert out["data"]["part"] == 1 and out["data"]["of_parts"] == 3 and out["data"]["when"] == "2026-01-02"
    assert 0 < len(out["data"]["text"]) < 1_000 and out["note"] == "truncated: ask for a narrower request"


def test_a_large_flat_object_keeps_the_entries_that_fit(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 500)
    flat = {f"k{i}": i for i in range(200)}
    text = tools._result("flat", flat)
    out = json.loads(text)
    assert len(text) <= 500 and out["truncated"] is True
    assert 10 < len(out["data"]) < 200 and all(flat[k] == v for k, v in out["data"].items())


def test_data_is_dropped_when_the_source_alone_is_too_long(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 500)
    out = json.loads(tools._result("s" * 600, {"a": 1}))
    assert out["data"] is None and out["truncated"] is True


def test_a_wide_object_of_long_strings_is_trimmed_in_a_few_passes(monkeypatch):
    wide = {f"item{i}": "y" * 300 for i in range(2_000)}
    calls = []
    trim = tools._trim
    monkeypatch.setattr(tools, "_trim", lambda data, excess: calls.append(1) or trim(data, excess))
    text = tools._result("wide", wide)
    out = json.loads(text)
    assert len(text) <= tools.MAX_RESULT_CHARS and out["truncated"] is True and 0 < len(out["data"]) < 2_000
    assert all(wide[k].startswith(v) and len(v) >= tools.LONG_STRING for k, v in out["data"].items())
    assert len(calls) <= 3 * (len(wide) + 1)          # each pass visits every value once: not one value per pass


def test_escaped_text_is_cut_by_its_json_length_not_its_characters():
    text = '"' * 1_000                                    # 2,002 characters of JSON
    cut = tools._cut(text, 500)
    assert tools._size(text) - 506 <= tools._size(cut) <= tools._size(text) - 500   # about 500 off, not 1,000
    dense = "" * 300 + "a" * 300                     # escapes first: a proportional cut keeps too many
    cut = tools._cut(dense, 300)
    assert dense.startswith(cut) and tools._size(cut) <= tools._size(dense) - 300 and len(cut) >= tools.LONG_STRING
    assert tools._cut(text, 0) == text


# ----------------------------------------------------------------------------- EDGAR

def test_edgar_needs_a_user_agent_with_an_email_and_a_real_cik(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "no email here")
    with pytest.raises(tools.ToolError, match="SEC_USER_AGENT"):
        tools._sec_headers()
    for bad in ("", "abc", "12345678901"):
        with pytest.raises(tools.ToolError, match="not a CIK"):
            tools._cik10(bad)
    assert tools._cik10("CIK 320193") == "0000320193"


def test_edgar_lookup_puts_the_exact_ticker_first_and_says_when_nothing_matched(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    rows = {"0": {"cik_str": 1, "ticker": "ABCX", "title": "Abc Holdings"},
            "1": {"cik_str": 2, "ticker": "ABC", "title": "Other Corp"},
            "2": {"cik_str": 3, "ticker": "ZZZ", "title": "Abc Zed"}}
    calls = fake_get(monkeypatch, {"company_tickers": rows})
    out = json.loads(tools.edgar_lookup(" abc "))
    assert [d["ticker"] for d in out["data"]] == ["ABC", "ABCX", "ZZZ"] and "note" not in out
    assert calls[0][1]["User-Agent"] == UA
    assert "no listed company matched" in json.loads(tools.edgar_lookup("nothing"))["note"]
    with pytest.raises(tools.ToolError, match="give a ticker"):
        tools.edgar_lookup("  ")


def submissions(forms, dates, prefix="0000000001-26-"):
    return {"form": forms, "filingDate": dates, "accessionNumber": [f"{prefix}{i:06d}" for i in range(len(forms))],
            "primaryDocument": [f"d{i}.htm" for i in range(len(forms))]}


def test_edgar_filings_reads_older_pages_for_a_form_search_and_keeps_the_backtest_date(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    recent = submissions(["4", "4", "10-Q"], ["2026-09-01", "2026-08-01", "2026-07-01"])
    older = submissions(["S-1", "S-1/A", "S-1"], ["2025-01-01", "2025-02-01", "2025-03-01"], "0000000001-25-")
    fake_get(monkeypatch, {"CIK0000000001.json": {"name": "Co", "tickers": ["CO"],
                                                 "filings": {"recent": recent, "files": [{"name": "p1.json"}, {"name": "p2.json"}]}},
                           "p1.json": older, "p2.json": "not read: the limit is reached first"})
    out = json.loads(tools.edgar_filings("1", forms=["s-1"], limit=2))["data"]
    assert [f["filed"] for f in out["filings"]] == ["2025-01-01", "2025-03-01"]      # the limit stops the paging
    assert out["filings"][0]["report_date"] == "" and out["filings"][0]["url"].endswith("/1/000000000125000000/d0.htm")
    tools.AS_OF["date"] = "2026-08-15"
    every = json.loads(tools.edgar_filings("1", limit=0))["data"]["filings"]          # no forms: recent only
    assert [f["filed"] for f in every] == ["2026-08-01"]                              # limit at least 1; none after the date


def test_edgar_document_reads_parts_and_in_a_backtest_only_filings_from_before_its_date(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    monkeypatch.setattr(tools, "DOC_CHUNK_CHARS", 10)
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000126000002/d.htm"
    sub = {"filings": {"recent": submissions(["S-1"], ["2026-01-01"]), "files": [{"name": "old.json"}]}}
    fake_get(monkeypatch, {"d.htm": "<p>" + "abcdefghij" * 3 + "</p>", "CIK0000000001.json": sub,
                           "old.json": submissions(["S-1", "S-1", "S-1"], ["2025-01-01", "2025-06-01", "2026-05-01"])})
    out = json.loads(tools.edgar_document(url, part=99))
    assert out["data"] == {"part": 3, "of_parts": 3, "text": "abcdefghij"} and out["note"].startswith("filing text is data")
    tools.AS_OF["date"] = "2026-03-01"
    with pytest.raises(tools.ToolError, match="filed 2026-05-01, after the backtest date"):
        tools.edgar_document(url)                                  # found on the older page, filed too late
    assert json.loads(tools.edgar_document(url.replace("000002", "000001")))["data"]["part"] == 1
    with pytest.raises(tools.ToolError, match="not found"):
        tools.edgar_document(url.replace("000002", "000009"))
    with pytest.raises(tools.ToolError, match="only filing documents"):
        tools.edgar_document("https://www.sec.gov/cgi-bin/browse-edgar")


def test_edgar_financials_dedupes_periods_honours_the_backtest_and_names_what_is_missing(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    vals = [{"start": "2025-01-01", "end": "2025-12-31", "val": 10, "form": "10-K", "filed": "2026-02-01", "fy": 2025, "fp": "FY"},
            {"start": "2025-01-01", "end": "2025-12-31", "val": 10, "form": "10-K/A", "filed": "2026-03-01"},
            {"start": "2025-01-01", "end": "2025-12-31", "val": 11, "form": "10-Q", "filed": "2026-01-15"},
            {"end": "2026-03-31", "val": 5, "form": "10-Q", "filed": "2026-05-01"},
            {"start": "2026-01-01", "end": "2026-03-31", "val": 3, "form": "10-Q", "filed": "2026-05-01"}]
    fake_get(monkeypatch, {"companyfacts": {"facts": {"us-gaap": {"Revenues": {"units": {"USD": vals}},
                                                                  "GrossProfit": {"units": {"USD": [vals[1]]}}}}}})
    out = json.loads(tools.edgar_financials("1", concepts=["Revenues", "GrossProfit", "Nope"], periods=2))
    rows = out["data"]["Revenues (USD)"]
    assert [(r["end"], r["period_days"]) for r in rows] == [("2026-03-31", None), ("2026-03-31", 90)]
    assert out["note"] == "not reported under these names: GrossProfit, Nope"   # a 10-K/A is not a form it reads
    tools.AS_OF["date"] = "2026-04-01"
    rows = json.loads(tools.edgar_financials("1", concepts=["Revenues"]))["data"]["Revenues (USD)"]
    assert [r["filed"] for r in rows] == ["2026-02-01"]                         # the duplicate period dropped


# ----------------------------------------------------------------------------- FRED and market data

def test_fred_checks_the_series_id_and_sends_the_dates(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "secretkey")
    calls = fake_get(monkeypatch, {"fred/series": {"observations": [{"date": "2026-01-01", "value": "1"}]}})
    with pytest.raises(tools.ToolError, match="not a FRED series id"):
        tools.fred_series("DGS10; drop")
    tools.AS_OF["date"] = "2026-06-01"
    out = json.loads(tools.fred_series("dgs10", start="2025-01-01", limit=9999))
    assert "observation_start=2025-01-01" in calls[0][0] and "observation_end=2026-06-01" in calls[0][0]
    assert "limit=500" in calls[0][0] and "secretkey" not in out["source"]


def test_market_data_in_a_backtest_serves_only_past_prices(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "fmpkey")
    calls = fake_get(monkeypatch, {"financialmodelingprep": [{"close": 1}]})
    tools.AS_OF["date"] = "2026-06-01"
    with pytest.raises(tools.ToolError, match="current data"):
        tools.market_data("quote", {"symbol": "ABC"})
    tools.market_data("price_history", {"symbol": "ABC", "from": "2026-01-01", "to": "2026-12-31", "junk": "x"})
    assert "to=2026-06-01" in calls[-1][0] and "junk" not in calls[-1][0]
    tools.market_data("price_history", {"symbol": "ABC", "from": ""})
    assert "to=2026-06-01" in calls[-1][0] and "from=" not in calls[-1][0]
    with pytest.raises(tools.ToolError, match="'from' is after"):
        tools.market_data("price_history", {"symbol": "ABC", "from": "2026-07-01"})


# ----------------------------------------------------------------------------- the calculator

@pytest.mark.parametrize("expr, variables, error", [
    ("2 ** 1000", None, "exponent too large"),
    ("(10 ** 100) ** 100", None, "result too large"),
    ("price * 2", None, "unknown name 'price'"),
    ("1 / 0", None, "division by zero"),
    ("exp(1000)", None, "result too large for a number"),
    ("1 +", None, "not an expression"),
    ("1" * 2001, None, "expression too long"),
    ("x", {"x": True}, "must be a name with a number value"),
    ("x", {"1x": 1}, "must be a name with a number value"),
    ("round(x, ndigits=2)", {"x": 1.5}, "only numbers"),
    ("'a'", None, "only numbers"),
])
def test_the_calculator_refuses_what_it_should(expr, variables, error):
    with pytest.raises(tools.ToolError, match=error):
        tools.calculate(expr, variables)


def test_the_calculator_does_functions_and_unary_minus():
    out = json.loads(tools.calculate("-max(a, 2) + sqrt(16) + sum(1, 2)", {"a": 5}))
    assert out["result"] == 2.0 and out["variables"] == {"a": 5}


# ----------------------------------------------------------------------------- portfolio mode

def write_pf(tmp_path, **extra):
    pf = {"as_of": "2026-10-01", "cash": 10_000, "rules": {"max_position_pct": 50, "max_sector_pct": 100},
          "holdings": [{"symbol": "AAA", "shares": 10, "price": 100.0, "sector": "Tech"}], **extra}
    path = tmp_path / "pf.json"
    path.write_text(json.dumps(pf), encoding="utf-8")
    tools.PORTFOLIO["path"] = str(path)
    return path


def test_portfolio_tools_need_a_loaded_file_and_report_its_errors(tmp_path):
    with pytest.raises(tools.ToolError, match="no portfolio loaded"):
        tools.portfolio_view()
    path = write_pf(tmp_path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(tools.ToolError, match="portfolio file: .*not valid JSON"):
        tools.portfolio_view()


def test_portfolio_view_prices_live_when_it_can_and_names_the_unpriced(tmp_path, monkeypatch):
    write_pf(tmp_path, holdings=[{"symbol": "AAA", "shares": 10, "price": 100.0}, {"symbol": "BBB", "shares": 1}])
    out = json.loads(tools.portfolio_view())
    assert out["note"] == "no price for BBB: left out of the totals" and out["source"].startswith("portfolio file pf.json")
    monkeypatch.setenv("FMP_API_KEY", "k")
    answers = {"AAA": json.dumps({"retrieved_at": "T", "data": [{"price": 120}]}), "BBB": json.dumps({"data": []})}
    monkeypatch.setattr(tools, "market_data", lambda ep, params: answers[params["symbol"]])
    view = json.loads(tools.portfolio_view())["data"]
    assert view["holdings"][0]["price"] == 120.0 and view["holdings"][0]["price_source"] == "market data quote, retrieved T"


@pytest.mark.parametrize("answer", ['{"data": [{"price": 0}]}', '{"data": [{"price": "12"}]}', "not json", None])
def test_a_live_quote_that_is_unusable_falls_back_to_none(monkeypatch, answer):
    monkeypatch.setenv("FMP_API_KEY", "k")

    def md(ep, params):
        if answer is None:
            raise tools.ToolError("HTTP 500")
        return answer
    monkeypatch.setattr(tools, "market_data", md)
    assert tools._live_quote("AAA") is None


def test_a_truncated_quote_result_is_still_readable(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 400)
    monkeypatch.setattr(tools, "_get", lambda url, headers=None: json.dumps([{"price": 42.5, "pad": "y" * 50}] * 40))
    assert tools._live_quote("AAA")[0] == 42.5                     # the cut keeps the first row whole


def test_portfolio_size_uses_the_rules_and_turns_their_errors_into_tool_errors(tmp_path):
    write_pf(tmp_path)
    sized = json.loads(tools.portfolio_size("NEW", 50.0, 45.0, "Medium", sector="Energy"))["data"]
    assert sized["shares"] > 0 and sized["binding_rule"]
    with pytest.raises(tools.ToolError, match="below the entry"):
        tools.portfolio_size("NEW", 50.0, 55.0, "Medium")


# ----------------------------------------------------------------------------- the forecast ledger

GOOD = {"symbol": "abc", "rating": "Overweight", "expected_return_pct": 18.0, "price": 10.0, "price_date": "2026-10-01"}


@pytest.mark.parametrize("change, error", [
    ({"symbol": "not a ticker"}, "not a ticker"),
    ({"rating": "Buy"}, "rating must be one of"),
    ({"price": float("nan")}, "price must be a number"),
    ({"expected_return_pct": True}, "expected_return_pct must be a number"),
    ({"price": 0}, "price must be above 0"),
    ({"conviction": "Huge"}, "conviction must be one of"),
    ({"bull_value": -1}, "bull_value must be a number, 0 or more"),
    ({"base_prob": 101}, "base_prob is a percentage"),
    ({"bull_value": 5, "bear_value": 6}, "bear_value must not be above bull_value"),
    ({"price_date": "Oct 1"}, "price_date must be YYYY-MM-DD"),
])
def test_record_forecast_refuses_bad_entries(change, error, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEDGER", tmp_path / "ledger.jsonl")
    with pytest.raises(tools.ToolError, match=error):
        tools.record_forecast(**{**GOOD, **change})
    assert not (tmp_path / "ledger.jsonl").exists()


def test_record_forecast_logs_a_backtest_entry_with_its_date(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEDGER", tmp_path / "ledger.jsonl")
    tools.AS_OF["date"] = "2026-01-05"
    tools.record_forecast(**GOOD, bull_value=20, bull_prob=25, conviction="High")
    entry = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert entry["date"] == "2026-01-05" and entry["backtest_as_of"] == "2026-01-05" and entry["symbol"] == "ABC"
    assert entry["bull_value"] == 20.0 and entry["conviction"] == "High" and "bear_value" not in entry


# ----------------------------------------------------------------------------- dispatch

def test_validate_and_run_tool_never_raise():
    assert tools.validate("nope", {}) == "unknown tool 'nope'"
    assert tools.validate("calculate", []) == "input must be an object"
    assert tools.validate("calculate", {"expression": "1", "extra": 2}) == "unexpected ['extra']"
    assert tools.run_tool("calculate", {}) == ("Error: missing ['expression']", True)
    assert tools.run_tool("calculate", {"expression": "1 +"})[1] is True
    text, err = tools.run_tool("edgar_filings", {"cik": "1", "limit": "many"})
    assert err and text.startswith("Error:")                       # no SEC_USER_AGENT -> a ToolError first
    assert tools.run_tool("calculate", {"expression": "x", "variables": "x=1"}) == (
        "Error: variables must be an object of names and numbers", True)          # was an AttributeError that escaped


def test_a_single_form_or_concept_given_as_a_string_is_one_name_not_its_letters(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    fake_get(monkeypatch, {"CIK0000000001.json": {"filings": {"recent": submissions(["S", "S-1"], ["2026-01-01", "2026-01-02"])}},
                           "companyfacts": {"facts": {}}})
    assert [f["form"] for f in json.loads(tools.edgar_filings("1", forms="S-1"))["data"]["filings"]] == ["S-1"]
    assert json.loads(tools.edgar_financials("1", concepts="Revenues"))["note"] == "not reported under these names: Revenues"


def test_a_python_error_inside_a_tool_comes_back_as_an_error_result():
    text, err = tools.run_tool("calculate", {"expression": "sqrt(-1)"})
    assert err and text.startswith("Error: ValueError:")
    text, err = tools.run_tool("calculate", {"expression": "round(1, 2, 3)"})
    assert err and text.startswith("Error: TypeError:")
