"""Live checks against today's market: real SEC filings and real prices. Skipped unless you ask for them:

    IPO_BOT_LIVE=1 python -m pytest -q tests/src/test_live_market.py

They need SEC_USER_AGENT (in .env) and an internet connection; they cost nothing. Prices for the portfolio
tests come from Yahoo's public chart endpoint, used here only as a test fixture: the bot itself reads prices
from FMP_API_KEY's provider. The checks are about consistency (the parts add up, the rules hold), never about
a particular price, so they pass on any trading day.
"""
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import ipo_bot  # noqa: E402
import portfolio  # noqa: E402
import tools  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("IPO_BOT_LIVE") != "1", reason="live: set IPO_BOT_LIVE=1")

ipo_bot.load_env()
TODAY = dt.date.today()


def call(name, args):
    text, err = tools.run_tool(name, args)
    assert not err, text
    return json.loads(text)["data"]


def cik(ticker):
    hits = call("edgar_lookup", {"query": ticker})
    assert hits and hits[0]["ticker"] == ticker, hits
    return hits[0]["cik"]


def yahoo_price(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        meta = json.loads(r.read())["chart"]["result"][0]["meta"]
    return float(meta["regularMarketPrice"]), dt.datetime.fromtimestamp(meta["regularMarketTime"]).date()


# ----------------------------------------------------------------------------- SEC filings

@pytest.mark.parametrize("ticker, name", [("AAPL", "Apple"), ("JPM", "JPMorgan"), ("XOM", "Exxon")])
def test_listed_companies_have_a_current_periodic_report(ticker, name):
    data = call("edgar_filings", {"cik": cik(ticker), "forms": ["10-K", "10-Q"], "limit": 1})
    assert name.lower() in data["company"].lower()
    filed = dt.date.fromisoformat(data["filings"][0]["filed"])
    assert (TODAY - filed).days < 140, f"{ticker}'s latest 10-K/10-Q is from {filed}: a quarter has been missed"


def test_quarterly_revenue_is_the_difference_of_the_year_to_date_figures():
    # 10-Qs report both the quarter and the year to date; the bot must not mistake one for the other.
    rows = call("edgar_financials", {"cik": cik("AAPL"), "periods": 12,
                                     "concepts": ["RevenueFromContractWithCustomerExcludingAssessedTax"]})
    rows = next(iter(rows.values()))
    quarters = [r for r in rows if 80 <= r["period_days"] <= 100]
    ytd = {r["end"]: r for r in rows if r["period_days"] > 100}
    q = next(q for q in quarters if q["end"] in ytd and q["start"] != ytd[q["end"]]["start"])
    before = next(r for r in rows if r["start"] == ytd[q["end"]]["start"] and r["end"] < q["start"])
    assert ytd[q["end"]]["value"] - before["value"] == q["value"]


def test_a_filing_reads_as_text_without_the_xbrl_header():
    latest = call("edgar_filings", {"cik": cik("AAPL"), "forms": ["10-Q", "10-K"], "limit": 1})["filings"][0]
    doc = call("edgar_document", {"url": latest["url"], "part": 1})
    assert doc["of_parts"] > 1 and "Apple Inc." in doc["text"]
    assert "ix:header" not in doc["text"] and "xbrli:" not in doc["text"][:2000]


@pytest.mark.parametrize("ticker, listed", [("FIG", "2025-07"), ("CRCL", "2025-06"), ("CRWV", "2025-03")])
def test_ipo_prospectuses_are_found_even_after_years_of_insider_filings(ticker, listed):
    # CoreWeave's S-1 is no longer in the SEC's "recent" list (1,000+ later filings): it is on an older page.
    data = call("edgar_filings", {"cik": cik(ticker), "forms": ["S-1", "S-1/A", "424B4"], "limit": 20})
    forms = {f["form"] for f in data["filings"]}
    assert "424B4" in forms, f"{ticker}: no final prospectus found, only {forms}"
    final = min(f["filed"] for f in data["filings"] if f["form"] == "424B4")
    assert final.startswith(listed[:4]), final


def test_documents_outside_sec_gov_are_refused():
    text, err = tools.run_tool("edgar_document", {"url": "https://example.com/s-1.htm"})
    assert err and "sec.gov" in text


# ----------------------------------------------------------------------------- portfolio on real prices

HOLD = [("AAPL", 30, "Technology"), ("MSFT", 15, "Technology"), ("JPM", 25, "Financials"), ("XOM", 40, "Energy")]


@pytest.fixture(scope="module")
def real():
    prices = {s: yahoo_price(s) for s, _, _ in HOLD + [("NVDA", 0, "")]}
    raw = {"as_of": TODAY.isoformat(), "cash": 5000,
           "holdings": [{"symbol": s, "shares": n, "sector": sec, "price": prices[s][0],
                         "price_date": prices[s][1].isoformat()} for s, n, sec in HOLD],
           "rules": {"max_position_pct": 25, "max_sector_pct": 40, "risk_per_trade_pct": 1, "min_cash_pct": 5}}
    return portfolio.validate(raw), prices


def test_prices_are_current(real):
    _, prices = real
    for s, (p, day) in prices.items():
        assert p > 0 and (TODAY - day).days <= 5, f"{s}: {p} on {day}"


def test_a_real_portfolio_adds_up_and_flags_its_breaches(real):
    pf, prices = real
    v = portfolio.valued(pf)
    assert v["equity"] == pytest.approx(5000 + sum(n * prices[s][0] for s, n, _ in HOLD))
    assert sum(h["weight_pct"] for h in v["holdings"]) + v["cash_pct"] == pytest.approx(100)
    for h in v["holdings"]:
        over = h["weight_pct"] > 25
        assert over == any(b.get("symbol") == h["symbol"] for b in v["rule_breaches"]), h
    tech = v["sector_pct"]["Technology"]
    assert (tech > 40) == any(b.get("sector") == "Technology" for b in v["rule_breaches"])


@pytest.mark.parametrize("conviction", ["Low", "Medium", "High"])
def test_a_real_purchase_obeys_every_rule(real, conviction):
    pf, prices = real
    v = portfolio.valued(pf)
    entry = prices["NVDA"][0]
    s = portfolio.size_position(v, "NVDA", entry, round(entry * 0.85, 2), conviction, "Technology")
    eq = v["equity"]
    assert s["loss_at_stop"] <= s["risk_budget"] + 1e-6
    assert s["weight_after_pct"] <= 25 + 1e-6
    assert v["sector_pct"]["Technology"] + 100 * s["cost"] / eq <= 40 + 1e-6 or s["shares"] == 0
    assert s["cash_after"] >= eq * 0.05 - 1e-6
    assert s["shares"] == min(s["limits_in_shares"].values())


def test_the_tools_size_from_the_file_on_real_prices(real, tmp_path, monkeypatch):
    pf, prices = real
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps({**pf, "holdings": pf["holdings"]}), encoding="utf-8")
    monkeypatch.setitem(tools.PORTFOLIO, "path", str(path))
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    view = call("portfolio_view", {})
    entry = prices["XOM"][0]
    size = call("portfolio_size", {"symbol": "XOM", "entry_price": entry, "stop_price": round(entry * 0.9, 2),
                                   "conviction": "Medium"})
    direct = portfolio.size_position(portfolio.valued(pf), "XOM", entry, round(entry * 0.9, 2), "Medium")
    assert view["equity"] == pytest.approx(portfolio.valued(pf)["equity"])
    assert size["shares"] == direct["shares"] and size["binding_rule"] == direct["binding_rule"]
