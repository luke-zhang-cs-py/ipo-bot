"""The NYSE list behind the demo's picker (docs/app/nyse.json) and the SIC-to-sector map that builds it."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_nyse  # noqa: E402

DATA = json.loads((ROOT / "docs" / "app" / "nyse.json").read_text(encoding="utf-8"))


def test_sic_codes_map_to_sectors():
    cases = {"2834": "Health Care", 3841: "Health Care", 8062: "Health Care", 7372: "Technology", 3674: "Technology",
             6021: "Financials", 6798: "Real Estate", 6512: "Real Estate", 1311: "Energy", 2911: "Energy", 4911: "Utilities",
             4953: "Industrials", 3711: "Consumer Discretionary", 3721: "Industrials", 2080: "Consumer Staples",
             2844: "Consumer Staples", 5411: "Consumer Staples", 5331: "Consumer Discretionary", 4813: "Communication",
             2711: "Communication", 2800: "Materials", 1040: "Materials", 9995: "Other", None: "Other", "": "Other"}
    for sic, want in cases.items():
        assert build_nyse.sic_sector(sic) == want, (sic, build_nyse.sic_sector(sic), want)


def test_industry_names_are_unescaped_and_readable():
    assert build_nyse.industry("FIRE, MARINE &amp;amp; CASUALTY INSURANCE") == "Fire, Marine & Casualty Insurance"
    assert build_nyse.industry("SERVICES-PREPACKAGED SOFTWARE") == "Services-Prepackaged Software"
    assert build_nyse.industry(None) == ""


def test_every_sector_the_map_can_give_is_a_known_one():
    assert {s for _, _, s in build_nyse.SIC_RANGES} | {"Other"} <= set(build_nyse.SECTORS)
    for code in range(0, 10000, 7):
        assert build_nyse.sic_sector(code) in build_nyse.SECTORS


def test_a_failed_lookup_is_retried_next_run_not_cached():
    calls = {}

    def lookup(cik):
        calls[cik] = calls.get(cik, 0) + 1
        if cik == 2:
            raise OSError("throttled")
        return [str(6000 + cik), "SOME INDUSTRY"]

    cache = {}
    failed = build_nyse.look_up_all([1, 2, 3], lookup, cache, interval=0, workers=2, retries=3, backoff=0)
    assert failed == [2] and set(cache) == {"1", "3"} and calls[2] == 3       # three tries, then left for the next run
    assert build_nyse.look_up_all([2], lambda cik: ["6798", "REIT"], cache, interval=0, backoff=0) == []
    assert cache["2"] == ["6798", "REIT"]


def test_rows_are_sorted_with_sector_and_readable_industry():
    fields = {"cik": 0, "name": 1, "ticker": 2}
    rows = build_nyse.rows_for([[7, "Zed Co", "ZED"], [5, "Abc Inc", "ABC"], [9, "No Sic", "NOS"]], fields,
                               {"7": ["4911", "ELECTRIC SERVICES"], "5": ["7372", "SERVICES-PREPACKAGED SOFTWARE"]})
    assert rows == [["ABC", "Abc Inc", 5, "Technology", "Services-Prepackaged Software"], ["NOS", "No Sic", 9, "Other", ""],
                    ["ZED", "Zed Co", 7, "Utilities", "Electric Services"]]


def test_the_list_is_the_whole_nyse_and_well_formed():
    rows, f = DATA["rows"], {k: i for i, k in enumerate(DATA["fields"])}
    assert DATA["fields"] == ["ticker", "name", "cik", "sector", "industry"] and DATA["as_of"]
    assert len(rows) > 2500                                    # the SEC lists about 3,300 NYSE tickers
    tickers = [r[f["ticker"]] for r in rows]
    assert tickers == sorted(tickers)
    assert all(isinstance(r[f["cik"]], int) and r[f["cik"]] > 0 and r[f["name"]] for r in rows)
    assert {r[f["sector"]] for r in rows} <= set(build_nyse.SECTORS)
    with_sector = sum(r[f["sector"]] != "Other" for r in rows)
    assert with_sector > 0.8 * len(rows)                        # nearly every operating company has a SIC code
    assert not any("&amp;" in r[f["industry"]].lower() for r in rows)
    by = {r[f["ticker"]]: r for r in rows}
    for t, sector in (("JPM", "Financials"), ("LLY", "Health Care"), ("XOM", "Energy")):
        assert t in by and by[t][f["sector"]] == sector, (t, by.get(t))
