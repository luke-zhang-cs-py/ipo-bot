"""A filing's document (the prospectus or registration statement), read up to its first few hundred kilobytes
(the cover page and summary), and what the prospectus parsers find in it."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from bot import prospectus

NAME = "sec_doc"
TERMS = "SEC public data; fair access: under 10 requests a second, a User-Agent with a contact"
LIMIT = 300_000


def url(cik: str, accession: str, file: str, **_: Any) -> str:
    if file.endswith(".txt") and file.startswith(accession):  # the daily index names the whole submission
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{file}"
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{file}"


def parse(body: bytes, form: str, marketed: Optional[Sequence[float]] = None, **_: Any) -> Dict[str, Any]:
    """The facts a form carries: a registration's price range (and shares), a final prospectus's offer price,
    shares, lead bookrunner and listing symbol. A fact the text does not state is None. `marketed`, the range
    from the last registration, keeps a fee on the 424B4's cover from being read as the offer price."""
    text = prospectus.text_of(body)
    rng = prospectus.price_range(text)
    out: Dict[str, Optional[Any]] = {
        "range": list(rng) if rng else None,
        "shares": prospectus.shares_offered(text),
        "lead": prospectus.lead_bank(text),
        "symbol": prospectus.listing_symbol(text),
    }
    if form == "424B4":
        out["offer"] = prospectus.offer_price(text, marketed or rng)
    return out
