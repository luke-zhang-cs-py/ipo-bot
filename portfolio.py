"""Portfolio mode: your holdings and your own sizing rules, and the arithmetic that turns a view into a trade size.

The model decides what it thinks of a stock. How many shares that means is not its call: size_position works it
out from the rules in your portfolio file, deterministically, and says which rule set the limit. The model is
told to use that number and nothing else.

The file (see portfolio.example.json; your real one, portfolio.json, is git-ignored):
{
  "as_of": "YYYY-MM-DD",            the date the holdings and any prices in it are true
  "cash": 10000,
  "holdings": [{"symbol": "ABC", "shares": 10, "cost_basis": 95.0, "sector": "Technology",
                "price": 101.2, "price_date": "YYYY-MM-DD"}],     price is optional when market data is set up
  "rules": {
    "max_position_pct": 10,          no single stock above this share of the portfolio
    "max_sector_pct": 30,            no sector above this share
    "risk_per_trade_pct": 1,         the most you accept losing on one idea if it reaches its stop-working price
    "min_cash_pct": 5,               cash you always keep
    "conviction_scale": {"Low": 0.5, "Medium": 1, "High": 1.5}   multiplies the risk per trade
  }
}
"""
import json
import math
import pathlib

DEFAULT_RULES = {"max_position_pct": 10.0, "max_sector_pct": 30.0, "risk_per_trade_pct": 1.0,
                 "min_cash_pct": 5.0, "conviction_scale": {"Low": 0.5, "Medium": 1.0, "High": 1.5}}


class PortfolioError(ValueError):
    pass


def _num(v, what, minimum=0.0, allow_zero=True):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
        raise PortfolioError(f"{what} must be a number")
    if v < minimum or (not allow_zero and v == 0):
        raise PortfolioError(f"{what} must be {'above' if not allow_zero else 'at least'} {minimum}")
    return float(v)


def load(path):
    try:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise PortfolioError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise PortfolioError(f"{path} is not valid JSON (line {e.lineno})") from e
    return validate(raw)


def validate(raw):
    if not isinstance(raw, dict):
        raise PortfolioError("the portfolio file must be a JSON object")
    rules = {**DEFAULT_RULES, **(raw.get("rules") or {})}
    for k in ("max_position_pct", "max_sector_pct", "risk_per_trade_pct", "min_cash_pct"):
        rules[k] = _num(rules[k], f"rules.{k}")
        if rules[k] > 100:
            raise PortfolioError(f"rules.{k} is a percentage: at most 100")
    scale = rules.get("conviction_scale") or {}
    rules["conviction_scale"] = {str(k): _num(v, f"conviction_scale.{k}") for k, v in scale.items()}
    holdings, seen = [], set()
    for i, h in enumerate(raw.get("holdings") or []):
        if not isinstance(h, dict) or not str(h.get("symbol", "")).strip():
            raise PortfolioError(f"holding {i + 1} needs a symbol")
        sym = str(h["symbol"]).strip().upper()
        if sym in seen:
            raise PortfolioError(f"{sym} is listed twice")
        seen.add(sym)
        holdings.append({
            "symbol": sym,
            "shares": _num(h.get("shares", 0), f"{sym} shares"),
            "cost_basis": _num(h["cost_basis"], f"{sym} cost_basis") if h.get("cost_basis") is not None else None,
            "sector": str(h.get("sector") or "Unknown"),
            "price": _num(h["price"], f"{sym} price", allow_zero=False) if h.get("price") is not None else None,
            "price_date": str(h.get("price_date") or raw.get("as_of") or ""),
        })
    return {"as_of": str(raw.get("as_of") or ""), "cash": _num(raw.get("cash", 0), "cash"),
            "holdings": holdings, "rules": rules}


def valued(pf, quote=None):
    """Holdings with a price, value and weight. `quote(symbol)` -> (price, source) or None, for live prices;
    without it, or when it has none, the file's own price is used. A holding with no price at all is
    listed as unpriced and left out of the totals, and the result says so."""
    rows, unpriced = [], []
    for h in pf["holdings"]:
        live = quote(h["symbol"]) if quote else None
        if live:
            price, source = live
        elif h["price"] is not None:
            price, source = h["price"], f"portfolio file, {h['price_date'] or 'undated'}"
        else:
            unpriced.append(h["symbol"])
            continue
        rows.append({**h, "price": price, "price_source": source, "value": h["shares"] * price})
    invested = sum(r["value"] for r in rows)
    equity = invested + pf["cash"]
    sectors = {}
    for r in rows:
        r["weight_pct"] = 100 * r["value"] / equity if equity else 0.0
        if r["cost_basis"]:
            r["gain_pct"] = round(100 * (r["price"] / r["cost_basis"] - 1), 2)
        sectors[r["sector"]] = sectors.get(r["sector"], 0.0) + r["value"]
    rules = pf["rules"]
    breaches = []
    for r in rows:
        if r["weight_pct"] > rules["max_position_pct"] + 1e-9:
            excess = r["value"] - equity * rules["max_position_pct"] / 100
            breaches.append({"rule": "max_position_pct", "symbol": r["symbol"], "weight_pct": round(r["weight_pct"], 2),
                             "shares_over": math.ceil(excess / r["price"])})
    for s, v in sectors.items():
        pct = 100 * v / equity if equity else 0.0
        if pct > rules["max_sector_pct"] + 1e-9:
            breaches.append({"rule": "max_sector_pct", "sector": s, "weight_pct": round(pct, 2)})
    cash_pct = 100 * pf["cash"] / equity if equity else 0.0
    if cash_pct < rules["min_cash_pct"] - 1e-9:
        breaches.append({"rule": "min_cash_pct", "cash_pct": round(cash_pct, 2)})
    return {"as_of": pf["as_of"], "equity": equity, "cash": pf["cash"], "cash_pct": cash_pct, "holdings": rows,
            "sector_pct": {s: 100 * v / equity if equity else 0.0 for s, v in sectors.items()},
            "unpriced": unpriced, "rule_breaches": breaches, "rules": rules}


def size_position(view, symbol, entry_price, stop_price, conviction="Medium", sector=None):
    """Shares to buy under the user's rules. Every limit is computed and the smallest wins:
    - risk:     risk budget (equity x risk_per_trade_pct x conviction scale) / (entry - stop)
    - position: room left under max_position_pct, after what is already held
    - sector:   room left under max_sector_pct for the stock's sector
    - cash:     cash above min_cash_pct
    """
    entry = _num(entry_price, "entry_price", allow_zero=False)
    stop = _num(stop_price, "stop_price")
    if stop >= entry:
        raise PortfolioError("the stop-working price must be below the entry price for a purchase")
    rules, equity = view["rules"], view["equity"]
    scale = rules["conviction_scale"].get(str(conviction))
    if scale is None:
        raise PortfolioError(f"conviction must be one of {sorted(rules['conviction_scale'])}")
    sym = str(symbol).strip().upper()
    held = next((h for h in view["holdings"] if h["symbol"] == sym), None)
    sector = sector or (held["sector"] if held else "Unknown")
    held_value = held["value"] if held else 0.0
    sector_value = view["sector_pct"].get(sector, 0.0) * equity / 100

    risk_budget = equity * rules["risk_per_trade_pct"] / 100 * scale
    limits = {
        "risk": math.floor(risk_budget / (entry - stop)),
        "position": math.floor(max(0.0, equity * rules["max_position_pct"] / 100 - held_value) / entry),
        "cash": math.floor(max(0.0, view["cash"] - equity * rules["min_cash_pct"] / 100) / entry),
    }
    if sector != "Unknown":
        limits["sector"] = math.floor(max(0.0, equity * rules["max_sector_pct"] / 100 - sector_value) / entry)
    shares = max(0, min(limits.values()))
    binding = min(limits, key=limits.get)
    cost = shares * entry
    new_value = held_value + cost
    return {
        "symbol": sym, "shares": shares, "cost": round(cost, 2), "entry_price": entry, "stop_price": stop,
        "conviction": conviction, "binding_rule": binding, "limits_in_shares": limits,
        "risk_budget": round(risk_budget, 2), "loss_at_stop": round(shares * (entry - stop), 2),
        "weight_after_pct": round(100 * new_value / equity, 2) if equity else 0.0,
        "cash_after": round(view["cash"] - cost, 2),
        "math": (f"risk: {equity:.2f} x {rules['risk_per_trade_pct']}% x {scale} = {risk_budget:.2f} / "
                 f"({entry} - {stop}) -> {limits['risk']} shares; the smallest limit ({binding}) wins: {shares} shares"),
    }
