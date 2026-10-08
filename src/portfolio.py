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
    Guardrails, each off unless set:
    "max_daily_loss_pct": 2,         no new buys once today's loss ("day_pnl") reaches this share of the portfolio
    "earnings_blackout_hours": 48,   no buys this close before the stock's next earnings release
    "max_volume_pct": 1,             no order above this share of the stock's average daily volume
    "max_order_value": 500,          no single order worth more than this
    "max_gross_exposure_pct": 90,    no new buys once this share of the portfolio is invested
    "max_drawdown_pct": 8            no new buys once the portfolio is this far below equity_high: review first
  },
  "day_pnl": -150.0,                 optional: today's gain or loss so far, for max_daily_loss_pct
  "equity_high": 12500,              optional: the portfolio's highest value, for max_drawdown_pct
  "halted": false                    the kill switch: true stops every new order until set back to false
}
"""
import datetime as dt
import json
import math
import pathlib

DEFAULT_RULES = {"max_position_pct": 10.0, "max_sector_pct": 30.0, "risk_per_trade_pct": 1.0,
                 "min_cash_pct": 5.0, "conviction_scale": {"Low": 0.5, "Medium": 1.0, "High": 1.5}}
GUARDRAILS = {"max_daily_loss_pct": 100.0, "earnings_blackout_hours": 24 * 30.0, "max_volume_pct": 100.0,   # each one's ceiling
              "max_order_value": 1e12, "max_gross_exposure_pct": 100.0, "max_drawdown_pct": 100.0}


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
    for k, ceiling in GUARDRAILS.items():
        if rules.get(k) is not None:
            rules[k] = _num(rules[k], f"rules.{k}")
            if rules[k] > ceiling:
                raise PortfolioError(f"rules.{k} is at most {ceiling:g}")
    day_pnl = raw.get("day_pnl", 0)
    if isinstance(day_pnl, bool) or not isinstance(day_pnl, (int, float)) or math.isnan(day_pnl) or math.isinf(day_pnl):
        raise PortfolioError("day_pnl must be a number (negative for a loss)")
    equity_high = raw.get("equity_high")
    if equity_high is not None:
        equity_high = _num(equity_high, "equity_high")
    halted = raw.get("halted", False)
    if not isinstance(halted, bool):
        raise PortfolioError("halted must be true or false")
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
            "holdings": holdings, "rules": rules, "day_pnl": float(day_pnl), "equity_high": equity_high, "halted": halted}


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
            "unpriced": unpriced, "rule_breaches": breaches, "rules": rules, "day_pnl": pf.get("day_pnl", 0.0),
            "equity_high": pf.get("equity_high"), "halted": pf.get("halted", False), "invested": invested}


def _when(v, what):
    """An ISO date or date-time as an aware UTC datetime; a bare date means the start of that day."""
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        d = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError as e:
        raise PortfolioError(f"{what} must be an ISO date or date-time") from e
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def size_position(view, symbol, entry_price, stop_price, conviction="Medium", sector=None, *,
                  next_earnings=None, avg_volume=None, now=None):
    """Shares to buy under the user's rules. Every limit is computed and the smallest wins:
    - risk:     risk budget (equity x risk_per_trade_pct x conviction scale) / (entry - stop)
    - position: room left under max_position_pct, after what is already held
    - sector:   room left under max_sector_pct for the stock's sector
    - cash:     cash above min_cash_pct
    and the guardrails the file sets:
    - daily_loss:        0 once today's loss (day_pnl) reaches max_daily_loss_pct of the portfolio
    - earnings_blackout: 0 within earnings_blackout_hours before next_earnings
    - liquidity:         max_volume_pct of avg_volume (the stock's average daily volume)
    - order_value:       max_order_value / entry
    - exposure:          room left under max_gross_exposure_pct for everything invested
    - drawdown:          0 once equity is max_drawdown_pct below equity_high
    - kill_switch:       0 while the file says "halted": true
    A guardrail that is set but can't be checked (no earnings date, no volume) is named in "warnings".
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
    if sym in view.get("unpriced", []):
        raise PortfolioError(f"{sym} is held but has no price: its position and sector limits cannot be checked")
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
    warnings = []
    if rules.get("max_daily_loss_pct") is not None:
        lost = -min(0.0, view.get("day_pnl", 0.0))
        if equity and lost >= equity * rules["max_daily_loss_pct"] / 100 - 1e-9:
            limits["daily_loss"] = 0
    if rules.get("earnings_blackout_hours") is not None:
        if next_earnings is None:
            warnings.append("earnings_blackout_hours is set but no earnings date was given: the blackout was not checked")
        else:
            hours = (_when(next_earnings, "next_earnings") - (_when(now, "now") if now else dt.datetime.now(dt.timezone.utc))).total_seconds() / 3600
            if 0 <= hours <= rules["earnings_blackout_hours"]:
                limits["earnings_blackout"] = 0
    if rules.get("max_volume_pct") is not None:
        if avg_volume is None:
            warnings.append("max_volume_pct is set but no average volume was given: the order size was not checked against it")
        else:
            limits["liquidity"] = math.floor(_num(avg_volume, "avg_volume") * rules["max_volume_pct"] / 100)
    if rules.get("max_order_value") is not None:
        limits["order_value"] = math.floor(rules["max_order_value"] / entry)
    if rules.get("max_gross_exposure_pct") is not None:
        invested = view.get("invested", sum(h["value"] for h in view["holdings"]))
        limits["exposure"] = math.floor(max(0.0, equity * rules["max_gross_exposure_pct"] / 100 - invested) / entry)
    if rules.get("max_drawdown_pct") is not None:
        high = view.get("equity_high")
        if high is None:
            warnings.append("max_drawdown_pct is set but equity_high is not: the drawdown was not checked")
        elif equity <= high * (1 - rules["max_drawdown_pct"] / 100) + 1e-9:
            limits["drawdown"] = 0
    if view.get("halted"):
        limits["kill_switch"] = 0
    shares = max(0, min(limits.values()))
    binding = min(limits, key=limits.get)
    cost = shares * entry
    new_value = held_value + cost
    return {
        "symbol": sym, "shares": shares, "cost": round(cost, 2), "entry_price": entry, "stop_price": stop,
        "conviction": conviction, "binding_rule": binding, "limits_in_shares": limits,
        "risk_budget": round(risk_budget, 2), "loss_at_stop": round(shares * (entry - stop), 2),
        "weight_after_pct": round(100 * new_value / equity, 2) if equity else 0.0,
        "cash_after": round(view["cash"] - cost, 2), "warnings": warnings,
        "math": (f"risk: {equity:.2f} x {rules['risk_per_trade_pct']}% x {scale} = {risk_budget:.2f} / "
                 f"({entry} - {stop}) -> {limits['risk']} shares; the smallest limit ({binding}) wins: {shares} shares"),
    }
