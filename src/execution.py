"""From a sized trade to filled shares: limit orders, partial fills, thin markets, broker errors.

portfolio.size_position decides how many shares the rules allow. This turns that into orders and copes with
what a broker does to them: fills part of an order, fills none (the price ran past the limit), or fails.
The guarantee: however many partial fills it takes, the shares bought never risk more than the trade's
risk budget at the stop-working price, and never break a rule, because every remainder is sized again
against the portfolio as it now stands (cash spent, shares held) and the risk already taken.

SimBroker is a broker for tests and paper runs without an account: a quote per symbol with a visible depth,
and a "spike" mode where the spread widens and the depth dries up, as at a busy open. paper.py also has an
adapter for Alpaca's paper-trading API with the same submit() shape.
"""
import math
import random
import time

import portfolio

MAX_ATTEMPTS = 3            # the first order and up to two for what's left
LIMIT_SLIPPAGE = 0.005      # a buy's limit sits 0.5% over the planned entry; above that, the order waits


class BrokerError(Exception):
    """The broker refused the order or failed to answer."""


class SimBroker:
    """A simulated broker. quotes: {symbol: {"bid", "ask", "depth"}} where depth is the shares available at the
    ask before the price moves. impact: the fraction the price rises as an order eats through the depth.
    fail_rate: the chance an order fails outright (an API error). spike(): spreads x5 and depth / 10."""

    def __init__(self, quotes, impact=0.01, fail_rate=0.0, seed=7, latency_s=0.0, fee_rate=0.0):
        self.quotes = {k.upper(): dict(v) for k, v in quotes.items()}
        self.impact, self.fail_rate, self.latency_s, self.fee_rate = impact, fail_rate, latency_s, fee_rate
        self.rng = random.Random(seed)
        self.orders = []

    def spike(self, spread_x=5.0, depth_div=10.0):
        for q in self.quotes.values():
            mid = (q["bid"] + q["ask"]) / 2
            half = (q["ask"] - q["bid"]) / 2 * spread_x
            q["bid"], q["ask"], q["depth"] = mid - half, mid + half, max(1, int(q["depth"] / depth_div))

    def submit(self, symbol, shares, limit):
        """A buy limit order: returns {"filled", "avg_price", "status"}; the unfilled part is cancelled."""
        if self.latency_s:
            time.sleep(self.latency_s)
        self.orders.append((symbol, shares, limit))
        order_id = f"sim-{len(self.orders)}"
        if self.rng.random() < self.fail_rate:
            raise BrokerError("simulated API error: 503 from the order endpoint")
        q = self.quotes.get(symbol.upper())
        if q is None:
            raise BrokerError(f"no quote for {symbol}")
        if q["ask"] > limit or q["depth"] <= 0:
            return {"order_id": order_id, "filled": 0, "avg_price": None, "fee": 0.0,
                    "status": "unfilled: the ask is above the limit" if q["ask"] > limit else "unfilled: no depth"}
        # fill in slices up the book until the limit or the depth runs out
        filled, cost, price = 0, 0.0, q["ask"]
        step = max(1, q["depth"] // 20)
        while filled < shares and q["depth"] > 0 and price <= limit:
            take = min(step, shares - filled, q["depth"])
            filled += take
            cost += take * price
            q["depth"] -= take
            price = q["ask"] * (1 + self.impact * filled / max(1, filled + q["depth"]))
        q["ask"] = max(q["ask"], price)
        status = "filled" if filled == shares else "partial"
        return {"order_id": order_id, "filled": filled, "avg_price": cost / filled if filled else None,
                "fee": round(cost * self.fee_rate, 4), "status": status}


class BarBroker:
    """Fills from a day's bar instead of a quote, for replays on daily data. A buy limit fills at the open when the
    open is at or under the limit; otherwise only if the day's low trades through it (strictly below, as a
    resting order behind others in the queue needs: after NautilusTrader's and backtrader's bar fill rules,
    re-implemented). Each fill is capped at volume_limit of the day's volume, with zipline's quadratic impact
    price_impact x (share of volume)^2. bars: {symbol: (open, high, low, close, volume)} for the day."""

    def __init__(self, bars, volume_limit=0.025, price_impact=0.1, fee_rate=0.0):
        self.bars = {k.upper(): v for k, v in bars.items()}
        self.volume_limit, self.price_impact, self.fee_rate = volume_limit, price_impact, fee_rate
        self.used = {}                                     # shares already filled today, per symbol
        self.orders = []

    def submit(self, symbol, shares, limit):
        sym = symbol.upper()
        self.orders.append((sym, shares, limit))
        order_id = f"bar-{len(self.orders)}"
        if sym not in self.bars:
            raise BrokerError(f"no bar for {symbol}")
        o, h, lo, c, vol = self.bars[sym]
        if o <= limit:
            price = o
        elif lo < limit:
            price = limit
        else:
            return {"order_id": order_id, "filled": 0, "avg_price": None, "fee": 0.0,
                    "status": "unfilled: the price never traded through the limit"}
        room = max(0, int(self.volume_limit * vol) - self.used.get(sym, 0))
        filled = min(int(shares), room)
        if filled <= 0:
            return {"order_id": order_id, "filled": 0, "avg_price": None, "fee": 0.0, "status": "unfilled: the day's volume share is used up"}
        self.used[sym] = self.used.get(sym, 0) + filled
        price = min(limit, price * (1 + self.price_impact * (filled / vol) ** 2)) if vol else price
        return {"order_id": order_id, "filled": filled, "avg_price": price, "fee": round(filled * price * self.fee_rate, 4),
                "status": "filled" if filled == int(shares) else "partial"}


def execute_buy(view, signal, broker, attempts=MAX_ATTEMPTS, limit_slippage=LIMIT_SLIPPAGE):
    """Size a buy under the rules and work it at the broker. signal: {symbol, entry, stop, conviction, sector?,
    next_earnings?, avg_volume?, now?}. Returns what happened, order by order:
    {"planned", "filled", "avg_price", "loss_at_stop", "risk_budget", "status", "blocked_by", "orders": [...]}."""
    sym = str(signal["symbol"]).upper()
    extra = {k: signal[k] for k in ("next_earnings", "avg_volume", "now") if signal.get(k) is not None}
    plan = portfolio.size_position(view, sym, signal["entry"], signal["stop"], signal.get("conviction", "Medium"),
                                   signal.get("sector"), **extra)
    out = {"symbol": sym, "planned": plan["shares"], "filled": 0, "avg_price": None, "loss_at_stop": 0.0, "fees": 0.0,
           "risk_budget": plan["risk_budget"], "warnings": plan["warnings"], "orders": [], "blocked_by": None}
    if plan["shares"] == 0:
        out.update(status="blocked", blocked_by=plan["binding_rule"])
        return out
    limit = round(signal["entry"] * (1 + limit_slippage), 4)
    stop = float(signal["stop"])
    want, spent, risk_used = plan["shares"], 0.0, 0.0
    current = view
    for _ in range(attempts):
        try:
            fill = broker.submit(sym, want, limit)
        except BrokerError as e:
            out["orders"].append({"shares": want, "limit": limit, "status": f"error: {e}"})
            out["status"] = "api error" if out["filled"] == 0 else "partial, then an api error"
            break
        out["orders"].append({"shares": want, "limit": limit, **fill})
        out["fees"] += fill.get("fee", 0.0)
        if fill["filled"]:
            spent += fill["filled"] * fill["avg_price"]
            risk_used += fill["filled"] * (fill["avg_price"] - stop)
            out["filled"] += fill["filled"]
            current = after_fill(current, sym, fill["filled"], fill["avg_price"], signal.get("sector"))
        if fill["status"] == "filled" or out["filled"] >= plan["shares"]:
            break
        if fill["filled"] == 0:
            break                                        # nothing traded at the limit: waiting is the user's call
        # the rest, sized again: the rules on the portfolio as it now is, and the risk budget less what's used
        again = portfolio.size_position(current, sym, signal["entry"], stop, signal.get("conviction", "Medium"),
                                        signal.get("sector"), **extra)
        risk_room = math.floor(max(0.0, plan["risk_budget"] - risk_used) / (limit - stop))
        want = max(0, min(plan["shares"] - out["filled"], again["shares"], risk_room))
        if want == 0:
            break
    out["avg_price"] = round(spent / out["filled"], 4) if out["filled"] else None
    out["loss_at_stop"] = round(risk_used, 2)
    if "status" not in out:
        out["status"] = ("filled" if out["filled"] == plan["shares"] else "partial" if out["filled"]
                         else out["orders"][-1]["status"])
    out["stop_order"] = {"symbol": sym, "shares": out["filled"], "stop": stop} if out["filled"] else None
    return out


def after_fill(view, symbol, shares, price, sector):
    """The portfolio view after buying `shares` at `price`: cash down, the holding up, weights recomputed."""
    holdings = [dict(h) for h in view["holdings"]]
    held = next((h for h in holdings if h["symbol"] == symbol), None)
    if held:
        held["shares"] += shares
    else:
        holdings.append({"symbol": symbol, "shares": shares, "cost_basis": price, "sector": sector or "Unknown",
                         "price": price, "price_date": view["as_of"]})
    pf = {"as_of": view["as_of"], "cash": view["cash"] - shares * price, "rules": view["rules"],
          "day_pnl": view.get("day_pnl", 0.0),
          "holdings": [{k: h[k] for k in ("symbol", "shares", "cost_basis", "sector", "price", "price_date")} for h in holdings]}
    return portfolio.valued(pf)
