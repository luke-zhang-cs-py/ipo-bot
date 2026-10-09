"""Paper trading: signals through the risk rules to a broker, every step logged, and where the two diverge.

    python src/paper.py run signals.json --portfolio portfolio.json            simulated broker, no account
    python src/paper.py run signals.json --portfolio portfolio.json --alpaca   Alpaca's paper-trading API
    python src/paper.py report                                                 signals against executions

signals.json: [{"time": ISO, "symbol", "entry", "stop", "conviction", "sector"?, "next_earnings"?, "avg_volume"?}]
(the bot's portfolio-mode buys, or any strategy's). Each run appends to forecasts/paper/log.jsonl: the signal,
what the rules allowed, each order sent, each fill, and the time from signal to fill.

Before the signals, a run asks the broker which of the log's open positions (its fills, less the exits already
logged) have closed since: SimBroker when a quote's bid reaches the stop, Alpaca when the position is gone and
filled sells since the buy say at what price. Each one gets an exit line ({"exit": {"day", "symbol", "shares",
"price", "stopped"}}) and goes to the monitor, so its cooldown and stoploss guard see stop-outs from this run and
every earlier one. Every line records its broker ("sim", "alpaca", ...): simulated runs and --alpaca runs share
the log, a broker is asked only about the positions it opened, and the monitor learns only that broker's exits.
A line from before brokers were recorded counts as "sim" (only simulated runs came before). A position Alpaca no
longer holds with no filled sell to show for it gets a note line with an "unpriced_exit" ({"symbol"}): it is
closed for later runs but, having no price, not an exit the monitor learns. A broker that fails to answer gets a
warning line.

The forward test the brief asks for is two weeks or more of this against Alpaca's paper account: set
ALPACA_KEY_ID and ALPACA_SECRET_KEY in .env (a free paper account; no real money moves) and run it daily.
The report then shows the signal-to-execution divergence: signals the rules blocked (and which rule),
orders unfilled or partly filled, API errors, the slippage from the signal's price, and the latency.
"""
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import execution
import monitor as monitor_mod
import portfolio
from common import HERE

LOG = HERE / "forecasts" / "paper" / "log.jsonl"
PAPER_HOST = "paper-api.alpaca.markets"
ALPACA_PAPER = "https://paper-api.alpaca.markets"
STOP_TYPES = ("stop", "stop_limit", "trailing_stop")   # Alpaca order types that are a stop being hit
SELL_ORDERS_LIMIT = 100       # closed sells fetched per symbol since its buy (Alpaca allows up to 500 a page)


class AlpacaPaper:
    """Alpaca's paper-trading API with SimBroker's submit() shape: a day limit order, polled until it is done
    or `wait_s` passes, then whatever is left is cancelled. http(method, url, body) -> dict is injectable for tests."""

    name = "alpaca"

    def __init__(self, key_id=None, secret=None, base=ALPACA_PAPER, wait_s=30.0, http=None):
        self.key_id = key_id or os.environ.get("ALPACA_KEY_ID")
        self.secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not (self.key_id and self.secret):
            raise execution.BrokerError("set ALPACA_KEY_ID and ALPACA_SECRET_KEY (a paper account) in .env")
        parts = urllib.parse.urlsplit(base)
        if parts.scheme != "https" or parts.hostname != PAPER_HOST:   # the keys go only to Alpaca's paper host
            raise execution.BrokerError("only Alpaca's paper endpoint is allowed here: no real orders")
        self.base, self.wait_s = base.rstrip("/"), wait_s
        self.http = http or self._http

    def _http(self, method, url, body=None):
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                text = r.read().decode()
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as e:
            raise execution.BrokerError(f"Alpaca {e.code}: {e.read().decode(errors='replace')[:200]}") from e
        except urllib.error.URLError as e:
            raise execution.BrokerError(f"could not reach Alpaca: {e.reason}") from e

    def submit(self, symbol, shares, limit):
        order = self.http("POST", f"{self.base}/v2/orders", {"symbol": symbol, "qty": str(int(shares)), "side": "buy",
                                                              "type": "limit", "limit_price": f"{limit:.2f}", "time_in_force": "day"})
        oid, deadline = order.get("id"), time.monotonic() + self.wait_s
        while order.get("status") not in ("filled", "canceled", "expired", "rejected") and time.monotonic() < deadline:
            time.sleep(1.0)
            order = self.http("GET", f"{self.base}/v2/orders/{oid}")
        if order.get("status") not in ("filled", "canceled", "expired", "rejected"):
            self.http("DELETE", f"{self.base}/v2/orders/{oid}")
            order = self.http("GET", f"{self.base}/v2/orders/{oid}")
        if order.get("status") == "rejected":
            raise execution.BrokerError(f"rejected: {order.get('reject_reason') or 'no reason given'}")
        filled = int(float(order.get("filled_qty") or 0))
        avg = float(order["filled_avg_price"]) if order.get("filled_avg_price") else None
        status = "filled" if filled == int(shares) else "partial" if filled else "unfilled: " + str(order.get("status"))
        # Alpaca charges no commission on US equities; regulatory fees on sales are not modelled here
        return {"order_id": oid, "filled": filled, "avg_price": avg, "fee": 0.0, "status": status}

    def closed(self, positions):
        """The positions (open_positions' shape) Alpaca no longer holds, priced from the sells filled since the
        position's last buy (its "since"): shares and price are those sells' total and fill-weighted average, and
        it counts as stopped when a stop order sold some of it or the average is at or under its stop. One gone
        with no such sell (closed by hand, or never this account's) comes back as {"symbol", "note"}, not an exit.
        The query has no "after": Alpaca's filters on when an order was submitted, and a stop placed with the
        first buy fills after the second; the fill time is checked here instead, on the latest closed sells."""
        held = {str(p.get("symbol", "")).upper() for p in self.http("GET", f"{self.base}/v2/positions") or []}
        out = []
        for sym, pos in positions.items():
            if sym in held:
                continue
            since = pos.get("since")
            q = urllib.parse.urlencode({"status": "closed", "symbols": sym, "side": "sell", "direction": "desc",
                                        "limit": SELL_ORDERS_LIMIT})
            sells = [o for o in self.http("GET", f"{self.base}/v2/orders?{q}") or []
                     if float(o.get("filled_qty") or 0) > 0 and o.get("filled_avg_price")
                     and not (since and o.get("filled_at")
                              and monitor_mod.parse_time(o["filled_at"]) < monitor_mod.parse_time(since))]
            if not sells:
                out.append({"symbol": sym, "note": f"Alpaca no longer holds {sym} but shows no filled sell since "
                                                   "its buy: closed, unpriced (no exit for the monitor)"})
                continue
            shares = sum(float(o["filled_qty"]) for o in sells)
            price = sum(float(o["filled_qty"]) * float(o["filled_avg_price"]) for o in sells) / shares
            days = sorted(str(o["filled_at"])[:10] for o in sells if o.get("filled_at"))
            stopped = any(o.get("type") in STOP_TYPES for o in sells) or price <= pos["stop"]
            out.append({"symbol": sym, "shares": int(shares) if shares == int(shares) else shares,
                        "price": round(price, 4), "stopped": bool(stopped), **({"day": days[-1]} if days else {})})
        return out


def _stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_log(log=LOG):
    """The paper log's lines, oldest first; [] before the first run."""
    log = pathlib.Path(log)
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


def open_positions(lines, broker=None):
    """{symbol: {"shares", "stop", "since"?}}: what the log's fills bought, less what its exit and unpriced-exit
    lines closed. A second buy of a stock adds its shares and moves the stop to the newer one; "since" is when the
    last buy was logged. broker: only the lines that broker wrote (a line from before brokers were recorded is
    "sim"'s)."""
    pos = {}
    for ln in lines:
        if broker is not None and ln.get("broker", "sim") != broker:
            continue
        if "exit" in ln or "unpriced_exit" in ln:
            pos.pop(str((ln.get("exit") or ln["unpriced_exit"])["symbol"]).upper(), None)
        elif (ln.get("result") or {}).get("stop_order"):
            order = ln["result"]["stop_order"]
            sym = str(order["symbol"]).upper()
            pos[sym] = {"shares": pos.get(sym, {}).get("shares", 0) + order["shares"], "stop": order["stop"],
                        **({"since": ln["logged"]} if ln.get("logged") else {})}
    return pos


def run(signals, view, broker, log=LOG, clock=time.time, monitor=None, now=None):
    """Each signal through the monitor, the rules and the broker, one log line each. Returns the signal lines.
    First, the log's open positions from this broker that it reports closed (a broker with closed()) are logged
    as exits and recorded in the monitor; if it fails to answer, a warning line is logged and the signals go on.
    A monitor made here also learns the exits this broker already logged (a line with no broker is "sim"'s), so a
    simulated stop-out does not pause an --alpaca run; one passed in keeps its own history (a replay passes the
    same monitor to every day's run and records its stop-outs itself)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    earlier = read_log(log)
    name = getattr(broker, "name", type(broker).__name__.lower())
    if monitor is None:
        monitor = monitor_mod.Monitor(kill_file=log.parent / "KILL")
        for ln in earlier:
            if "exit" in ln and ln.get("broker", "sim") == name:
                monitor.record_exit(ln["exit"]["day"], ln["exit"]["symbol"], ln["exit"]["stopped"])
    held = open_positions(earlier, broker=name)
    warning = None
    try:
        closed = broker.closed(held) if held and hasattr(broker, "closed") else []
    except execution.BrokerError as e:              # a 5xx or a timeout: the exits wait for the next run
        closed, warning = [], f"could not check open positions with the broker: {e}"
    today = monitor_mod.utc_day(now).isoformat()
    lines = []
    with log.open("a", encoding="utf-8") as f:
        if warning:
            f.write(json.dumps({"logged": _stamp(), "broker": name, "warning": warning}) + "\n")
        for e in closed:
            if "note" in e:
                f.write(json.dumps({"logged": _stamp(), "broker": name, "note": e["note"],    # closed: not asked again
                                    "unpriced_exit": {"symbol": e["symbol"]}}) + "\n")
                continue
            e = {"day": today, **e}
            monitor.record_exit(e["day"], e["symbol"], e["stopped"])
            f.write(json.dumps({"logged": _stamp(), "broker": name, "exit": e}) + "\n")
        for s in signals:
            t0 = clock()
            stop = monitor.gate(s, now=now)
            if stop:
                result = {"symbol": str(s.get("symbol", "")).upper(), "planned": 0, "filled": 0, "status": "stopped by the monitor",
                          "blocked_by": stop, "orders": [], "fees": 0.0}
            else:
                try:
                    result = execution.execute_buy(view, s, broker)
                except portfolio.PortfolioError as e:
                    result = {"symbol": str(s.get("symbol", "")).upper(), "planned": 0, "filled": 0, "status": "rejected by the rules",
                              "blocked_by": str(e), "orders": [], "fees": 0.0}
                monitor.record(result)
            if result.get("filled"):
                view = execution.after_fill(view, result["symbol"], result["filled"], result["avg_price"], s.get("sector"))
            line = {"logged": _stamp(), "broker": name, "signal": s, "result": result, "latency_s": round(clock() - t0, 3)}
            f.write(json.dumps(line) + "\n")
            lines.append(line)
    return lines


def divergence(lines):
    """Signals against executions: how many went through whole, in part, not at all, and why; and the exits."""
    exits = [ln["exit"] for ln in lines if "exit" in ln]
    lines = [ln for ln in lines if "result" in ln]
    n = len(lines)
    by = {}
    for ln in lines:
        r = ln["result"]
        key = (("blocked: " if r["status"] == "blocked" else "monitor: ") + str(r.get("blocked_by")).split(":")[0]
               if r["status"] in ("blocked", "stopped by the monitor") else r["status"])
        by[key] = by.get(key, 0) + 1
    filled = [ln for ln in lines if ln["result"].get("filled")]
    slip = [ln["result"]["avg_price"] / ln["signal"]["entry"] - 1 for ln in filled]
    planned = sum(ln["result"].get("planned", 0) for ln in lines)
    return {"signals": n, "outcomes": dict(sorted(by.items(), key=lambda kv: -kv[1])),
            "fill_rate_of_planned_shares": (sum(ln["result"]["filled"] for ln in lines) / planned) if planned else None,
            "mean_slippage_vs_signal": statistics.fmean(slip) if slip else None,
            "worst_slippage_vs_signal": max(slip) if slip else None,
            "median_latency_s": statistics.median(ln["latency_s"] for ln in lines) if lines else None,
            "fees": round(sum(ln["result"].get("fees", 0.0) for ln in lines), 2),
            "warnings": sorted({w for ln in lines for w in ln["result"].get("warnings", [])}),
            "exits": len(exits), "stop_outs": sum(1 for e in exits if e.get("stopped"))}


def main(argv):
    if not argv or argv[0] not in ("run", "report") or (argv[0] == "run" and len(argv) < 2):
        print(__doc__)
        return 1
    if argv[0] == "report":
        print(json.dumps(divergence(read_log(LOG)), indent=1))
        return 0
    import ipo_bot
    ipo_bot.load_env()
    signals = json.loads(pathlib.Path(argv[1]).read_text(encoding="utf-8"))
    pf_path = argv[argv.index("--portfolio") + 1] if "--portfolio" in argv else str(HERE / "portfolio.example.json")
    view = portfolio.valued(portfolio.load(pf_path))
    if "--alpaca" in argv:
        broker = AlpacaPaper()
    else:
        broker = execution.SimBroker({s["symbol"]: {"bid": s["entry"] * 0.999, "ask": s["entry"] * 1.001,
                                                    "depth": int(s.get("avg_volume") or 100_000) // 50} for s in signals})
    before = len(read_log(LOG))
    run(signals, view, broker, log=LOG)
    print(json.dumps(divergence(read_log(LOG)[before:]), indent=1))   # this run's lines: its exits and its signals
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
