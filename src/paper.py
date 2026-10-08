"""Paper trading: signals through the risk rules to a broker, every step logged, and where the two diverge.

    python src/paper.py run signals.json --portfolio portfolio.json            simulated broker, no account
    python src/paper.py run signals.json --portfolio portfolio.json --alpaca   Alpaca's paper-trading API
    python src/paper.py report                                                 signals against executions

signals.json: [{"time": ISO, "symbol", "entry", "stop", "conviction", "sector"?, "next_earnings"?, "avg_volume"?}]
(the bot's portfolio-mode buys, or any strategy's). Each run appends to forecasts/paper/log.jsonl: the signal,
what the rules allowed, each order sent, each fill, and the time from signal to fill.

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "forecasts" / "paper" / "log.jsonl"
PAPER_HOST = "paper-api.alpaca.markets"
ALPACA_PAPER = "https://paper-api.alpaca.markets"


class AlpacaPaper:
    """Alpaca's paper-trading API with SimBroker's submit() shape: a day limit order, polled until it is done
    or `wait_s` passes, then whatever is left is cancelled. http(method, url, body) -> dict is injectable for tests."""

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


def run(signals, view, broker, log=LOG, clock=time.time, monitor=None, now=None):
    """Each signal through the monitor, the rules and the broker, one log line each. Returns the lines."""
    log.parent.mkdir(parents=True, exist_ok=True)
    monitor = monitor or monitor_mod.Monitor(kill_file=log.parent / "KILL")
    lines = []
    with log.open("a", encoding="utf-8") as f:
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
            line = {"logged": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "signal": s, "result": result,
                    "latency_s": round(clock() - t0, 3)}
            f.write(json.dumps(line) + "\n")
            lines.append(line)
    return lines


def divergence(lines):
    """Signals against executions: how many went through whole, in part, not at all, and why."""
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
            "warnings": sorted({w for ln in lines for w in ln["result"].get("warnings", [])})}


def main(argv):
    if not argv or argv[0] not in ("run", "report"):
        print(__doc__)
        return 1
    if argv[0] == "report":
        lines = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()] if LOG.exists() else []
        print(json.dumps(divergence(lines), indent=1))
        return 0
    import ipo_bot
    ipo_bot.load_env()
    signals = json.loads(pathlib.Path(argv[1]).read_text(encoding="utf-8"))
    pf_path = argv[argv.index("--portfolio") + 1] if "--portfolio" in argv else str(ROOT / "portfolio.example.json")
    view = portfolio.valued(portfolio.load(pf_path))
    if "--alpaca" in argv:
        broker = AlpacaPaper()
    else:
        broker = execution.SimBroker({s["symbol"]: {"bid": s["entry"] * 0.999, "ask": s["entry"] * 1.001,
                                                    "depth": int(s.get("avg_volume") or 100_000) // 50} for s in signals})
    lines = run(signals, view, broker)
    print(json.dumps(divergence(lines), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
