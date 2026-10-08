"""Paper trading, replayed: the last N trading days of real prices through the monitor, the rules and a simulated
broker, day by day, with the same log and divergence report as a live paper run.

    python evaluation/paper_replay.py              the last 10 trading days (two weeks)
    python evaluation/paper_replay.py --days 20

A stand-in for a live paper test, not a replacement: the prices, volumes and lows are real, but the order book
is simulated (depth from each stock's volume, a spread of a few basis points), so queue position, real
latency and broker outages are not tested. src/paper.py with --alpaca is the live version.

The signal (fixed in advance; a test signal, not a recommendation): after each close, a stock in the benchmark's
30 US large caps that closed at a 20-day high while its 50-day average was above its 200-day is a buy at the
next open, with its stop at the 20-day low. Each morning: open positions whose stop was crossed are sold (at
the stop, or the open if it gapped below); new signals go through the monitor and every rule, and what fills
at the open is held against the same day's low (sold at its stop if the low reaches it); the book is marked at
the close, and the day's loss and the account's high feed the guardrails for the next morning.
"""
import datetime as dt
import json
import pathlib
import statistics
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import benchmark as bm  # noqa: E402
import execution  # noqa: E402
import monitor as monitor_mod  # noqa: E402
import paper  # noqa: E402
import portfolio  # noqa: E402

START_CASH = 100_000.0
SPREAD = 0.0004                 # 4 basis points, about a large cap's quoted spread
DEPTH_SHARE = 0.002             # shares on show at the ask: 0.2% of the day's volume
FEE = 0.0                       # US equity commissions are zero at most brokers; slippage comes from the book
SECTORS = {"AAPL": "Technology", "MSFT": "Technology", "CSCO": "Technology", "INTC": "Technology", "IBM": "Technology",
           "ORCL": "Technology", "AMZN": "Consumer", "HD": "Consumer", "MCD": "Consumer", "WMT": "Staples", "PG": "Staples",
           "KO": "Staples", "PEP": "Staples", "GOOGL": "Communication", "DIS": "Communication", "T": "Communication",
           "VZ": "Communication", "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "XOM": "Energy", "CVX": "Energy",
           "JNJ": "Health Care", "PFE": "Health Care", "MRK": "Health Care", "UNH": "Health Care", "CAT": "Industrials",
           "GE": "Industrials", "MMM": "Industrials", "BA": "Industrials"}
RULES = {"max_position_pct": 10, "max_sector_pct": 30, "risk_per_trade_pct": 0.5, "min_cash_pct": 5,
         "conviction_scale": {"Low": 0.5, "Medium": 1, "High": 1.5}, "max_order_value": 15_000, "max_gross_exposure_pct": 80,
         "max_daily_loss_pct": 2, "max_drawdown_pct": 8, "max_volume_pct": 1}


def bars(symbol, days=420):
    """[(date, open, high, low, close, volume)] for the last `days` calendar days. Yahoo's quote prices are
    split-adjusted (earlier days divided by every later split) but not dividend-adjusted. [] when Yahoo fails
    three times; callers check for missing symbols (tracker.run refuses to run without the full universe)."""
    t1 = int(time.time())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={t1 - days * 86400}&period2={t1}&interval=1d"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                res = json.loads(r.read())["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            return [(dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), o, h, lo, c, v)
                    for t, o, h, lo, c, v in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"])
                    if None not in (o, h, lo, c, v)]
        except Exception:
            time.sleep(3 * (attempt + 1))
    return []


def signal_on(rows, i):
    """The breakout signal after close i: (stop) or None. Uses closes up to i only."""
    closes = [r[4] for r in rows[:i + 1]]
    if len(closes) < 200:
        return None
    if not (statistics.fmean(closes[-50:]) > statistics.fmean(closes[-200:]) and closes[-1] >= max(closes[-20:])):
        return None
    return min(r[3] for r in rows[i - 19:i + 1])                           # the 20-day low


def replay(data, days=10, log=None):
    """data: {symbol: bars}. Returns (log lines, daily account rows, exits)."""
    if not data:
        raise ValueError("no price history for any symbol: nothing to replay")
    calendar = sorted(set.intersection(*(set(r[0] for r in rows) for rows in data.values())))
    run_days = calendar[-days:]
    by = {s: {r[0]: k for k, r in enumerate(rows)} for s, rows in data.items()}
    cash, positions = START_CASH, {}                                      # symbol -> {shares, stop, cost}
    watch = monitor_mod.Monitor(kill_file=(log or ROOT / "forecasts" / "paper" / "x").parent / "KILL")   # one for the whole run
    high, prev_close_equity = START_CASH, START_CASH
    lines, account, exits = [], [], []
    log = log or (ROOT / "forecasts" / "paper" / f"replay_{dt.date.today().isoformat()}.jsonl")
    if log.exists():
        log.unlink()
    def stop_out(day, symbols):
        """Sell each of `symbols` whose stop the day's low crossed: at the stop, or at the open below it."""
        nonlocal cash
        for s in symbols:
            o, lo = data[s][by[s][day]][1], data[s][by[s][day]][3]
            pos = positions[s]
            if lo <= pos["stop"]:
                px = min(o, pos["stop"])
                cash += pos["shares"] * px * (1 - FEE)
                exits.append({"day": day, "symbol": s, "shares": pos["shares"], "price": round(px, 2),
                              "pnl": round(pos["shares"] * px - pos["cost"], 2)})
                watch.record_exit(day, s, stopped=True)
                del positions[s]

    for day in run_days:
        # 1. stops crossed overnight or during the day: sold at the stop, or at the open below it
        stop_out(day, list(positions))
        # 2. the morning's signals, from yesterday's close, through the monitor, the rules and the broker
        i_prev = {s: by[s][day] - 1 for s in data}
        open_equity = cash + sum(p["shares"] * data[s][by[s][day]][1] for s, p in positions.items())
        holdings = [{"symbol": s, "shares": p["shares"], "cost_basis": p["cost"] / p["shares"], "sector": SECTORS[s],
                     "price": data[s][by[s][day]][1], "price_date": day} for s, p in positions.items()]
        view = portfolio.valued(portfolio.validate({"as_of": day, "cash": cash, "holdings": holdings, "rules": RULES,
                                                    "day_pnl": open_equity - prev_close_equity, "equity_high": high}))
        quotes, signals, day_bars = {}, [], {}
        when = f"{day}T13:30:00Z"                                         # the open, 9:30 New York
        for s in sorted(data):
            if s in positions:
                continue
            stop = signal_on(data[s], i_prev[s])
            if stop is None:
                continue
            o, v_prev = data[s][by[s][day]][1], data[s][i_prev[s]][5]
            quotes[s] = {"bid": o * (1 - SPREAD / 2), "ask": o * (1 + SPREAD / 2), "depth": max(1, int(v_prev * DEPTH_SHARE))}
            day_bars[s] = data[s][by[s][day]][1:6]
            avg_vol = statistics.fmean(r[5] for r in data[s][i_prev[s] - 19:i_prev[s] + 1])
            signals.append({"time": f"{data[s][i_prev[s]][0]}T20:00:00Z", "symbol": s, "entry": round(data[s][i_prev[s]][4], 2),
                            "stop": round(stop, 2), "conviction": "Medium", "sector": SECTORS[s], "avg_volume": avg_vol,
                            "quote": {"bid": quotes[s]["bid"], "ask": quotes[s]["ask"], "time": when}})
        broker = execution.BarBroker(day_bars, fee_rate=FEE)
        day_lines = paper.run(signals, view, broker, log=log, monitor=watch, now=when)
        bought = []
        for ln in day_lines:
            r = ln["result"]
            if r.get("filled"):
                spent = r["filled"] * r["avg_price"] + r.get("fees", 0.0)
                cash -= spent
                positions[r["symbol"]] = {"shares": r["filled"], "stop": ln["signal"]["stop"], "cost": spent}
                bought.append(r["symbol"])
            ln["day"] = day
        lines += day_lines
        # a position bought at the open is live for the rest of the day: today's low can take out its stop too
        stop_out(day, bought)
        # 3. the close: mark the book; the high and the day's result feed tomorrow's guardrails
        equity = cash + sum(p["shares"] * data[s][by[s][day]][4] for s, p in positions.items())
        high = max(high, equity)
        account.append({"day": day, "equity": round(equity, 2), "cash": round(cash, 2), "positions": len(positions),
                        "signals": len(signals), "filled": sum(1 for ln in day_lines if ln["result"].get("filled"))})
        prev_close_equity = equity
    return lines, account, exits


def main(argv):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    days = int(argv[argv.index("--days") + 1]) if "--days" in argv else 10
    symbols = bm.UNIVERSES["US large caps"]
    data = {s: bars(s) for s in symbols}
    data = {s: rows for s, rows in data.items() if len(rows) > 230}
    missing = [s for s in symbols if s not in data]
    spy = bars("SPY")
    if not spy:
        raise SystemExit("no prices for SPY from Yahoo; try again later")
    lines, account, exits = replay(data, days)
    d = paper.divergence(lines)
    first, last = account[0], account[-1]
    spy_by = {r[0]: r for r in spy}
    spy_ret = spy_by[last["day"]][4] / spy_by[first["day"]][1] - 1
    out = [f"# Paper trading replay: {first['day']} to {last['day']} ({len(account)} trading days)", "",
           f"{len(data)} large caps{' (no prices for ' + ', '.join(missing) + ')' if missing else ''}; ${START_CASH:,.0f} to start; every guardrail on (0.5% risk a trade, $15,000 an order, 80% "
           "gross exposure, 2% daily loss, 8% drawdown, 1% of volume). Real prices and volumes, a simulated order book.", "",
           "| Day | signals | filled | positions | equity |", "|---|---|---|---|---|"]
    out += [f"| {a['day']} | {a['signals']} | {a['filled']} | {a['positions']} | ${a['equity']:,.2f} |" for a in account]
    out += ["", f"Account {100 * (last['equity'] / START_CASH - 1):+.2f}% against SPY {100 * spy_ret:+.2f}% over the same days "
            f"(two weeks says nothing about skill; this run tests the plumbing).", "",
            "## Signals against executions", "", "| Outcome | count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in d["outcomes"].items()]
    out += ["", f"Fill rate of planned shares: {100 * d['fill_rate_of_planned_shares']:.1f}%; mean slippage from the signal's price "
            f"(yesterday's close to today's fill, so it includes the overnight gap): {100 * d['mean_slippage_vs_signal']:+.2f}%; "
            f"worst {100 * d['worst_slippage_vs_signal']:+.2f}%; fees ${d['fees']:.2f}; median time per signal {d['median_latency_s'] * 1000:.1f} ms."
            if d["fill_rate_of_planned_shares"] is not None else "No orders were planned.", ""]
    if d["warnings"]:
        out += ["Warnings: " + "; ".join(d["warnings"]), ""]
    out += ["## Stops hit", ""] + ([f"- {e['day']} {e['symbol']}: {e['shares']} shares at ${e['price']}, {e['pnl']:+,.2f}" for e in exits]
                                   or ["None."]) + [""]
    text = "\n".join(out)
    path = ROOT / "bench_runs" / f"paper_replay_{dt.date.today().isoformat()}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}; the full log is in forecasts/paper/")


if __name__ == "__main__":
    main(sys.argv[1:])
