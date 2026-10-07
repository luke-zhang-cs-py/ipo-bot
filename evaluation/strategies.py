"""Five beginner trading bots, each one rule written down before any test, backtested without fooling ourselves.

    python evaluation/strategies.py          every bot, every stress test; results to bench_runs/

The bots (SPY and IEF daily bars from Yahoo, adjusted for dividends and splits, cached for the day):
  DCA             buy $100 of SPY at the first open of every week
  Rebalancing     60% SPY / 40% IEF, put back when SPY's weight drifts 5 points or more
  Trend           all in SPY while its 50-day average is above its 200-day average, else all cash
  Mean reversion  buy SPY after a close 2 standard deviations below its 20-day mean; sell after a close back at
                  the mean, or after 20 days
  Grid            20 levels, 10% either side of SPY's first close of the year; hold a twentieth more for every
                  level the price is below the top (buys low, sells high; a breakout down leaves it fully loaded)
The protocol, as the brief has it: every rule is fixed here and not tuned; a signal is read from a completed
close and filled at the next day's open (never at the close it was read from); every fill pays a fee and
slippage (0.10% and 0.05% a side by default); the history is split in two to see if the result holds; and
every bot is re-run at 1.5x and 2x the friction, with fills a day late, and with each parameter nudged
20% either way. A bot whose result against simply holding SPY changes sign under any of those is fragile.
"""
import datetime as dt
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))
try:                                   # the OS certificate store, as the bot uses
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

START = "2005-01-01"
FEE = 0.0010                    # per side
SLIPPAGE = 0.0005               # per side
STRESS = (1.0, 1.5, 2.0)        # multiples of the friction
NUDGE = 0.2                     # each parameter moved this much either way
LEVEL = 0.001                   # an edge within 0.1% a year of holding SPY counts as level with it, not as a sign
MIN_TRADE = 0.02                # weight changes smaller than this are not traded (saves dust orders)
SPLIT = "2016-01-01"            # the history in two halves


# ----------------------------------------------------------------------------- costs

def friction(position, fee=FEE, slippage=SLIPPAGE, round_trips=1):
    """The cost of trading `position` in and out `round_trips` times: (per round trip, in total, round-trip rate)."""
    rate = 2 * fee + 2 * slippage
    return position * rate, position * rate * round_trips, rate


# ----------------------------------------------------------------------------- data

def daily(symbol):
    """[(date, open, close)] adjusted for splits and dividends, from START; cached for the day."""
    path = ROOT / "bench_data" / f"daily_{symbol}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fetched") == dt.date.today().isoformat():
            return [tuple(b) for b in cached["bars"]]
    t0 = int(dt.datetime.fromisoformat(START).replace(tzinfo=dt.timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={t0}&period2={int(time.time())}&interval=1d&events=div%2Csplit"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    q, adj = res["indicators"]["quote"][0], res["indicators"]["adjclose"][0]["adjclose"]
    bars = []
    for t, o, c, a in zip(res["timestamp"], q["open"], q["close"], adj):
        if o and c and a:
            k = a / c                                   # the adjustment, applied to the open as well
            bars.append((dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), o * k, a))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"fetched": dt.date.today().isoformat(), "bars": bars}), encoding="utf-8")
    return bars


def aligned(*symbols):
    """Dates every symbol traded, and {symbol: [(open, close)]} on those dates."""
    series = {s: {d: (o, c) for d, o, c in daily(s)} for s in symbols}
    dates = sorted(set.intersection(*(set(v) for v in series.values())))
    return dates, {s: [series[s][d] for d in dates] for s in symbols}


# ----------------------------------------------------------------------------- the bots: weights from closes only

def sma(xs, n, i):
    return statistics.fmean(xs[i - n + 1:i + 1]) if i + 1 >= n else None


def dca(params=None):
    p = {"amount": 100.0, **(params or {})}
    return {"name": "DCA", "assets": ["SPY"], "params": {}, "contribution": p["amount"],
            "decide": lambda i, closes, held: {"SPY": 1.0}}


def rebalancing(params=None):
    p = {"spy": 0.6, "drift": 0.05, **(params or {})}

    def decide(i, closes, held):
        if held is None or abs(held.get("SPY", 0.0) - p["spy"]) >= p["drift"]:
            return {"SPY": p["spy"], "IEF": 1 - p["spy"]}
        return held                                     # inside the band: leave it
    return {"name": "Rebalancing 60/40", "assets": ["SPY", "IEF"], "params": {"drift": p["drift"]}, "decide": decide}


def trend(params=None):
    p = {"fast": 50, "slow": 200, **(params or {})}

    def decide(i, closes, held):
        f, s = sma(closes["SPY"], int(p["fast"]), i), sma(closes["SPY"], int(p["slow"]), i)
        return {"SPY": 1.0 if (f is not None and s is not None and f > s) else 0.0}
    return {"name": "Trend 50/200", "assets": ["SPY"], "params": {"fast": p["fast"], "slow": p["slow"]}, "decide": decide}


def mean_reversion(params=None):
    p = {"window": 20, "z": 2.0, "max_hold": 20, **(params or {})}
    state = {"since": None}

    def decide(i, closes, held):
        n = int(p["window"])
        if i + 1 < n:
            return {"SPY": 0.0}
        xs = closes["SPY"][i - n + 1:i + 1]
        mu, sd = statistics.fmean(xs), statistics.pstdev(xs)
        c = closes["SPY"][i]
        inside = (held or {}).get("SPY", 0.0) > 0.5
        if not inside and sd and c < mu - p["z"] * sd:
            state["since"] = i
            return {"SPY": 1.0}
        if inside and (c >= mu or i - (state["since"] or i) >= p["max_hold"]):
            return {"SPY": 0.0}
        return {"SPY": 1.0 if inside else 0.0}
    return {"name": "Mean reversion 20d/2sd", "assets": ["SPY"], "params": {"window": p["window"], "z": p["z"]}, "decide": decide}


def grid(params=None):
    p = {"levels": 20, "width": 0.10, **(params or {})}
    state = {"year": None, "lo": None, "hi": None}

    def decide(i, closes, held, dates=None):
        c = closes["SPY"][i]
        year = (dates or {}).get(i, "")[:4]
        if state["year"] != year:
            state.update(year=year, lo=c * (1 - p["width"]), hi=c * (1 + p["width"]))
        levels = int(p["levels"])
        below = math.floor((state["hi"] - c) / ((state["hi"] - state["lo"]) / levels))
        return {"SPY": min(1.0, max(0.0, below / levels))}
    return {"name": "Grid 20 levels, +/-10%", "assets": ["SPY"], "params": {"levels": p["levels"], "width": p["width"]},
            "decide": decide, "needs_dates": True}


BOTS = {"dca": dca, "rebalancing": rebalancing, "trend": trend, "mean_reversion": mean_reversion, "grid": grid}


# ----------------------------------------------------------------------------- the engine

def simulate(bot, dates, bars, cost_x=1.0, delay=0, start_cash=10_000.0, fee=FEE, slippage=SLIPPAGE):
    """Run a bot: after each close it names target weights; they are traded at the open `1 + delay` days later,
    paying fee and slippage (times cost_x) on every fill. Returns the equity curve, flows and closed trades."""
    assets = bot["assets"]
    closes = {a: [c for _, c in bars[a]] for a in assets}
    opens = {a: [o for o, _ in bars[a]] for a in assets}
    side = (fee + slippage) * cost_x
    contribution = bot.get("contribution")
    cash = 0.0 if contribution else start_cash
    shares = {a: 0.0 for a in assets}
    lots = {a: [] for a in assets}                       # FIFO lots: [shares, cost per share incl. buying costs]
    trades, curve, flows = [], [], []
    pending = {}                                          # fill day -> target weights
    held_weights = None
    week = None
    invested_days = 0
    for i, d in enumerate(dates):
        # 1. fills at today's open, from a decision made 1 + delay closes ago
        flow = 0.0
        if contribution:
            wk = dt.date.fromisoformat(d).isocalendar()[:2]
            if wk != week:
                week = wk
                cash += contribution
                flow = contribution
                pending.setdefault(i, held_weights or {"SPY": 1.0})
        if i in pending:
            target = pending.pop(i)
            equity_open = cash + sum(shares[a] * opens[a][i] for a in assets)
            orders = []
            for a in assets:
                want = target.get(a, 0.0) * equity_open / opens[a][i]
                diff = want - shares[a]
                if abs(diff * opens[a][i]) < MIN_TRADE * equity_open and not contribution and target.get(a, 0.0) not in (0.0, 1.0):
                    continue
                if abs(diff) * opens[a][i] < 1.0:
                    continue
                orders.append((a, diff))
            for a, diff in sorted(orders, key=lambda x: x[1]):          # sells first, to fund buys
                px = opens[a][i]
                if diff < 0:
                    qty = -diff
                    got = qty * px * (1 - side)
                    cash += got
                    shares[a] -= qty
                    left = qty
                    while left > 1e-12 and lots[a]:
                        lot = lots[a][0]
                        take = min(left, lot[0])
                        trades.append({"asset": a, "pnl": take * (px * (1 - side) - lot[1]), "cost": take * lot[1], "closed": d})
                        lot[0] -= take
                        left -= take
                        if lot[0] <= 1e-12:
                            lots[a].pop(0)
                else:
                    spend = min(cash, diff * px * (1 + side))
                    qty = spend / (px * (1 + side))
                    if qty <= 0:
                        continue
                    cash -= spend
                    shares[a] += qty
                    lots[a].append([qty, px * (1 + side)])
        # 2. today's close: value, weights, and the next decision
        equity = cash + sum(shares[a] * closes[a][i] for a in assets)
        curve.append(equity)
        invested_days += any(shares[a] * closes[a][i] > 0.01 * max(equity, 1e-9) for a in assets)
        flows.append(flow)
        held_weights = {a: shares[a] * closes[a][i] / equity for a in assets} if equity > 0 else None
        decide = bot["decide"]
        target = decide(i, closes, held_weights, {i: d}) if bot.get("needs_dates") else decide(i, closes, held_weights)
        if target != held_weights and i + 1 + delay < len(dates):
            pending[i + 1 + delay] = target
    return {"curve": curve, "flows": flows, "trades": trades, "dates": dates, "exposure": invested_days / max(1, len(dates))}


def metrics(run, dates=None):
    """Net return, CAGR, max drawdown (on a time-weighted index, so contributions don't count as gains), trades,
    win rate, average win and loss, profit factor, expectancy, and the share of days invested."""
    curve, flows = run["curve"], run["flows"]
    index = [1.0]
    for k in range(1, len(curve)):
        prev = curve[k - 1]
        index.append(index[-1] * ((curve[k] - flows[k]) / prev if prev > 0 else 1.0))
    peak, mdd = index[0], 0.0
    for v in index:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    years = max(1e-9, (dt.date.fromisoformat(run["dates"][-1]) - dt.date.fromisoformat(run["dates"][0])).days / 365.25)
    contributed = sum(flows) or curve[0]
    pnls = [t["pnl"] for t in run["trades"]]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else float("nan")
    avg_win = statistics.fmean(wins) if wins else 0.0
    avg_loss = -statistics.fmean(losses) if losses else 0.0
    return {"net_return": curve[-1] / contributed - 1, "cagr": index[-1] ** (1 / years) - 1, "max_drawdown": mdd,
            "trades": len(pnls), "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf") if wins else float("nan"),
            "expectancy": (win_rate * avg_win - (1 - win_rate) * avg_loss) if pnls else float("nan"),
            "ending": curve[-1], "contributed": contributed, "exposure": run["exposure"]}


def buy_and_hold(dates, bars, cost_x=1.0):
    bot = {"name": "Buy and hold SPY", "assets": ["SPY"], "decide": lambda i, closes, held: {"SPY": 1.0}}
    return simulate(bot, dates, bars, cost_x)


def window(dates, bars, lo=None, hi=None):
    keep = [k for k, d in enumerate(dates) if (lo is None or d >= lo) and (hi is None or d < hi)]
    return [dates[k] for k in keep], {a: [bars[a][k] for k in keep] for a in bars}


def nudges(name):
    """The bot's parameters, each moved NUDGE either way, as constructor arguments."""
    base = BOTS[name]()["params"]
    out = []
    for k, v in base.items():
        for f in (1 - NUDGE, 1 + NUDGE):
            nv = round(v * f) if isinstance(v, int) else v * f
            if name == "trend" and k == "fast" and nv >= base["slow"]:
                continue
            out.append((f"{k} {v}->{nv:g}", {k: nv}))
    return out


def edge(m, bh):
    """A bot's CAGR minus buy-and-hold's over the same days: the number whose sign decides fragility."""
    return m["cagr"] - bh["cagr"]


def side(v):
    """+1 ahead of holding SPY, -1 behind, 0 level (within LEVEL): a move between level and either side is no flip."""
    return 0 if abs(v) < LEVEL else (1 if v > 0 else -1)


def flips(base, others):
    """The labels whose edge lands on the other side of holding SPY from the base result."""
    b = side(base)
    return [label for label, v in others if side(v) and b and side(v) != b]


# ----------------------------------------------------------------------------- the report

def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{100 * x:+.{d}f}%"


def trade_cells(m):
    """Win rate, average win and loss, profit factor and expectancy, or n/a for a bot that closed no trades."""
    if not m["trades"]:
        return "n/a | | | | "
    pf = m["profit_factor"]
    pf_text = "n/a" if math.isnan(pf) else "no losses" if math.isinf(pf) else f"{pf:.2f}"
    return f"{pct(m['win_rate'], 0)} | ${m['avg_win']:.0f} | ${m['avg_loss']:.0f} | {pf_text} | ${m['expectancy']:.0f}"


def report():
    from ipo_eval import spy_regime
    dates, bars = aligned("SPY", "IEF")
    out = [f"# Trading bot backtests, {dt.date.today():%d %b %Y}", "",
           f"Daily bars {dates[0]} to {dates[-1]}; fills at the next open; {100 * FEE:.2f}% fee and {100 * SLIPPAGE:.2f}% slippage a side, "
           f"so a round trip costs {100 * friction(1)[2]:.2f}%. Rules fixed in advance (see the module), not tuned.", ""]
    _, per40, _ = friction(1000, round_trips=40)
    out += [f"Cost check: a $1,000 position traded 40 round trips costs ${friction(1000)[0]:.2f} each, ${per40:.0f} in all.", ""]
    bh = metrics(buy_and_hold(dates, bars))
    out += ["## Results (base costs)", "", "| Bot | net return | CAGR | vs holding SPY | max drawdown | time invested | trades | win rate | avg win | avg loss | profit factor | expectancy |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
            f"| Buy and hold SPY | {pct(bh['net_return'])} | {pct(bh['cagr'])} | | {pct(bh['max_drawdown'])} | {pct(bh['exposure'], 0)} | {bh['trades']} | | | | | |"]
    results = {}
    for key, make in BOTS.items():
        m = metrics(simulate(make(), dates, bars))
        results[key] = m
        out.append(f"| {make()['name']} | {pct(m['net_return'])} | {pct(m['cagr'])} | {pct(edge(m, bh))} | {pct(m['max_drawdown'])} | "
                   f"{pct(m['exposure'], 0)} | {m['trades']} | {trade_cells(m)} |")
    out += ["", "DCA's net return is on the money put in (weekly $100); its CAGR and drawdown use a time-weighted index.", ""]

    out += ["## Stress tests: CAGR against holding SPY", "", "| Bot | base | 1.5x costs | 2x costs | fills a day late | first half | second half | parameters nudged 20% | verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    halves = [window(dates, bars, hi=SPLIT), window(dates, bars, lo=SPLIT)]
    bh_halves = [metrics(buy_and_hold(*h)) for h in halves]
    for key, make in BOTS.items():
        e = {}
        for x in STRESS[1:]:
            e[f"{x}x"] = edge(metrics(simulate(make(), dates, bars, cost_x=x)), metrics(buy_and_hold(dates, bars, cost_x=x)))
        e["late"] = edge(metrics(simulate(make(), dates, bars, delay=1)), bh)
        for k, (hd, hb) in enumerate(halves):
            e[f"half{k}"] = edge(metrics(simulate(make(), hd, hb)), bh_halves[k])
        nudged = [(label, edge(metrics(simulate(make(p), dates, bars)), bh)) for label, p in nudges(key)]
        base = edge(results[key], bh)
        flipped = flips(base, list(e.items()) + nudged)
        verdict = ("level with holding SPY" if side(base) == 0 else "holds" if not flipped
                   else "fragile: flips under " + ", ".join(flipped))
        nudge_range = f"{pct(min(v for _, v in nudged))} to {pct(max(v for _, v in nudged))}" if nudged else "no parameters"
        out.append(f"| {make()['name']} | {pct(base)} | {pct(e['1.5x'])} | {pct(e['2.0x'])} | {pct(e['late'])} | {pct(e['half0'])} | "
                   f"{pct(e['half1'])} | {nudge_range} | {verdict} |")
    out += ["", f"Halves split at {SPLIT}. 'Holds' means the sign of the edge over holding SPY never changed; it does not mean the "
            "edge is positive.", ""]

    out += ["## Regimes: each bot's average daily return against SPY's, by the market on the day", "",
            "| Bot | bull | drawdown | sideways | ordinary |", "|---|---|---|---|---|"]
    spy_close = {d: c for d, (_, c) in zip(dates, bars["SPY"])}
    labels = [spy_regime(spy_close, d) for d in dates]
    bh_curve = buy_and_hold(dates, bars)["curve"]
    for key, make in BOTS.items():
        if key == "dca":
            continue
        curve = simulate(make(), dates, bars)["curve"]
        row = []
        for reg in ("bull", "drawdown", "sideways", "ordinary"):
            diffs = [(curve[k] / curve[k - 1] - 1) - (bh_curve[k] / bh_curve[k - 1] - 1) for k in range(1, len(dates)) if labels[k] == reg]
            row.append(f"{100 * 252 * statistics.fmean(diffs):+.1f}% a yr ({len(diffs)} d)" if len(diffs) > 20 else "too few days")
        out.append(f"| {make()['name']} | " + " | ".join(row) + " |")
    out += ["", "Annualised difference in daily return from holding SPY, on the days the market was in each state "
            "(labelled from SPY's past year only).", ""]
    return "\n".join(out)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    text = report()
    runs = ROOT / "bench_runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"strategies_{dt.datetime.now():%Y-%m-%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
