"""The monitor: the last step of the loop (read, decide, size, execute, monitor), and the one that can say stop.

Before an order goes out it checks, in order:
  - the kill switch: a file named KILL in forecasts/paper/ (or the portfolio's "halted": true, which sizing checks)
  - API health: after MAX_API_FAILURES failed orders in a row, nothing more is sent until reset()
  - stale data: a quote older than MAX_QUOTE_AGE_S seconds is not traded on
  - an abnormal spread: wider than MAX_SPREAD_PCT of the mid price
  - price deviation: an ask more than MAX_DEVIATION_PCT from the signal's own price (the guard against a
    stale or broken reference)
and two protections after freqtrade's (re-implemented from its documented behaviour, no code copied):
  - cooldown: no new buy of a stock within COOLDOWN_DAYS of selling it
  - stoploss guard: after STOP_GUARD_COUNT stop-outs within STOP_GUARD_DAYS, no new buys for STOP_GUARD_PAUSE days
Each check names itself when it blocks, so the paper log says why a signal never became an order.
"""
import datetime as dt
import pathlib

MAX_API_FAILURES = 3
MAX_QUOTE_AGE_S = 60.0
MAX_SPREAD_PCT = 0.5
MAX_DEVIATION_PCT = 0.5
COOLDOWN_DAYS = 3
STOP_GUARD_COUNT, STOP_GUARD_DAYS, STOP_GUARD_PAUSE = 3, 10, 5
KILL_FILE = pathlib.Path(__file__).resolve().parents[1] / "forecasts" / "paper" / "KILL"


def _ts(v):
    d = v if isinstance(v, dt.datetime) else dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


class Monitor:
    def __init__(self, max_api_failures=MAX_API_FAILURES, max_quote_age_s=MAX_QUOTE_AGE_S, max_spread_pct=MAX_SPREAD_PCT,
                 max_deviation_pct=MAX_DEVIATION_PCT, kill_file=KILL_FILE, cooldown_days=COOLDOWN_DAYS,
                 stop_guard=(STOP_GUARD_COUNT, STOP_GUARD_DAYS, STOP_GUARD_PAUSE)):
        self.max_api_failures, self.max_quote_age_s = max_api_failures, max_quote_age_s
        self.max_spread_pct, self.max_deviation_pct, self.kill_file = max_spread_pct, max_deviation_pct, pathlib.Path(kill_file)
        self.cooldown_days, self.stop_guard = cooldown_days, stop_guard
        self.failures = 0
        self.exits = []                                   # (date, symbol, was a stop-out)

    def record_exit(self, day, symbol, stopped):
        """A position closed on `day` (ISO date); stopped: whether its stop was hit."""
        self.exits.append((str(day)[:10], str(symbol).upper(), bool(stopped)))

    def gate(self, signal, now=None):
        """None to go ahead, or the reason not to. signal["quote"] = {"bid", "ask", "time"} when there is one."""
        if self.kill_file.exists():
            return "kill switch: " + self.kill_file.name + " is present"
        if self.failures >= self.max_api_failures:
            return f"paused: {self.failures} API failures in a row"
        today = (_ts(now) if now else dt.datetime.now(dt.timezone.utc)).date()
        days_since = lambda d: (today - dt.date.fromisoformat(d)).days
        sym = str(signal.get("symbol", "")).upper()
        recent = [d for d, s, _ in self.exits if s == sym and 0 <= days_since(d) <= self.cooldown_days]
        if recent:
            return f"cooldown: {sym} was sold on {recent[-1]}"
        count, window, pause = self.stop_guard
        stops = sorted(d for d, _, stopped in self.exits if stopped and 0 <= days_since(d) <= window + pause)
        for k in range(len(stops) - count + 1):
            first, last = stops[k], stops[k + count - 1]
            if (dt.date.fromisoformat(last) - dt.date.fromisoformat(first)).days <= window and days_since(last) <= pause:
                return f"stoploss guard: {count} stop-outs between {first} and {last}"
        q = signal.get("quote")
        if not q:
            return None
        now = _ts(now) if now else dt.datetime.now(dt.timezone.utc)
        if q.get("time") is None:
            return "stale data: the quote has no time"
        age = (now - _ts(q["time"])).total_seconds()
        if age > self.max_quote_age_s:
            return f"stale data: the quote is {age:.0f}s old"
        bid, ask = q.get("bid"), q.get("ask")
        if not (isinstance(bid, (int, float)) and isinstance(ask, (int, float))) or bid <= 0 or ask < bid:
            return "bad quote: bid and ask are missing or crossed"
        mid = (bid + ask) / 2
        if 100 * (ask - bid) / mid > self.max_spread_pct:
            return f"abnormal spread: {100 * (ask - bid) / mid:.2f}% of the mid"
        ref = signal.get("entry")
        if ref and 100 * abs(ask - ref) / ref > self.max_deviation_pct:
            return f"price deviation: the ask is {100 * (ask / ref - 1):+.2f}% from the signal's price"
        return None

    def record(self, result):
        """Count consecutive API failures from an execution result; any answered order resets the count."""
        if str(result.get("status", "")).startswith(("api error", "partial, then an api error")):
            self.failures += 1
        elif result.get("orders"):
            self.failures = 0

    def reset(self):
        self.failures = 0
