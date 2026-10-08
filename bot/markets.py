"""The NYSE calendar and US Eastern time, without a dependency: trading days, holidays and the close in UTC.

Holidays follow NYSE rule 7.2: New Year's Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday,
Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving and Christmas, moved to Friday
when they fall on a Saturday and to Monday on a Sunday (except a Saturday New Year's Day, which is not moved),
plus the market's unscheduled closures. Eastern time is UTC-5, or UTC-4 from the second Sunday of March to the
first Sunday of November (the US rule since 2007).
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import FrozenSet, Iterator, List, Set

# Days the exchange closed outside the rule: storms, national days of mourning, the 2001 attacks.
SPECIAL_CLOSURES: Set[dt.date] = {
    dt.date(2001, 9, 11),
    dt.date(2001, 9, 12),
    dt.date(2001, 9, 13),
    dt.date(2001, 9, 14),
    dt.date(2004, 6, 11),
    dt.date(2007, 1, 2),
    dt.date(2012, 10, 29),
    dt.date(2012, 10, 30),
    dt.date(2018, 12, 5),
    dt.date(2025, 1, 9),
}
CLOSE_HOUR_EASTERN = 16
FILING_DELAY = dt.timedelta(hours=6)  # EDGAR disseminates until 22:00 New York: a day's filings are public by then


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lr = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lr) // 451
    month = (h + lr - 7 * m + 114) // 31
    day = (h + lr - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th given weekday of a month (n = -1 for the last)."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


@functools.lru_cache(maxsize=256)
def holidays(year: int) -> FrozenSet[dt.date]:
    """NYSE full-day holidays in a year."""
    out = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter(year) - dt.timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(dt.date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(dt.date(year, 12, 25)),  # Christmas
    }
    new_year = dt.date(year, 1, 1)
    if new_year.weekday() != 5:  # a Saturday New Year's Day is not moved to Friday
        out.add(_observed(new_year))
    if year >= 2022:
        out.add(_observed(dt.date(year, 6, 19)))  # Juneteenth
    return frozenset(out)


def quarter_start(d: dt.date) -> dt.date:
    """The first day of d's calendar quarter."""
    return dt.date(d.year, (d.month - 1) // 3 * 3 + 1, 1)


def next_quarter(d: dt.date) -> dt.date:
    """The first day of the quarter after d's."""
    q = quarter_start(d)
    return dt.date(q.year + 1, 1, 1) if q.month == 10 else dt.date(q.year, q.month + 3, 1)


def quarter_label(d: dt.date) -> str:
    """2025Q3."""
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year) and d not in SPECIAL_CLOSURES


def trading_days(start: dt.date, end: dt.date) -> List[dt.date]:
    """Every trading day from start to end, both included."""
    return [d for d in _days(start, end) if is_trading_day(d)]


def _days(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def previous_trading_day(d: dt.date) -> dt.date:
    """The last trading day strictly before d."""
    d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def next_trading_day(d: dt.date) -> dt.date:
    """The first trading day strictly after d."""
    d += dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def eastern_offset_hours(moment: dt.datetime) -> int:
    """UTC offset of New York at a UTC moment: -4 in daylight time, -5 otherwise (DST switches at 2:00 local)."""
    utc = moment.astimezone(dt.UTC).replace(tzinfo=None)
    year = utc.year
    start = dt.datetime.combine(_nth_weekday(year, 3, 6, 2), dt.time(7))  # 2:00 EST = 07:00 UTC
    end = dt.datetime.combine(_nth_weekday(year, 11, 6, 1), dt.time(6))  # 2:00 EDT = 06:00 UTC
    return -4 if start <= utc < end else -5


def close_utc(d: dt.date) -> dt.datetime:
    """The 16:00 New York close of a day, in UTC."""
    noon = dt.datetime.combine(d, dt.time(12), tzinfo=dt.UTC)
    return dt.datetime.combine(d, dt.time(CLOSE_HOUR_EASTERN), tzinfo=dt.UTC) - dt.timedelta(
        hours=eastern_offset_hours(noon)
    )


def open_utc(d: dt.date) -> dt.datetime:
    """The 9:30 New York open of a day, in UTC."""
    return close_utc(d) - dt.timedelta(hours=6, minutes=30)


def last_closed_session(now: dt.datetime) -> dt.date:
    """The most recent trading day whose close has passed at `now` (UTC)."""
    d = now.astimezone(dt.UTC).date()
    if is_trading_day(d) and now >= close_utc(d):
        return d
    return previous_trading_day(d)
