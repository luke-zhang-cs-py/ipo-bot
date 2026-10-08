"""The NYSE calendar and US Eastern time: holidays, special closures, DST and the close in UTC."""

import datetime as dt

from hypothesis import given
from hypothesis import strategies as st

from bot import markets

D = dt.date


def test_holidays_match_the_exchange() -> None:
    # NYSE's published holidays
    assert markets.holidays(2024) == {
        D(2024, 1, 1),
        D(2024, 1, 15),
        D(2024, 2, 19),
        D(2024, 3, 29),
        D(2024, 5, 27),
        D(2024, 6, 19),
        D(2024, 7, 4),
        D(2024, 9, 2),
        D(2024, 11, 28),
        D(2024, 12, 25),
    }
    assert markets.holidays(2021) == {
        D(2021, 1, 1),
        D(2021, 1, 18),
        D(2021, 2, 15),
        D(2021, 4, 2),
        D(2021, 5, 31),
        D(2021, 7, 5),
        D(2021, 9, 6),
        D(2021, 11, 25),
        D(2021, 12, 24),
    }  # no Juneteenth before 2022; Independence Day on a Sunday moves to Monday; Christmas on a Saturday to Friday


def test_saturday_new_year_is_not_moved() -> None:
    assert D(2021, 12, 31) not in markets.holidays(2021)
    assert D(2022, 1, 1) not in markets.holidays(2022) and markets.is_trading_day(D(2021, 12, 31))
    assert D(2023, 1, 2) in markets.holidays(2023)  # a Sunday New Year's Day is observed on Monday


def test_special_closures_and_weekends() -> None:
    assert not markets.is_trading_day(D(2012, 10, 29))  # Hurricane Sandy
    assert not markets.is_trading_day(D(2025, 1, 9))  # national day of mourning
    assert not markets.is_trading_day(D(2024, 6, 15))  # a Saturday
    assert len(markets.trading_days(D(2024, 1, 1), D(2024, 12, 31))) == 252


def test_easter() -> None:
    assert [markets.easter(y) for y in (2024, 2025, 2026)] == [D(2024, 3, 31), D(2025, 4, 20), D(2026, 4, 5)]


def test_neighbours() -> None:
    assert markets.previous_trading_day(D(2024, 7, 5)) == D(2024, 7, 3)
    assert markets.next_trading_day(D(2024, 7, 3)) == D(2024, 7, 5)
    assert markets.next_trading_day(D(2024, 12, 24)) == D(2024, 12, 26)


def test_dst_and_the_close_in_utc() -> None:
    utc = dt.UTC
    assert markets.eastern_offset_hours(dt.datetime(2024, 3, 10, 6, 59, tzinfo=utc)) == -5
    assert markets.eastern_offset_hours(dt.datetime(2024, 3, 10, 7, 0, tzinfo=utc)) == -4  # 2:00 EST -> 3:00 EDT
    assert markets.eastern_offset_hours(dt.datetime(2024, 11, 3, 5, 59, tzinfo=utc)) == -4
    assert markets.eastern_offset_hours(dt.datetime(2024, 11, 3, 6, 0, tzinfo=utc)) == -5  # 2:00 EDT -> 1:00 EST
    assert markets.close_utc(D(2024, 1, 2)) == dt.datetime(2024, 1, 2, 21, tzinfo=utc)
    assert markets.close_utc(D(2024, 7, 2)) == dt.datetime(2024, 7, 2, 20, tzinfo=utc)
    assert markets.open_utc(D(2024, 7, 2)) == dt.datetime(2024, 7, 2, 13, 30, tzinfo=utc)


def test_last_closed_session() -> None:
    utc = dt.UTC
    assert markets.last_closed_session(dt.datetime(2024, 7, 2, 19, 59, tzinfo=utc)) == D(2024, 7, 1)
    assert markets.last_closed_session(dt.datetime(2024, 7, 2, 20, 0, tzinfo=utc)) == D(2024, 7, 2)
    assert markets.last_closed_session(dt.datetime(2024, 7, 6, 12, tzinfo=utc)) == D(2024, 7, 5)  # a Saturday
    assert markets.last_closed_session(dt.datetime(2024, 1, 2, 20, 30, tzinfo=utc)) == D(2023, 12, 29)  # EST: 21:00


@given(st.dates(min_value=D(1990, 1, 1), max_value=D(2040, 12, 31)))
def test_calendar_properties(d: dt.date) -> None:
    nxt, prev = markets.next_trading_day(d), markets.previous_trading_day(d)
    assert prev < d < nxt and markets.is_trading_day(nxt) and markets.is_trading_day(prev)
    assert markets.trading_days(prev, nxt)[0] == prev and markets.trading_days(prev, nxt)[-1] == nxt
    assert all(
        not markets.is_trading_day(x)
        for x in (prev + dt.timedelta(days=i) for i in range(1, (nxt - prev).days))
        if x != d
    )
    assert (nxt - prev).days <= 6  # the longest closure since 1990 (2001) still fits a week either side
    assert len(markets.holidays(d.year)) in (8, 9, 10)  # 8 in a year whose New Year falls on a Saturday


@given(st.datetimes(min_value=dt.datetime(2008, 1, 1), max_value=dt.datetime(2040, 1, 1), timezones=st.just(dt.UTC)))
def test_offset_is_eastern(t: dt.datetime) -> None:
    assert markets.eastern_offset_hours(t) in (-4, -5)
    s = markets.last_closed_session(t)
    assert markets.close_utc(s) <= t and markets.is_trading_day(s)
