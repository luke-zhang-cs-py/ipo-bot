"""Regression tests for verify.py's date rule: an as-of date is written YYYY-MM-DD. Each failed before its fix. Offline.

The browser port (docs/app/ipo-core.js) is held to the same cases in test_js_parity.py.
"""
import copy
import datetime as dt
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "src"))

import verify  # noqa: E402
from test_accuracy import BLOCK  # noqa: E402


@pytest.mark.parametrize("raw", ["20261006", "2026-W41-1", "2026W411", ["2026-10-06"], "2026-02-30", "0000-10-06",
                                 "２０２６-10-06", 20261006, None, ""])
def test_only_a_yyyy_mm_dd_date_is_a_date(raw):
    assert verify._date(raw) is None


@pytest.mark.parametrize("raw, want", [("2026-10-06", dt.date(2026, 10, 6)), ("2026-10-06T09:30:00Z", dt.date(2026, 10, 6)),
                                       ("0999-12-31", dt.date(999, 12, 31))])
def test_a_yyyy_mm_dd_date_reads_its_first_ten_characters(raw, want):
    assert verify._date(raw) == want


def failed(change):
    k = copy.deepcopy(BLOCK["key_numbers"])
    change(k)
    return [n for n, ok, _ in verify.check(k) if not ok]


@pytest.mark.parametrize("raw", ["20261006", "2026-W41-1"])
def test_a_compact_or_week_date_fails_the_block_and_the_input(raw):
    assert "as_of date present" in failed(lambda k: k.update(as_of=raw))
    assert "revenue: has an as-of date" in failed(lambda k: k["inputs"]["revenue"].update(as_of=raw))
