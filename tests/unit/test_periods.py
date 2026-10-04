from datetime import date

import pytest

from pramaan.core.periods import (
    PeriodKey,
    duration_keywords,
    fiscal_quarter_label,
    fiscal_year_label,
    parse_dates,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30 June 2026", date(2026, 6, 30)),
        ("30.06.2026", date(2026, 6, 30)),
        ("30-06-2026", date(2026, 6, 30)),
        ("30/06/2026", date(2026, 6, 30)),
        ("June 30, 2026", date(2026, 6, 30)),
        ("30th June, 2026", date(2026, 6, 30)),
        ("30-Jun-26", date(2026, 6, 30)),
        ("Jun-26", date(2026, 6, 30)),
        ("2026-06-30", date(2026, 6, 30)),
        ("31st March 2026", date(2026, 3, 31)),
    ],
)
def test_parse_dates(text, expected):
    found = parse_dates(text)
    assert found and found[0].value == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30 sepember 2025", date(2025, 9, 30)),  # misspelling seen in a real filing (PREMIERENE)
        ("Sepember 30, 2025", date(2025, 9, 30)),
        ("Sepember-25", date(2025, 9, 30)),
    ],
)
def test_misspelled_month_does_not_raise(text, expected):
    found = parse_dates(text)
    assert found and found[0].value == expected


def test_no_false_year_from_next_date():
    assert parse_dates("30 June 31 March 30 June 31 March") == []


def test_period_key_roundtrip_and_shift():
    k = PeriodKey(date(2026, 6, 30), 3)
    assert str(k) == "2026-06-30/3m"
    assert PeriodKey.parse(str(k)) == k
    assert k.shift_years(-1) == PeriodKey(date(2025, 6, 30), 3)
    assert k.start == date(2026, 4, 1)
    with pytest.raises(ValueError):
        PeriodKey.parse("bad")


def test_duration_keywords():
    kws = duration_keywords("Quarter Ended Year Ended")
    assert [m for _, _, m in kws] == [3, 12]
    assert duration_keywords("Three months period ended")[0][2] == 3
    assert duration_keywords("Half year ended")[0][2] == 6
    assert duration_keywords("Nine months ended")[0][2] == 9


def test_fiscal_labels():
    assert fiscal_year_label(date(2026, 6, 30)) == "FY27"
    assert fiscal_year_label(date(2026, 3, 31)) == "FY26"
    assert fiscal_quarter_label(date(2026, 9, 30)) == "Q2 FY27"
