from pramaan.core.numbers import (
    find_numbers,
    format_number,
    group_indian,
    group_western,
    matches_printed,
    parse_number,
    round_half_up,
)


def test_parse_basic_forms():
    assert parse_number("1,058.38") == 1058.38
    assert parse_number("(6.28)") == -6.28
    assert parse_number("-6.28") == -6.28
    assert parse_number("− 6.28") == -6.28
    assert parse_number("1,23,456.78") == 123456.78
    assert parse_number("12.5%") == 12.5
    assert parse_number("₹ 4,200") == 4200
    assert parse_number("-") is None
    assert parse_number("nil") is None
    assert parse_number("abc") is None


def test_grouping():
    assert group_indian("123456789") == "12,34,56,789"
    assert group_western("123456789") == "123,456,789"
    assert group_indian("999") == "999"
    assert group_indian("1000") == "1,000"


def test_format_number_styles():
    assert format_number(123456.789, 2, "indian") == "1,23,456.79"
    assert format_number(123456.789, 2, "western") == "123,456.79"
    assert format_number(-6.28, 2, "indian", "paren") == "(6.28)"
    assert format_number(-6.28, 2, "indian", "minus") == "-6.28"
    assert format_number(0.004, 2, "indian", "paren") == "0.00"


def test_find_numbers_tokens():
    toks = find_numbers("Revenue 1,058.38 (16.30) and 7.54 plus note")
    assert [t.value for t in toks] == [1058.38, -16.3, 7.54]
    assert toks[0].decimals == 2


def test_round_half_up():
    assert round_half_up(2.675, 2) == 2.68
    assert round_half_up(0.125, 2) == 0.13


def test_matches_printed_uses_printed_precision():
    assert matches_printed(309.77, 309.77, 2)
    assert not matches_printed(309.78, 309.77, 2)
    assert matches_printed(309.7712, 309.77, 2)
    assert matches_printed(41.4, 41, 0)
    assert not matches_printed(41.6, 41, 0)
