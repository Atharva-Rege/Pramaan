import contextlib

from hypothesis import given, settings
from hypothesis import strategies as st

from pramaan.agent.calc import calculate
from pramaan.core.errors import ToolError
from pramaan.core.numbers import format_number, parse_number, round_half_up
from pramaan.core.periods import PeriodKey

amounts = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)


@settings(max_examples=300, deadline=None)
@given(amounts, st.integers(0, 3), st.sampled_from(["indian", "western", "none"]), st.sampled_from(["paren", "minus"]))
def test_format_parse_roundtrip(x, decimals, grouping, neg):
    text = format_number(x, decimals, grouping, neg)
    back = parse_number(text)
    expected = round_half_up(x, decimals)
    assert back is not None
    assert abs(back - expected) <= 10 ** (-decimals) / 2 + 1e-9 or (expected == 0 and back == 0)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=80))
def test_calculator_only_ever_returns_a_float_or_a_tool_error(expr):
    try:
        out = calculate(expr, {"a": 2.0, "b": 3.0})
    except ToolError:
        return
    assert isinstance(out, float)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.sampled_from(["a", "b", "1", "2.5", "+", "-", "*", "/", "(", ")", " ", "abs(", "round("]), max_size=14).map("".join)
)
def test_arithmetic_grammar_is_safe(expr):
    with contextlib.suppress(ToolError):
        calculate(expr, {"a": 2.0, "b": 3.0})


@given(st.integers(2020, 2035), st.integers(1, 12), st.sampled_from([3, 6, 9, 12]))
def test_period_key_roundtrip(y, m, months):
    import calendar
    from datetime import date

    k = PeriodKey(date(y, m, calendar.monthrange(y, m)[1]), months)
    assert PeriodKey.parse(str(k)) == k
