import pytest

from pramaan.agent.calc import calculate
from pramaan.core.errors import ToolError


def test_arithmetic():
    assert calculate("(cur - prev) / abs(prev) * 100", {"cur": 120.0, "prev": -100.0}) == pytest.approx(220.0)
    assert calculate("round(10 / 3, 2)") == 3.33
    assert calculate("-a + 2", {"a": 5}) == -3


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo x')",
        "open('/etc/passwd')",
        "().__class__",
        "a.b",
        "lambda: 1",
        "[1,2]",
        "1 if 1 else 2",
        "2 ** 9999999",
        "unknown + 1",
        "1 / 0",
        "'a' + 'b'",
        "True + 1",
    ],
)
def test_rejects_everything_but_arithmetic(expr):
    with pytest.raises(ToolError):
        calculate(expr, {"a": 1})
