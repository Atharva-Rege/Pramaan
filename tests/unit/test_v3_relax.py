from types import SimpleNamespace

import pytest

from pramaan.verify.checks import v3_units


def _ctx(unit, pages, relax):
    values = {
        f"v{i}": SimpleNamespace(found=True, item="revenue_from_operations", page_id=p, page_unit=None, value=100.0)
        for i, p in enumerate(pages)
    }
    return SimpleNamespace(values=values, answer=SimpleNamespace(unit=unit, inputs=[]), flags={"relax_ratio_units": relax})


def test_strict_rule_is_the_default():
    assert v3_units(_ctx("percent", ["p1", "p1"], relax=False)).status == "hard_fail"


def test_percent_from_one_page_is_not_failed_when_relaxed():
    assert v3_units(_ctx("percent", ["p1", "p1"], relax=True)).status == "soft_fail"


@pytest.mark.parametrize(
    "unit,pages",
    [
        ("percent", ["p1", "p2"]),  # inputs from different pages may carry different units
        ("INR_crore", ["p1"]),  # an absolute value needs its unit
        ("INR_per_share", ["p1"]),
    ],
)
def test_relaxation_does_not_cover_other_cases(unit, pages):
    assert v3_units(_ctx(unit, pages, relax=True)).status == "hard_fail"
