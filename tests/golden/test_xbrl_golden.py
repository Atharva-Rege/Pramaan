from pathlib import Path

import pytest
from lxml import etree

from pramaan.data.xbrl import parse_xbrl

FIX = Path(__file__).parent / "fixtures"
FILES = sorted(FIX.glob("*.xml"))
NS = {"x": "http://www.xbrl.org/2003/instance"}


def raw_value(path: Path, local: str, ctx: str = "OneD") -> float | None:
    root = etree.parse(str(path)).getroot()
    for el in root:
        if isinstance(el.tag, str) and etree.QName(el).localname == local and el.get("contextRef") == ctx:
            try:
                return float(el.text)
            except (TypeError, ValueError):
                return None
    return None


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_parser_matches_raw_elements(path):
    parsed = parse_xbrl(path.read_bytes())
    assert parsed.meta["scope"] in ("standalone", "consolidated")
    assert parsed.meta["period_end"] is not None
    for item, el in (("revenue_from_operations", "RevenueFromOperations"), ("profit_before_tax", "ProfitBeforeTax"), ("total_expenses", "Expenses")):
        want = raw_value(path, el)
        fact = parsed.fact(item)
        if want is None:
            assert fact is None
        else:
            assert fact is not None and fact.value == want
            assert fact.duration_months in (3, 6, 9, 12)


def test_ten_real_filings_present():
    assert len(FILES) >= 10


def test_known_sedemac_values():
    p = parse_xbrl((FIX / "sedemac_standalone.xml").read_bytes())
    f = p.fact("revenue_from_operations")
    assert f is not None and f.value == 3097700000.0 and f.decimals == -5
    assert p.meta["rounding"] == "Crores" and p.meta["scope"] == "standalone"
    assert p.fact("basic_eps").value == 7.54
