import pytest

from pramaan.agent.tools import run_facts_query, validate_select
from pramaan.core.errors import ToolError

ROWS = [{"company_id": "A", "filing_id": "f", "scope": "consolidated", "item": "revenue_from_operations", "period": "p", "value": 10.0, "unit": "INR_crore", "page_id": "f#p1", "quote": "10.00"}]


def test_select_over_extracted_only():
    assert run_facts_query(ROWS, "SELECT item, value FROM extracted")[0]["value"] == 10.0


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE extracted",
        "DELETE FROM extracted",
        "SELECT * FROM xbrl_facts",
        "SELECT * FROM read_csv('/etc/passwd')",
        "INSERT INTO extracted VALUES (1)",
        "SELECT 1; DROP TABLE extracted",
        "COPY extracted TO '/tmp/x'",
    ],
)
def test_blocked(sql):
    with pytest.raises(ToolError):
        validate_select(sql)
