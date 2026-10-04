import pytest

from pramaan.core.config import line_items


@pytest.mark.parametrize(
    "label,item",
    [
        ("Revenue from operations", "revenue_from_operations"),
        ("I Revenue from operations", "revenue_from_operations"),
        ("Total income (I+II)", "total_income"),
        ("Profit before tax (III-IV)", "profit_before_tax"),
        ("Profit for the period/year (V-VI)", "net_profit_for_period"),
        ("Profit for the quarter / year", "net_profit_for_period"),
        ("Total comprehensive income for the quarter / year", "total_comprehensive_income"),
        ("Employee benefits expense", "employee_benefits_expense"),
        ("Depreciation and amortization expense", "depreciation_and_amortisation"),
        ("-Basic (INR)", "basic_eps"),
        ("-Diluted (INR)", "diluted_eps"),
    ],
)
def test_labels_map(label, item):
    assert line_items().match_label(label) == item


@pytest.mark.parametrize(
    "label",
    [
        "Sales and marketing expenses",
        "Short/(Excess) provision in respect of earlier years",
        "Profit attributable to owners",
        "Other equity",
        "Network operating expenses",
    ],
)
def test_labels_do_not_overmatch(label):
    assert line_items().match_label(label) is None
