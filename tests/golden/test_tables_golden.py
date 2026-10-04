from pathlib import Path

import pdfplumber
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from pramaan.data.pdf import extract_page
from pramaan.data.tables import detect_unit, parse_table

SEDEMAC_LINES = [
    ("I Revenue from operations", ["309.77", "287.71", "217.36", "1,058.38"]),
    ("II Other income", ["0.86", "0.63", "2.60", "5.27"]),
    ("III Total income (I+II)", ["310.63", "288.34", "219.96", "1,063.65"]),
    ("Changes in inventories of finished goods and work-in-", None),
    ("", ["(6.28)", "1.94", "(8.51)", "(16.30)"]),
    ("progress", None),
    ("Current tax (refer note 3)", ["4.72", "7.23", "5.00", "24.66"]),
    ("Profit for the period/year (V-VI)", ["33.31", "32.08", "17.07", "103.58"]),
    ("-Basic (INR)", ["7.54", "7.32", "4.02", "23.91"]),
]


def make_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica", 9)
    c.drawString(60, h - 60, "STATEMENT OF UNAUDITED FINANCIAL RESULTS FOR THE QUARTER ENDED 30 JUNE 2026")
    c.drawString(60, h - 76, "(INR in Crores)")
    xs = [330, 395, 460, 525]
    c.drawString(xs[1] - 10, h - 100, "Quarter Ended")
    c.drawString(xs[3] - 10, h - 100, "Year Ended")
    for x, d in zip(xs, ["30 June 2026", "31 March 2026", "30 June 2025", "31 March 2026"], strict=True):
        c.drawRightString(x + 60, h - 114, d)
    c.drawString(60, h - 114, "Particulars")
    y = h - 140
    for label, vals in SEDEMAC_LINES:
        if label:
            c.drawString(60, y, label)
        if vals:
            for x, v in zip(xs, vals, strict=True):
                c.drawRightString(x + 60, y, v)
        y -= 14
    c.save()


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    p = tmp_path_factory.mktemp("pdf") / "t.pdf"
    make_pdf(p)
    with pdfplumber.open(str(p)) as pdf:
        pd = extract_page(pdf.pages[0], 1)
    return parse_table(pd.words, pd.width)


def test_unit_and_columns(table):
    assert table.unit == "crore"
    periods = [str(c.period) for c in table.columns]
    assert periods == ["2026-06-30/3m", "2026-03-31/3m", "2025-06-30/3m", "2026-03-31/12m"]


def test_rows_values_and_wrapped_label(table):
    rows = {r.item: r for r in table.rows if r.item}
    assert rows["revenue_from_operations"].values == ["309.77", "287.71", "217.36", "1,058.38"]
    assert rows["changes_in_inventories"].values[0] == "(6.28)"
    assert "progress" in rows["changes_in_inventories"].label
    assert rows["current_tax"].values[0] == "4.72"
    assert rows["basic_eps"].values[3] == "23.91"
    assert rows["net_profit_for_period"].exact


@pytest.mark.parametrize(
    "text,unit",
    [
        ("(INR in Crores)", "crore"),
        ("(₹ in lakhs)", "lakh"),
        ("Rs. in Million", "million"),
        ("(Rs. In Lacs)", "lakh"),
        ("figures in ₹ crore", "crore"),
        ("nothing here", None),
    ],
)
def test_unit_detection(text, unit):
    assert detect_unit(text)[0] == unit
