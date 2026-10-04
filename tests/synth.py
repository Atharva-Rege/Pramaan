from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from pramaan.core.ids import sha256_bytes
from pramaan.data.collector import FilingMeta, XbrlFile

ITEMS = [
    ("revenue_from_operations", "Revenue from operations", "I"),
    ("other_income", "Other income", "II"),
    ("total_income", "Total income (I+II)", "III"),
    ("cost_of_materials_consumed", "Cost of materials consumed", ""),
    ("employee_benefits_expense", "Employee benefits expense", ""),
    ("finance_costs", "Finance costs", ""),
    ("depreciation_and_amortisation", "Depreciation and amortisation expense", ""),
    ("other_expenses", "Other expenses", ""),
    ("total_expenses", "Total expenses (IV)", ""),
    ("profit_before_tax", "Profit before tax (III-IV)", "V"),
    ("current_tax", "Current tax", ""),
    ("deferred_tax", "Deferred tax", ""),
    ("total_tax_expense", "Total tax expense", "VI"),
    ("net_profit_for_period", "Profit for the period (V-VI)", "VII"),
    ("other_comprehensive_income", "Other comprehensive income (net of tax)", "VIII"),
    ("total_comprehensive_income", "Total comprehensive income for the period", "IX"),
    ("basic_eps", "Basic EPS (INR)", ""),
    ("diluted_eps", "Diluted EPS (INR)", ""),
]
ELEMENT = {
    "revenue_from_operations": "RevenueFromOperations",
    "other_income": "OtherIncome",
    "total_income": "Income",
    "cost_of_materials_consumed": "CostOfMaterialsConsumed",
    "employee_benefits_expense": "EmployeeBenefitExpense",
    "finance_costs": "FinanceCosts",
    "depreciation_and_amortisation": "DepreciationDepletionAndAmortisationExpense",
    "other_expenses": "OtherExpenses",
    "total_expenses": "Expenses",
    "profit_before_tax": "ProfitBeforeTax",
    "current_tax": "CurrentTax",
    "deferred_tax": "DeferredTax",
    "total_tax_expense": "TaxExpense",
    "net_profit_for_period": "ProfitLossForPeriod",
    "other_comprehensive_income": "OtherComprehensiveIncomeNetOfTaxes",
    "total_comprehensive_income": "ComprehensiveIncomeForThePeriod",
    "basic_eps": "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "diluted_eps": "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
}


def statement(rng: random.Random, scale: float) -> dict[str, float]:
    rev = round(rng.uniform(300, 9000) * scale, 2)
    oth = round(rev * rng.uniform(0.002, 0.03), 2)
    mat = round(rev * rng.uniform(0.35, 0.55), 2)
    emp = round(rev * rng.uniform(0.06, 0.12), 2)
    fin = round(rev * rng.uniform(0.005, 0.03), 2)
    dep = round(rev * rng.uniform(0.02, 0.05), 2)
    oex = round(rev * rng.uniform(0.08, 0.15), 2)
    tinc = round(rev + oth, 2)
    texp = round(mat + emp + fin + dep + oex, 2)
    pbt = round(tinc - texp, 2)
    cur = round(max(pbt, 1) * 0.2, 2)
    dfr = round(max(pbt, 1) * 0.04, 2)
    tax = round(cur + dfr, 2)
    pat = round(pbt - tax, 2)
    oci = round(rng.uniform(-3, 3), 2)
    tci = round(pat + oci, 2)
    shares = rng.uniform(20, 200)
    return {
        "revenue_from_operations": rev,
        "other_income": oth,
        "total_income": tinc,
        "cost_of_materials_consumed": mat,
        "employee_benefits_expense": emp,
        "finance_costs": fin,
        "depreciation_and_amortisation": dep,
        "other_expenses": oex,
        "total_expenses": texp,
        "profit_before_tax": pbt,
        "current_tax": cur,
        "deferred_tax": dfr,
        "total_tax_expense": tax,
        "net_profit_for_period": pat,
        "other_comprehensive_income": oci,
        "total_comprehensive_income": tci,
        "basic_eps": round(pat / shares, 2),
        "diluted_eps": round(pat / shares * 0.98, 2),
    }


def fmt(v: float) -> str:
    s = f"{abs(v):,.2f}"
    return f"({s})" if v < 0 else s


def draw_page(c: canvas.Canvas, title: str, columns: list[tuple[str, int]], quarters_kw: tuple[str, str], rows: list[tuple[str, list[float]]], unit_text: str) -> None:
    w, h = A4
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 50, "SYNTHETIC COMPANY LIMITED")
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 64, "Regd. Office: Mumbai 400001, CIN: L00000MH2000PLC000000")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 90, title)
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 110, unit_text)
    xs = [300, 365, 430, 495]
    c.drawString(xs[1] - 50, h - 135, quarters_kw[0])
    c.drawString(xs[3] - 8, h - 135, quarters_kw[1])
    c.drawString(40, h - 150, "Particulars")
    for x, (d, _) in zip(xs, columns, strict=True):
        c.drawRightString(x + 55, h - 150, d)
    y = h - 175
    for label, vals in rows:
        c.drawString(40, y, label)
        for x, v in zip(xs, vals, strict=True):
            c.drawRightString(x + 55, y, fmt(v))
        y -= 14
    c.showPage()


def build_pdf(path: Path, company: str, stmts: dict[str, dict[str, list[float]]], unit: str, period_end: date) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 60, f"Outcome of the Board Meeting held on {period_end + timedelta(days=28):%d %B %Y}")
    c.drawString(40, h - 80, "Pursuant to Regulation 30 and 33 we wish to inform you that the Board approved the financial results.")
    c.showPage()
    cols = [(f"{period_end:%d.%m.%Y}", 3), (f"{date(period_end.year, 3, 31):%d.%m.%Y}", 3), (f"{period_end.replace(year=period_end.year - 1):%d.%m.%Y}", 3), (f"{date(period_end.year, 3, 31):%d.%m.%Y}", 12)]
    unit_text = {"crore": "(INR in Crores)", "lakh": "(INR in Lakhs)"}[unit]
    mult = {"crore": 1.0, "lakh": 100.0}[unit]
    for scope in ("standalone", "consolidated"):
        rows = []
        for key, label, serial in ITEMS:
            vals = stmts[scope][key]
            v = [x * mult if not key.endswith("eps") else x for x in vals]
            rows.append(((serial + " " if serial else "") + label, v))
        draw_page(c, f"STATEMENT OF UNAUDITED {scope.upper()} FINANCIAL RESULTS FOR THE QUARTER ENDED {period_end:%d %B %Y}".upper(), cols, ("Quarter Ended", "Year Ended"), rows, unit_text)
    c.save()


def xbrl_text(company: str, symbol: str, isin: str, scope: str, period_end: date, vals: dict[str, float], months: int, unit_rounding: str) -> str:
    start = date(period_end.year, period_end.month, 1)
    mm = period_end.month - months + 1
    y = period_end.year
    while mm < 1:
        mm += 12
        y -= 1
    start = date(y, mm, 1)
    facts = []
    for key, val in vals.items():
        el = ELEMENT[key]
        if key.endswith("eps"):
            facts.append(f'<in-capmkt:{el} contextRef="OneD" unitRef="INRPerShare" decimals="INF">{val}</in-capmkt:{el}>')
        else:
            facts.append(f'<in-capmkt:{el} contextRef="OneD" unitRef="INR" decimals="-5">{round(val * 1e7)}</in-capmkt:{el}>')
    meta = {
        "ScripCode": "500001",
        "Symbol": symbol,
        "ISIN": isin,
        "NameOfTheCompany": company,
        "LevelOfRounding": unit_rounding,
        "DateOfBoardMeetingWhenFinancialResultsWereApproved": str(period_end + timedelta(days=28)),
        "DateOfStartOfReportingPeriod": str(start),
        "DateOfEndOfReportingPeriod": str(period_end),
        "WhetherResultsAreAuditedOrUnaudited": "Unaudited",
        "NatureOfReportStandaloneConsolidated": scope.capitalize(),
        "ReportingQuarter": "First quarter",
    }
    mfacts = "".join(f'<in-capmkt:{k} contextRef="OneD">{v}</in-capmkt:{k}>' for k, v in meta.items())
    return (
        '<?xml version="1.0" encoding="UTF-8"?><xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:in-capmkt="http://www.sebi.gov.in/xbrl/2026-01-31/in-capmkt" xmlns:link="http://www.xbrl.org/2003/linkbase" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:iso4217="http://www.xbrl.org/2003/iso4217">'
        '<link:schemaRef xlink:href="in-capmkt-ent-2026-01-31.xsd" xlink:type="simple"/>'
        f'<xbrli:context id="OneD"><xbrli:entity><xbrli:identifier scheme="http://www.sebi.gov.in/in-capmkt/ScripCode">500001</xbrli:identifier></xbrli:entity>'
        f"<xbrli:period><xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate></xbrli:period></xbrli:context>"
        '<xbrli:unit id="INR"><xbrli:measure>iso4217:INR</xbrli:measure></xbrli:unit>'
        '<xbrli:unit id="INRPerShare"><xbrli:divide><xbrli:unitNumerator><xbrli:measure>iso4217:INR</xbrli:measure></xbrli:unitNumerator>'
        "<xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unitDenominator></xbrli:divide></xbrli:unit>"
        f"{mfacts}{''.join(facts)}</xbrli:xbrl>"
    )


def build_synth_corpus(work: Path, n: int = 8, seed: int = 5) -> dict[str, dict]:
    rng = random.Random(seed)
    target = date(2026, 6, 30)
    prior = date(2025, 6, 30)
    info: dict[str, dict] = {}
    for i in range(n):
        symbol = f"SYN{i:02d}"
        name = ["Alpha", "Bravo", "Cobalt", "Delta", "Everest", "Fusion", "Granite", "Helix", "Indus", "Jade"][i % 10] + " Industries Limited"
        isin = f"INE{i:03d}A01024"
        unit = "crore" if i % 2 == 0 else "lakh"
        scale = rng.uniform(0.5, 3.0)
        cur = {s: statement(rng, scale * (1.0 if s == "consolidated" else 0.8)) for s in ("standalone", "consolidated")}
        prev_q = {s: statement(rng, scale * 0.95) for s in cur}
        yoy = {s: statement(rng, scale * 0.85) for s in cur}
        fy = {s: statement(rng, scale * 4) for s in cur}
        stmts = {s: {k: [cur[s][k], prev_q[s][k], yoy[s][k], fy[s][k]] for k in cur[s]} for s in cur}
        board = target + timedelta(days=14 + i * 3)
        pub = datetime(board.year, board.month, board.day, 11, 30) + timedelta(minutes=17 * i)
        fid = f"nse-{symbol.lower()}-{target.isoformat()}"
        d = work / "data" / "raw" / "nse" / board.isoformat() / fid
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / "outcome.pdf"
        build_pdf(pdf_path, name, stmts, unit, target)
        xfiles = []
        for scope in ("standalone", "consolidated"):
            data = xbrl_text(name, symbol, isin, scope, target, cur[scope], 3, "Crores" if unit == "crore" else "Lakhs").encode()
            xp = d / f"xbrl_{scope}_{1000 + i}.xml"
            xp.write_bytes(data)
            xfiles.append(
                XbrlFile(scope, str(1000 + i), "Original", f"https://example.test/{xp.name}", str(xp.relative_to(work)), sha256_bytes(data), (pub + timedelta(minutes=10)).isoformat(), None, None, None, "Unaudited")
            )
        pdf_bytes = pdf_path.read_bytes()
        meta = FilingMeta(
            filing_id=fid,
            exchange="nse",
            symbol=symbol,
            isin=isin,
            company=name,
            scrip_code="500001",
            period_end=target.isoformat(),
            role="target",
            board_meeting_date=board.isoformat(),
            pdf={"url": f"https://example.test/{fid}.pdf", "path": str(pdf_path.relative_to(work)), "sha256": sha256_bytes(pdf_bytes), "bytes": len(pdf_bytes), "fetched_at": pub.isoformat()},
            pdf_published_at=pub.isoformat(),
            pdf_source="nse",
            pdf_text="synthetic",
            xbrl=xfiles,
            xbrl_published_at=(pub + timedelta(minutes=10)).isoformat(),
            status="complete",
            in_nifty200=i < n // 2,
            industry="Synthetic",
        )
        (d / "meta.json").write_text(json.dumps(asdict(meta), indent=1, default=str), encoding="utf-8")
        pd_dir = work / "data" / "raw" / "nse" / (prior + timedelta(days=30)).isoformat() / f"nse-{symbol.lower()}-{prior.isoformat()}"
        pd_dir.mkdir(parents=True, exist_ok=True)
        pdata = xbrl_text(name, symbol, isin, "consolidated", prior, yoy["consolidated"], 3, "Crores").encode()
        pxp = pd_dir / "xbrl_consolidated_2000.xml"
        pxp.write_bytes(pdata)
        pdata_s = xbrl_text(name, symbol, isin, "standalone", prior, yoy["standalone"], 3, "Crores").encode()
        pxs = pd_dir / "xbrl_standalone_2001.xml"
        pxs.write_bytes(pdata_s)
        pmeta = FilingMeta(
            filing_id=f"nse-{symbol.lower()}-{prior.isoformat()}",
            exchange="nse",
            symbol=symbol,
            isin=isin,
            company=name,
            scrip_code="500001",
            period_end=prior.isoformat(),
            role="prior_year",
            board_meeting_date=(prior + timedelta(days=30)).isoformat(),
            xbrl=[
                XbrlFile("consolidated", "2000", "Original", "https://example.test/p.xml", str(pxp.relative_to(work)), sha256_bytes(pdata), "2025-07-30T10:00:00", None, None, None, "Unaudited"),
                XbrlFile("standalone", "2001", "Original", "https://example.test/ps.xml", str(pxs.relative_to(work)), sha256_bytes(pdata_s), "2025-07-30T10:00:00", None, None, None, "Unaudited"),
            ],
            xbrl_published_at="2025-07-30T10:00:00",
            status="xbrl_only",
        )
        (pd_dir / "meta.json").write_text(json.dumps(asdict(pmeta), indent=1, default=str), encoding="utf-8")
        info[fid] = {"company": name, "symbol": symbol, "isin": isin, "unit": unit, "cur": cur, "yoy": yoy, "pub": pub}
    return info
