from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pramaan.core.config import gym_config, line_items
from pramaan.core.numbers import find_numbers, matches_printed, round_half_up
from pramaan.core.periods import PeriodKey
from pramaan.core.textmatch import edit_distance_leq1
from pramaan.data.store import jload

UNIT_CANDIDATES = (1.0, 1e3, 1e5, 1e6, 1e7)


@dataclass
class Cell:
    page_id: str
    page_no: int
    row_label: str
    row_item: str | None
    col_index: int
    col_period: str | None
    col_scope: str | None
    text: str
    value: float
    decimals: int
    box: list[float] | None
    fuzzy: bool = False
    row_exact: bool = False


@dataclass
class PageCells:
    page_id: str
    page_no: int
    scope: str | None
    unit_text: str | None
    cells: list[Cell]
    fuzzy: bool


@dataclass
class Located:
    page_id: str
    box: list[float] | None
    unit_inr: float
    approx: bool
    ambiguous: bool
    col_index: int
    row_label: str
    cell_text: str
    score: float = 0.0
    decimals: int = 2


@dataclass
class PageCalibration:
    page_id: str
    unit_inr: float | None
    matches: int
    by_unit: dict[float, int] = field(default_factory=dict)


def page_cells(page: dict[str, Any]) -> PageCells | None:
    tbl = jload(page.get("tbl"))
    if not tbl:
        return None
    cols = tbl["columns"]
    cells: list[Cell] = []
    fuzzy = bool(page.get("ocr_layer")) or page.get("text_source", "text_layer") != "text_layer"
    for row in tbl["rows"]:
        for j, text in enumerate(row["values"]):
            if text is None:
                continue
            toks = find_numbers(text)
            if text.strip() in {"-", "–", "—"}:
                value, decimals = 0.0, 0
            elif toks:
                value, decimals = toks[0].value, toks[0].decimals
            else:
                continue
            box = row["boxes"][j] if j < len(row["boxes"]) else None
            col = cols[j] if j < len(cols) else {"period": None, "scope": None}
            cells.append(
                Cell(
                    page_id=page["page_id"],
                    page_no=page["page_no"],
                    row_label=row["label"],
                    row_item=row.get("item"),
                    col_index=j,
                    col_period=col.get("period"),
                    col_scope=col.get("scope"),
                    text=text,
                    value=value,
                    decimals=decimals,
                    box=box,
                    fuzzy=fuzzy,
                    row_exact=bool(row.get("exact")),
                )
            )
    return PageCells(page["page_id"], page["page_no"], page.get("scope"), tbl.get("unit"), cells, fuzzy)


def _digits_expected(d: float, decimals: int) -> str:
    r = round_half_up(abs(d), decimals)
    return "".join(ch for ch in f"{r:.{decimals}f}" if ch.isdigit())


def cell_matches(cell: Cell, d: float, slack: float = 0.0) -> str | None:
    if matches_printed(d, cell.value, cell.decimals, slack):
        return "exact"
    if cell.fuzzy:
        got = "".join(ch for ch in cell.text if ch.isdigit())
        want = _digits_expected(d, cell.decimals)
        if (d < 0) == (cell.value < 0 or "(" in cell.text) and edit_distance_leq1(got, want) and len(want) >= 3:
            return "fuzzy"
    return None


def _scope_ok(cell: Cell, page: PageCells, scope: str) -> bool:
    if cell.col_scope:
        return cell.col_scope == scope
    return page.scope in (None, scope, "both")


def calibrate_page(
    page: PageCells,
    scope: str,
    facts: dict[str, float],
    period: PeriodKey | None,
) -> PageCalibration:
    li = line_items()
    cells = [c for c in page.cells if _scope_ok(c, page, scope)]
    by_unit: dict[float, int] = {}
    for unit in UNIT_CANDIDATES:
        n = 0
        for item, v in facts.items():
            if item in ("paid_up_equity_share_capital",):
                continue
            kind = li.kind(item)
            d = v if kind == "per_share" else v / unit
            if kind == "per_share" and unit != 1.0:
                continue
            if abs(d) < 1e-9:
                continue
            if any(cell_matches(c, d) for c in cells):
                n += 1
        by_unit[unit] = n
    best_unit = max(by_unit, key=lambda u: (by_unit[u], u))
    best_n = by_unit[best_unit]
    min_matches = int(gym_config().evidence.get("min_page_matches", 4))
    return PageCalibration(page.page_id, best_unit if best_n >= min_matches else None, best_n, by_unit)


def locate_fact(
    pages: list[PageCells],
    calibrations: dict[str, PageCalibration],
    scope: str,
    item: str,
    value_inr: float,
    period: PeriodKey | None,
) -> list[Located]:
    li = line_items()
    kind = li.kind(item)
    out: list[Located] = []
    for page in pages:
        cal = calibrations.get(page.page_id)
        if cal is None or cal.unit_inr is None:
            continue
        unit = 1.0 if kind == "per_share" else cal.unit_inr
        d = value_inr if kind == "per_share" else value_inr / unit
        cells = [c for c in page.cells if _scope_ok(c, page, scope)]
        hits: list[tuple[float, Cell, str]] = []
        for c in cells:
            how = cell_matches(c, d)
            if not how:
                continue
            score = 1.0 if how == "exact" else 0.6
            if period is not None and c.col_period == str(period):
                score += 2.0
            elif period is not None and c.col_period is not None:
                score -= 1.0
            if c.row_item == item:
                score += 1.5
            hits.append((score, c, how))
        if not hits:
            continue
        hits.sort(key=lambda t: -t[0])
        top = hits[0][0]
        best = [h for h in hits if h[0] >= top - 1e-9]
        ambiguous = len(best) > 1 and len({(h[1].row_label, h[1].col_index) for h in best}) > 1
        score, c, how = best[0]
        out.append(
            Located(
                page_id=page.page_id,
                box=c.box,
                unit_inr=unit,
                approx=how == "fuzzy",
                ambiguous=ambiguous,
                col_index=c.col_index,
                row_label=c.row_label,
                cell_text=c.text,
                score=score,
                decimals=c.decimals,
            )
        )
    out.sort(key=lambda L: -L.score)
    return out


def build_page_cells(store: Any, filing_id: str) -> list[PageCells]:
    out: list[PageCells] = []
    for p in store.q(
        "SELECT page_id, page_no, scope, text_source, ocr_layer, tbl FROM pages WHERE filing_id = ? AND (is_results_page OR page_type IN ('segment','results_statement')) ORDER BY page_no",
        [filing_id],
    ):
        pc = page_cells(p)
        if pc and pc.cells:
            out.append(pc)
    return out


def period_of(end: date, months: int) -> PeriodKey:
    return PeriodKey(end, months)


def filing_period_facts(store: Any, filing_id: str, scope: str, period: PeriodKey) -> dict[str, float]:
    rows = store.q(
        "SELECT line_item, value_inr FROM xbrl_facts WHERE filing_id = ? AND scope = ? AND is_latest AND period_end = ? AND duration_months = ?",
        [filing_id, scope, period.end, period.months],
    )
    return {r["line_item"]: r["value_inr"] for r in rows}


def compute_filing_evidence(
    store: Any,
    filing_id: str,
    period: PeriodKey,
    scopes: tuple[str, ...] = ("standalone", "consolidated"),
) -> list[dict[str, Any]]:
    pages = build_page_cells(store, filing_id)
    rows: list[dict[str, Any]] = []
    if not pages:
        return rows
    for scope in scopes:
        facts = filing_period_facts(store, filing_id, scope, period)
        if not facts:
            continue
        cals = {p.page_id: calibrate_page(p, scope, facts, period) for p in pages}
        for item, v in facts.items():
            located = locate_fact(pages, cals, scope, item, v, period)
            for k, loc in enumerate(located[:3]):
                rows.append(
                    {
                        "filing_id": filing_id,
                        "scope": scope,
                        "line_item": item,
                        "period": str(period),
                        "page_id": loc.page_id,
                        "box": None if loc.box is None else __import__("json").dumps(loc.box),
                        "unit_inr": loc.unit_inr,
                        "approx": loc.approx,
                        "ambiguous": loc.ambiguous or k > 0,
                        "col_index": loc.col_index,
                        "row_label": loc.row_label,
                        "cell_text": loc.cell_text,
                        "source_filing": filing_id,
                    }
                )
    return rows


class EvidenceIndex:
    def __init__(self, store: Any, filing_id: str) -> None:
        self.store = store
        self.filing_id = filing_id
        self.pages = build_page_cells(store, filing_id)
        self._cal: dict[tuple[str, str], dict[str, PageCalibration]] = {}
        self.conflicts: list[dict[str, Any]] = []

    def calibrations(self, scope: str, period: PeriodKey) -> dict[str, PageCalibration]:
        key = (scope, str(period))
        if key not in self._cal:
            facts = filing_period_facts(self.store, self.filing_id, scope, period)
            self._cal[key] = {p.page_id: calibrate_page(p, scope, facts, period) for p in self.pages}
        return self._cal[key]

    def locate(
        self,
        scope: str,
        item: str,
        value_inr: float,
        period: PeriodKey,
        calibration_period: PeriodKey,
    ) -> Located | None:
        if not self.pages:
            return None
        cals = self.calibrations(scope, calibration_period)
        found = locate_fact(self.pages, cals, scope, item, value_inr, period)
        for loc in found:
            if not self._label_ok(loc, item, value_inr):
                continue
            clash = self._exact_row_conflict(loc, cals, scope, item, value_inr, period)
            if clash is not None:
                self.conflicts.append({"filing_id": self.filing_id, "scope": scope, "item": item, "xbrl_value_inr": value_inr, "period": str(period), **clash})
                return None
            return loc
        return None

    def _exact_row_conflict(
        self,
        loc: Located,
        cals: dict[str, PageCalibration],
        scope: str,
        item: str,
        value_inr: float,
        period: PeriodKey,
    ) -> dict[str, Any] | None:
        page = next((p for p in self.pages if p.page_id == loc.page_id), None)
        cal = cals.get(loc.page_id)
        if page is None or cal is None or cal.unit_inr is None:
            return None
        kind = line_items().kind(item)
        d = value_inr if kind == "per_share" else value_inr / cal.unit_inr
        exact = [
            c
            for c in page.cells
            if c.row_item == item and c.row_exact and _scope_ok(c, page, scope) and c.col_period in (str(period), None)
        ]
        if not exact or any(cell_matches(c, d) for c in exact):
            return None
        c0 = exact[0]
        return {"pdf_row": c0.row_label, "pdf_cell": c0.text, "page_id": loc.page_id}

    @staticmethod
    def _label_ok(loc: Located, item: str, value_inr: float) -> bool:
        li = line_items()
        got = li.match_label(loc.row_label)
        digits = len("".join(ch for ch in loc.cell_text if ch.isdigit()))
        if got == item:
            return True
        if got is None and digits >= 5:
            return True
        return digits >= 7 and got is not None and li.kind(got) == li.kind(item)
