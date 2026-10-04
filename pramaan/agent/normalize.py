from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from pramaan.core.config import line_items
from pramaan.core.numbers import is_dash, parse_number
from pramaan.core.periods import PeriodKey
from pramaan.core.schemas import TableRead
from pramaan.data.layout import Word
from pramaan.data.tables import UNIT_FACTOR_CRORE, detect_unit


@dataclass
class NColumn:
    index: int
    period: PeriodKey | None
    scope: str | None
    header: str = ""


@dataclass
class NRow:
    label: str
    item: str | None
    exact: bool
    raw: list[str | None]
    box: list[float] | None = None
    cell_boxes: list[list[float] | None] = field(default_factory=list)


@dataclass
class NormTable:
    page_id: str
    unit: str | None
    columns: list[NColumn]
    rows: list[NRow]
    source: str
    page_scope: str | None = None
    unit_text: str = ""

    def n_items(self) -> int:
        return len({r.item for r in self.rows if r.item})


@dataclass
class Selection:
    value: float
    raw: str
    row: NRow
    col: NColumn
    box: list[float] | None
    unit: str | None
    per_share: bool
    n_candidates: int


def parse_period(s: str | None) -> PeriodKey | None:
    if not s:
        return None
    try:
        return PeriodKey.parse(s)
    except ValueError:
        return None


def from_parsed_json(page_id: str, tbl: dict, page_scope: str | None) -> NormTable:
    cols = [NColumn(c["index"], parse_period(c.get("period")), c.get("scope"), "") for c in tbl.get("columns", [])]
    rows = [
        NRow(r["label"], r.get("item"), bool(r.get("exact")), list(r["values"]), r.get("box"), list(r.get("boxes", [])))
        for r in tbl.get("rows", [])
    ]
    return NormTable(page_id, tbl.get("unit"), cols, rows, "rules", page_scope, tbl.get("unit_text") or "")


def _unit_from_read(tr: TableRead) -> str | None:
    if tr.unit:
        u = tr.unit.strip().lower().rstrip("s")
        mapping = {"crore": "crore", "cr": "crore", "lakh": "lakh", "lac": "lakh", "million": "million", "mn": "million", "thousand": "thousand", "rupee": "rupee"}
        if u in mapping:
            return mapping[u]
    if tr.unit_header:
        unit, _ = detect_unit(tr.unit_header)
        if unit:
            return unit
        low = tr.unit_header.lower()
        for k in ("crore", "lakh", "lac", "million", "thousand"):
            if k in low:
                return {"lac": "lakh"}.get(k, k)
    return None


def from_table_read(page_id: str, tr: TableRead, words: list[Word] | None, page_scope: str | None) -> NormTable:
    li = line_items()
    cols: list[NColumn] = []
    for i, c in enumerate(tr.columns):
        pk: PeriodKey | None = None
        if c.period_end and c.months:
            try:
                pk = PeriodKey(date.fromisoformat(c.period_end[:10]), int(c.months))
            except ValueError:
                pk = None
        cols.append(NColumn(i, pk, c.scope, c.header))
    rows: list[NRow] = []
    for r in tr.rows:
        item, exact = li.match_label_ex(r.label)
        raw = [None if v is None else str(v).strip() for v in r.values]
        box = r.box
        cell_boxes: list[list[float] | None] = [None] * len(raw)
        if words:
            lbl = r.label.split()[0] if r.label.split() else ""
            anchor = next((w for w in words if w.text.lower().startswith(lbl.lower()[:6]) and lbl), None) if lbl else None
            for j, v in enumerate(raw):
                if v is None:
                    continue
                cands = [w for w in words if w.text == v]
                if anchor is not None and cands:
                    cands.sort(key=lambda w: abs(w.cy - anchor.cy))
                if cands:
                    w = cands[0]
                    cell_boxes[j] = [w.x0, w.y0, w.x1, w.y1]
        rows.append(NRow(r.label, item, exact, raw, box, cell_boxes))
    return NormTable(page_id, _unit_from_read(tr), cols, rows, "llm", tr.scope or page_scope, tr.unit_header or "")


def _value_of(raw: str | None) -> float | None:
    if raw is None:
        return None
    if is_dash(raw):
        return 0.0
    return parse_number(re.sub(r"[^\d,.()\-−–%]", "", raw))


def select_value(nt: NormTable, item: str, period: PeriodKey, scope: str | None) -> Selection | None:
    li = line_items()
    per_share = li.kind(item) == "per_share"
    cols = [c for c in nt.columns if c.period == period and (scope is None or c.scope in (None, scope))]
    if not cols:
        return None
    rows = [r for r in nt.rows if r.item == item]
    if not rows:
        return None
    rows.sort(key=lambda r: (not r.exact, nt.rows.index(r)))
    best: Selection | None = None
    for r in rows:
        for c in cols:
            if c.index >= len(r.raw):
                continue
            raw = r.raw[c.index]
            v = _value_of(raw)
            if v is None or raw is None:
                continue
            box = r.cell_boxes[c.index] if c.index < len(r.cell_boxes) else None
            sel = Selection(v, raw, r, c, list(box) if box else (list(r.box) if r.box else None), nt.unit, per_share, len(rows) * len(cols))
            best = best or sel
            if r.exact:
                return sel
    return best


def to_answer_units(sel: Selection, unit: str | None) -> float | None:
    if sel.per_share:
        return sel.value
    f = UNIT_FACTOR_CRORE.get(unit or "")
    if f is None:
        return None
    return sel.value * f
