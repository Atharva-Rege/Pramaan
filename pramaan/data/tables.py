from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from pramaan.core.config import line_items
from pramaan.core.numbers import is_dash, parse_number
from pramaan.core.periods import (
    DateMatch,
    PeriodKey,
    duration_keywords,
    fiscal_ytd_months,
    parse_dates,
)
from pramaan.data.layout import Line, Word, group_lines

_VALUE_WORD = re.compile(
    r"^(?:\([-−–]?\d[\d,]*(?:\.\d+)?\)|[-−–]?\d[\d,]*(?:\.\d+)?%?|\.\d+)$|^[-–—]+$"
)
_FORMATTED = re.compile(r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{2,3})+")
_UNIT_PATTERNS = [
    re.compile(r"\bin\s+(?:rs\.?\s*|inr\s*|₹\s*|rupees\s*)?(crores?|cr\b|lakhs?|lacs?|lakh|millions?|mn\b|thousands?|'000s?)", re.I),
    re.compile(r"(?:rs\.?|inr|₹|`|rupees?)\s*\(?\s*(?:in\s*)?(crores?|lakhs?|lacs?|millions?|thousands?)\b", re.I),
    re.compile(r"\(\s*(?:₹|`|rs\.?|inr)?\s*(crores?|lakhs?|lacs?|millions?)\s*\)", re.I),
    re.compile(r"\b(crores?|lakhs?|lacs?)\s*\)", re.I),
]
_UNIT_NORMAL = {
    "crore": "crore",
    "crores": "crore",
    "cr": "crore",
    "lakh": "lakh",
    "lakhs": "lakh",
    "lac": "lakh",
    "lacs": "lakh",
    "million": "million",
    "millions": "million",
    "mn": "million",
    "thousand": "thousand",
    "thousands": "thousand",
    "000": "thousand",
    "000s": "thousand",
}
UNIT_FACTOR_CRORE = {"crore": 1.0, "lakh": 0.01, "million": 0.1, "thousand": 0.0001, "rupee": 1e-7}
UNIT_INR = {"crore": 1e7, "lakh": 1e5, "million": 1e6, "thousand": 1e3, "rupee": 1.0}

_SERIAL = re.compile(r"^\s*(?:\(?[ivxlcIVXLC]{1,5}\)|[ivxlcIVXLC]{1,5}\.?|\d{1,2}\.?|\(?[a-zA-Z]\.?\)\.?|[a-z]\))\s+")
_SERIAL_ONLY = re.compile(r"^\s*(?:\(?[ivxlcIVXLC]{1,5}\)?\.?|\d{1,2}\.?|\(?[a-zA-Z]\.?\)\.?)\s*$")
_UNFINISHED = re.compile(r"(?:[-/,(]|\b(?:of|and|the|for|in|to|on|from|per|by|with|or|at|as|an|a|net|before|after)\b)\s*$", re.I)


@dataclass
class Cell:
    text: str
    value: float | None
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class ParsedRow:
    label: str
    box: tuple[float, float, float, float]
    section: str = ""
    item: str | None = None
    exact: bool = False
    values: list[str | None] = field(default_factory=list)
    cell_boxes: list[tuple[float, float, float, float] | None] = field(default_factory=list)

    @property
    def n_values(self) -> int:
        return sum(1 for v in self.values if v is not None)


@dataclass
class ParsedColumn:
    index: int
    x: float
    header: str = ""
    period: PeriodKey | None = None
    scope: str | None = None
    inferred: bool = False


@dataclass
class ParsedTable:
    rows: list[ParsedRow] = field(default_factory=list)
    columns: list[ParsedColumn] = field(default_factory=list)
    unit: str | None = None
    unit_text: str = ""
    title: str = ""
    header_text: str = ""

    @property
    def unit_factor_crore(self) -> float | None:
        return UNIT_FACTOR_CRORE.get(self.unit) if self.unit else None

    @property
    def unit_inr(self) -> float | None:
        return UNIT_INR.get(self.unit) if self.unit else None

    def column_for(self, key: PeriodKey, scope: str | None = None) -> ParsedColumn | None:
        for c in self.columns:
            if c.period == key and (scope is None or c.scope is None or c.scope == scope):
                return c
        return None

    def rows_for_item(self, item: str) -> list[ParsedRow]:
        return [r for r in self.rows if r.item == item]


def detect_unit(text: str) -> tuple[str | None, str]:
    for pat in _UNIT_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).lower().rstrip(".").strip("'")
            unit = _UNIT_NORMAL.get(raw)
            if unit:
                return unit, m.group(0).strip()
    return None, ""


def is_value_word(t: str) -> bool:
    return bool(_VALUE_WORD.match(t))


def split_line(line: Line) -> tuple[list[Word], list[Word]]:
    words = line.words
    i = len(words)
    while i > 0 and is_value_word(words[i - 1].text):
        i -= 1
    label_words, value_words = words[:i], words[i:]
    if value_words and not any(not is_dash(w.text) for w in value_words):
        if len(value_words) < 2:
            return words, []
    return label_words, value_words


def _mk_cells(words: list[Word]) -> list[Cell]:
    return [Cell(w.text, 0.0 if is_dash(w.text) else parse_number(w.text), w.x0, w.x1, w.y0, w.y1) for w in words]


def strip_serial(label: str) -> str:
    s = label.strip()
    for _ in range(2):
        if len(s.split()) > 1 and _SERIAL.match(s):
            s = _SERIAL.sub("", s, count=1).strip()
    return s


def _cluster_positions(values: list[float], tol: float) -> list[tuple[float, int]]:
    if not values:
        return []
    vals = sorted(values)
    clusters: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def _date_positions(ln: Line) -> list[tuple[DateMatch, float]]:
    text = ""
    spans: list[tuple[int, int, Word]] = []
    for w in ln.words:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w.text), w))
        text += w.text
    out: list[tuple[DateMatch, float]] = []
    for dm in parse_dates(text):
        covered = [w for a, b, w in spans if a < dm.end and b > dm.start]
        if covered:
            out.append((dm, sum(w.cx for w in covered) / len(covered)))
    return out


def _keyword_positions(ln: Line) -> list[tuple[int | None, float]]:
    text = ""
    spans: list[tuple[int, int, Word]] = []
    for w in ln.words:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w.text), w))
        text += w.text
    out: list[tuple[int | None, float]] = []
    for a, b, months in duration_keywords(text):
        covered = [w for s, e, w in spans if s < b and e > a]
        if covered:
            out.append((months, sum(w.cx for w in covered) / len(covered)))
    return out


_DAY_MONTH = re.compile(
    r"^(?P<d>\d{1,2})(?:st|nd|rd|th)?$",
    re.I,
)
_MONTH_WORD = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?$", re.I)
_YEAR_WORD = re.compile(r"^(?:19|20)\d{2}$")


def reassemble_split_dates(window: list[Line]) -> list[Line]:
    out = list(window)
    for k, ln in enumerate(window):
        if len(_date_positions(ln)) >= 2:
            continue
        frags: list[tuple[Word, Word]] = []
        for a, b in zip(ln.words, ln.words[1:], strict=False):
            if _DAY_MONTH.match(a.text) and _MONTH_WORD.match(b.text):
                nxt = ln.words[ln.words.index(b) + 1] if ln.words.index(b) + 1 < len(ln.words) else None
                if nxt is not None and _YEAR_WORD.match(nxt.text):
                    continue
                frags.append((a, b))
        if len(frags) < 2:
            continue
        years = [w for later in window[k + 1 : k + 4] for w in later.words if _YEAR_WORD.match(w.text)]
        synth: list[Word] = []
        used: set[int] = set()
        for a, b in frags:
            cx = (a.x0 + b.x1) / 2
            cand = [(abs(y.cx - cx), i, y) for i, y in enumerate(years) if i not in used and abs(y.cx - cx) <= 40]
            if not cand:
                continue
            _, i, y = min(cand)
            used.add(i)
            synth.append(Word(f"{a.text} {b.text} {y.text}", a.x0, a.y0, b.x1, b.y1))
        if len(synth) >= 2:
            keep = [w for w in ln.words if all(w is not a and w is not b for a, b in frags)]
            out[k] = Line(sorted(keep + synth, key=lambda w: w.x0))
            return out
    return out


def header_region(lines: list[Line], first_data: int) -> list[Line]:
    window = reassemble_split_dates(lines[max(0, first_data - 10) : first_data])
    date_rows = [k for k, ln in enumerate(window) if len(_date_positions(ln)) >= 2]
    if not date_rows:
        return window
    k0 = date_rows[0]
    above = [
        ln
        for ln in window[max(0, k0 - 5) : k0]
        if not _date_positions(ln) and (duration_keywords(ln.text) or _scope_words([ln]))
    ]
    return [*above, *window[k0:]]


def _scope_words(header: list[Line]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for ln in header:
        for w in ln.words:
            t = w.text.lower().strip(":|-")
            if t in {"standalone", "consolidated"}:
                out.append((t, w.cx))
    return out


def _assign_block(
    cols: list[ParsedColumn],
    dates: list[tuple[date, float]],
    kws: list[tuple[int | None, float]],
) -> None:
    n = len(cols)
    if not dates or not n:
        return
    dates = sorted(dates, key=lambda t: t[1])
    assigned: list[tuple[date | None, float]]
    if len(dates) == n:
        assigned = [(d, x) for d, x in dates]
    else:
        assigned = []
        for c in cols:
            best = min(dates, key=lambda d: abs(d[1] - c.x))
            assigned.append((best[0], best[1]) if abs(best[1] - c.x) <= 55 else (None, c.x))
    kws = sorted(kws, key=lambda t: t[1])
    named = [k for k in kws if k[0] is not None]
    ytd = [k for k in kws if k[0] is None]
    months_for: list[int | None] = [None] * n
    if named:
        groups: list[list[int]] = [[0]]
        for i in range(1, n):
            prev, cur = assigned[i - 1][0], assigned[i][0]
            if prev is not None and cur is not None and cur > prev:
                groups.append([i])
            else:
                groups[-1].append(i)
        if len(groups) == len(named):
            for g, (m, _) in zip(groups, named, strict=True):
                for i in g:
                    months_for[i] = m
        else:
            for i in range(n):
                months_for[i] = min(named, key=lambda k: abs(k[1] - cols[i].x))[0]
    for i, c in enumerate(cols):
        d = assigned[i][0]
        if d is None:
            continue
        months = months_for[i]
        if months is None:
            months = fiscal_ytd_months(d) if ytd and not named else 3
        c.period = PeriodKey(d, months)


def assign_columns(header: list[Line], col_x: list[float]) -> list[ParsedColumn]:
    cols = [ParsedColumn(i, x) for i, x in enumerate(col_x)]
    header_text = " | ".join(ln.text for ln in header)
    for c in cols:
        c.header = header_text
    dates: list[tuple[date, float]] = []
    kws: list[tuple[int | None, float]] = []
    for ln in header:
        for dm, cx in _date_positions(ln):
            dates.append((dm.value, cx))
        kws.extend(_keyword_positions(ln))
    scope_words = _scope_words(header)
    kinds = {k for k, _ in scope_words}
    if len(kinds) == 2 and len(cols) >= 4:
        centers = sorted(scope_words, key=lambda t: t[1])

        def block_of(x: float) -> int:
            return min(range(len(centers)), key=lambda k: abs(centers[k][1] - x))

        for b in range(len(centers)):
            bcols = [c for c in cols if block_of(c.x) == b]
            for c in bcols:
                c.scope = centers[b][0]
            _assign_block(
                bcols,
                [d for d in dates if block_of(d[1]) == b],
                [k for k in kws if block_of(k[1]) == b],
            )
        return cols
    _assign_block(cols, dates, kws)
    return cols


def _line_has_formatted_number(ln: Line, vw: list[Word]) -> bool:
    return any(_FORMATTED.search(w.text) for w in vw)


def _is_header_line(ln: Line, vw: list[Word]) -> bool:
    has_meta = bool(parse_dates(ln.text)) or bool(duration_keywords(ln.text))
    return has_meta and not _line_has_formatted_number(ln, vw)


def _unfinished(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.count("(") > t.count(")"):
        return True
    return bool(_UNFINISHED.search(t))


def _looks_like_tail(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t[0].islower() or (t[0] == "(" and len(t.split()) <= 4) or len(t.split()) <= 2 and t[0] in "-/"


def parse_table(words: list[Word], page_width: float = 595.0) -> ParsedTable:
    li = line_items()
    lines = group_lines(words)
    table = ParsedTable()
    if not lines:
        return table
    table.unit, table.unit_text = detect_unit("\n".join(ln.text for ln in lines))

    split: list[tuple[list[Word], list[Word]]] = []
    for ln in lines:
        lw, vw = split_line(ln)
        if _is_header_line(ln, vw):
            lw, vw = ln.words, []
        split.append((lw, vw))

    first_data = next(
        (i for i, (_, vw) in enumerate(split) if len(vw) >= 2 and _line_has_formatted_number(lines[i], vw)),
        None,
    )
    if first_data is None:
        return table

    data_rows = [(i, vw) for i, (_, vw) in enumerate(split) if i >= first_data and len(vw) >= 2 and any(_FORMATTED.search(w.text) for w in vw)]
    counts: dict[int, int] = {}
    for _, vw in data_rows:
        counts[len(vw)] = counts.get(len(vw), 0) + 1
    mode_n = max(counts, key=lambda k: (counts[k], k))
    full_rows = [vw for _, vw in data_rows if len(vw) == mode_n]
    right_x = [sorted(r[k].x1 for r in full_rows)[len(full_rows) // 2] for k in range(mode_n)]
    center_x = [sorted(r[k].cx for r in full_rows)[len(full_rows) // 2] for k in range(mode_n)]
    spread_r = sum(max(r[k].x1 for r in full_rows) - min(r[k].x1 for r in full_rows) for k in range(mode_n))
    spread_c = sum(max(r[k].cx for r in full_rows) - min(r[k].cx for r in full_rows) for k in range(mode_n))
    use_right = spread_r <= spread_c
    col_x = right_x if use_right else center_x

    table.columns = assign_columns(header_region(lines, first_data), col_x)
    table.header_text = table.columns[0].header if table.columns else ""
    table.title = " ".join(ln.text for ln in lines[: max(0, first_data)] if len(ln.text) > 12)[-400:]

    n_cols = len(table.columns)
    rows: list[ParsedRow] = []
    buffer: tuple[str, float, float, float, float] | None = None
    section = ""
    consumed = [False] * len(lines)

    def ref_x(w: Word) -> float:
        return w.x1 if use_right else w.cx

    for i in range(first_data, len(lines)):
        if consumed[i]:
            continue
        ln = lines[i]
        lw, vw = split[i]
        label_raw = " ".join(w.text for w in lw).strip()
        label_own = strip_serial(label_raw)
        if not vw:
            if not label_raw:
                continue
            prev_ln = lines[i - 1] if i > first_data else None
            close = prev_ln is not None and ln.y0 - prev_ln.y1 <= 0.9 * max(ln.y1 - ln.y0, 6.0)
            if rows and close and _looks_like_tail(label_raw) and rows[-1].n_values > 0 and not (buffer and buffer[0]):
                r = rows[-1]
                r.label = f"{r.label} {label_own}".strip()
                r.box = (min(r.box[0], ln.x0), r.box[1], max(r.box[2], ln.x1), ln.y1)
                continue
            if buffer and _unfinished(buffer[0]) and close:
                text = f"{buffer[0]} {label_own}".strip()
                buffer = (text, min(buffer[1], ln.x0), buffer[2], max(buffer[3], ln.x1), ln.y1)
            else:
                if buffer and buffer[0] and len(buffer[0]) < 70:
                    section = buffer[0]
                buffer = (label_own, ln.x0, ln.y0, ln.x1, ln.y1)
            continue
        if n_cols == 0:
            continue
        aligned = [w for w in vw if min(abs(c.x - ref_x(w)) for c in table.columns) <= 22.0]
        if len(vw) == 1 and not aligned:
            continue
        box = (ln.x0, ln.y0, ln.x1, ln.y1)
        label = label_own
        empty_label = (not label) or bool(_SERIAL_ONLY.match(label_raw)) or len(label) <= 1
        if empty_label:
            label = ""
        prev_ln = lines[i - 1] if i > first_data else None
        close = prev_ln is not None and ln.y0 - prev_ln.y1 <= 1.6 * max(ln.y1 - ln.y0, 6.0)
        if buffer and close and (empty_label or _unfinished(buffer[0])):
            label = f"{buffer[0]} {label}".strip()
            box = (min(box[0], buffer[1]), min(box[1], buffer[2]), max(box[2], buffer[3]), box[3])
            if empty_label:
                nxt = i + 1
                if nxt < len(lines) and not split[nxt][1]:
                    nxt_text = " ".join(w.text for w in split[nxt][0]).strip()
                    if nxt_text and _looks_like_tail(nxt_text) and lines[nxt].y0 - ln.y1 <= 1.2 * (ln.y1 - ln.y0):
                        label = f"{label} {strip_serial(nxt_text)}".strip()
                        box = (box[0], box[1], max(box[2], lines[nxt].x1), lines[nxt].y1)
                        consumed[nxt] = True
        elif empty_label:
            nxt = i + 1
            if nxt < len(lines) and not split[nxt][1]:
                nxt_text = " ".join(w.text for w in split[nxt][0]).strip()
                if nxt_text:
                    label = strip_serial(nxt_text)
                    consumed[nxt] = True
        buffer = None
        values: list[str | None] = [None] * n_cols
        boxes: list[tuple[float, float, float, float] | None] = [None] * n_cols
        for w in vw:
            j = min(range(n_cols), key=lambda k: abs(table.columns[k].x - ref_x(w)))
            if len(vw) == n_cols:
                j = vw.index(w)
            if values[j] is None:
                values[j] = w.text
                boxes[j] = (w.x0, w.y0, w.x1, w.y1)
        rows.append(ParsedRow(label=label, box=box, section=section, values=values, cell_boxes=boxes))

    for r in rows:
        r.item, r.exact = li.match_label_ex(r.label)
        if r.item is None and r.section and len(r.label.split()) <= 3:
            r.item, r.exact = li.match_label_ex(f"{r.section} {r.label}")
    table.rows = rows
    return table


def table_signature(table: ParsedTable) -> dict[str, int]:
    return {
        "rows": len(table.rows),
        "columns": len(table.columns),
        "items": len({r.item for r in table.rows if r.item}),
    }


def _layouts(period_end: date) -> dict[int, list[PeriodKey]]:
    from pramaan.core.periods import previous_quarter_end

    pe = period_end
    pq = previous_quarter_end(pe)
    yoy = pe.replace(year=pe.year - 1, day=min(pe.day, 28)) if pe.month == 2 else date(pe.year - 1, pe.month, pe.day)
    fy_prev = date(pe.year, 3, 31) if pe.month != 3 else date(pe.year - 1, 3, 31)
    fy_cur = date(pe.year, 3, 31) if pe.month == 3 else None
    q = [PeriodKey(pe, 3), PeriodKey(pq, 3), PeriodKey(yoy, 3)]
    if pe.month == 6:
        return {4: [*q, PeriodKey(fy_prev, 12)]}
    if pe.month == 9:
        return {6: [*q, PeriodKey(pe, 6), PeriodKey(yoy, 6), PeriodKey(fy_prev, 12)], 5: [*q, PeriodKey(pe, 6), PeriodKey(yoy, 6)]}
    if pe.month == 12:
        return {6: [*q, PeriodKey(pe, 9), PeriodKey(yoy, 9), PeriodKey(fy_prev, 12)], 5: [*q, PeriodKey(pe, 9), PeriodKey(yoy, 9)]}
    if pe.month == 3 and fy_cur is not None:
        return {5: [*q, PeriodKey(fy_cur, 12), PeriodKey(date(fy_cur.year - 1, 3, 31), 12)], 4: [*q, PeriodKey(fy_cur, 12)]}
    return {}


def infer_columns(table: ParsedTable, period_end: date) -> bool:
    cols = table.columns
    if len(cols) < 3:
        return False
    layouts = _layouts(period_end)
    scopes = {c.scope for c in cols if c.scope}
    blocks: list[list[ParsedColumn]]
    if len(scopes) == 2:
        blocks = [[c for c in cols if c.scope == s] for s in sorted(scopes)]
    else:
        blocks = [cols]
    changed = False
    for block in blocks:
        known = [c.period for c in block if c.period]
        broken = any(c.period is None for c in block) or len(set(known)) < len(known)
        layout = layouts.get(len(block))
        if broken and layout is not None:
            for c, k in zip(block, layout, strict=True):
                c.period, c.inferred = k, True
            changed = True
    return changed
