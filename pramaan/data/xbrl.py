from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from lxml import etree

from pramaan.core.config import line_items
from pramaan.core.timeutil import months_between, parse_date_flexible

XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
LINK = "http://www.xbrl.org/2003/linkbase"


class XbrlError(ValueError):
    pass


@dataclass(frozen=True)
class Context:
    id: str
    start: date | None
    end: date | None
    instant: date | None
    dimensional: bool

    @property
    def period_end(self) -> date | None:
        return self.instant or self.end


@dataclass
class ParsedFact:
    line_item: str
    scope: str | None
    period_end: date
    duration_months: int
    value: float
    decimals: int | None
    raw_element: str
    context_id: str
    unit: str | None

    @property
    def value_inr(self) -> float:
        return self.value


@dataclass
class ParsedFiling:
    meta: dict[str, Any] = field(default_factory=dict)
    facts: list[ParsedFact] = field(default_factory=list)
    unmapped_numeric: dict[str, int] = field(default_factory=dict)
    n_contexts: int = 0
    n_facts: int = 0

    @property
    def scope(self) -> str | None:
        return self.meta.get("scope")

    @property
    def period_end(self) -> date | None:
        return self.meta.get("period_end")

    def fact(self, item: str, period_end: date | None = None, months: int | None = None) -> ParsedFact | None:
        for f in self.facts:
            if f.line_item != item:
                continue
            if period_end is not None and f.period_end != period_end:
                continue
            if months is not None and f.duration_months != months:
                continue
            return f
        return None

    def values_by_period(self) -> dict[tuple[date, int], dict[str, float]]:
        out: dict[tuple[date, int], dict[str, float]] = {}
        for f in self.facts:
            out.setdefault((f.period_end, f.duration_months), {})[f.line_item] = f.value
        return out


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _read_contexts(root: etree._Element) -> dict[str, Context]:
    out: dict[str, Context] = {}
    for c in root.findall(f"{{{XBRLI}}}context"):
        cid = c.get("id")
        if not cid:
            continue
        period = c.find(f"{{{XBRLI}}}period")
        start = end = instant = None
        if period is not None:
            s = period.find(f"{{{XBRLI}}}startDate")
            e = period.find(f"{{{XBRLI}}}endDate")
            i = period.find(f"{{{XBRLI}}}instant")
            start = parse_date_flexible(s.text) if s is not None and s.text else None
            end = parse_date_flexible(e.text) if e is not None and e.text else None
            instant = parse_date_flexible(i.text) if i is not None and i.text else None
        entity = c.find(f"{{{XBRLI}}}entity")
        dimensional = c.find(f"{{{XBRLI}}}scenario") is not None
        if entity is not None and entity.find(f"{{{XBRLI}}}segment") is not None:
            dimensional = True
        out[cid] = Context(cid, start, end, instant, dimensional)
    return out


def _decimals(raw: str | None) -> int | None:
    if raw is None or raw.upper() == "INF":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_NUM = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def parse_xbrl(data: bytes) -> ParsedFiling:
    li = line_items()
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True, recover=False, no_network=True)
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise XbrlError(f"not well-formed XML: {exc}") from exc
    if _local(root.tag) != "xbrl":
        raise XbrlError(f"root element is {root.tag!r}, not xbrl")

    contexts = _read_contexts(root)
    out = ParsedFiling(n_contexts=len(contexts))

    text_facts: dict[str, str] = {}
    numeric: list[tuple[str, etree._Element, Context]] = []
    for el in root:
        if not isinstance(el.tag, str):
            continue
        ns = el.tag[1:].split("}", 1)[0] if el.tag.startswith("{") else ""
        if ns in (XBRLI, LINK):
            continue
        out.n_facts += 1
        ctx_ref = el.get("contextRef")
        if not ctx_ref or ctx_ref not in contexts:
            continue
        ctx = contexts[ctx_ref]
        name = _local(el.tag)
        text = (el.text or "").strip()
        if ctx.dimensional:
            continue
        if el.get("unitRef") is None:
            if name not in text_facts and text:
                text_facts[name] = text
            continue
        if _NUM.match(text):
            numeric.append((name, el, ctx))

    m = li.meta
    scope = li.scope_from_xbrl(text_facts.get(m["scope"]))
    out.meta = {
        "scope": scope,
        "audited": text_facts.get(m["audited"]),
        "rounding": text_facts.get(m["rounding"]),
        "board_meeting_date": parse_date_flexible(text_facts.get(m["board_meeting_date"])),
        "isin": text_facts.get(m["isin"]),
        "symbol": text_facts.get(m["symbol"]),
        "scrip_code": text_facts.get(m["scrip_code"]),
        "company_name": text_facts.get(m["company_name"]),
        "period_start": parse_date_flexible(text_facts.get(m["period_start"])),
        "period_end": parse_date_flexible(text_facts.get(m["period_end"])),
        "reporting_quarter": text_facts.get(m["reporting_quarter"]),
    }

    best: dict[tuple[str, date, int], tuple[int, ParsedFact]] = {}
    for name, el, ctx in numeric:
        item = li.element_item(name)
        pe = ctx.period_end
        if item is None or pe is None:
            out.unmapped_numeric[name] = out.unmapped_numeric.get(name, 0) + 1
            continue
        months = 0 if ctx.instant is not None else months_between(ctx.start or pe, ctx.end or pe)
        rank = li.element_rank(name)
        fact = ParsedFact(
            line_item=item,
            scope=scope,
            period_end=pe,
            duration_months=months,
            value=float(el.text.strip()),  # type: ignore[union-attr]
            decimals=_decimals(el.get("decimals")),
            raw_element=name,
            context_id=ctx.id,
            unit=el.get("unitRef"),
        )
        key = (item, pe, months)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, fact)
    out.facts = [f for _, f in sorted(best.values(), key=lambda t: (t[1].period_end, t[1].line_item))]
    return out


def has_core_items(parsed: ParsedFiling) -> bool:
    items = {f.line_item for f in parsed.facts}
    return {"revenue_from_operations", "profit_before_tax", "total_expenses"} <= items
