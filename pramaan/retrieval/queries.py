from __future__ import annotations

import calendar

from pramaan.core.config import line_items
from pramaan.core.periods import PeriodKey
from pramaan.core.schemas import Intent, Question

RATIO_TERMS = {
    "net_profit_margin": ["net_profit_for_period", "revenue_from_operations"],
    "effective_tax_rate": ["total_tax_expense", "profit_before_tax"],
    "other_income_share_of_pbt": ["other_income", "profit_before_tax"],
}


def period_text(p: str) -> str:
    k = PeriodKey.parse(p)
    unit = {3: "quarter", 6: "half year six months", 9: "nine months", 12: "year"}.get(k.months, "period")
    return f"{unit} ended {k.end.day} {calendar.month_name[k.end.month]} {k.end.year}"


def item_terms(item: str, n: int = 2) -> str:
    li = line_items()
    spec = li.items.get(item, {})
    labels = list(spec.get("aliases", []))[:n]
    return " ".join(dict.fromkeys([li.human_name(item), *labels]))


def build_query(company_name: str, scope: str | None, items: list[str], periods: list[str]) -> str:
    parts = [company_name]
    if scope:
        parts.append(scope)
    parts.append("financial results statement")
    parts += [item_terms(i) for i in items]
    parts += [period_text(p) for p in periods[:2]]
    return " ".join(parts)


def question_query(q: Question, company_name: str) -> str:
    s: Intent = q.slots
    items = list(s.items)
    if s.ratio_name in RATIO_TERMS:
        items += RATIO_TERMS[s.ratio_name]
    if s.metric and s.metric in line_items().items:
        items.append(s.metric)
    return build_query(company_name, s.scope, items, s.periods)
