from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from pramaan.core.config import gym_config, line_items, normalize_label
from pramaan.core.periods import (
    PeriodKey,
    fiscal_year_label,
    month_end,
    month_number,
    parse_dates,
)
from pramaan.core.schemas import Intent

EXTRA_NAMES: dict[str, list[str]] = {
    "revenue_from_operations": ["revenue", "revenues", "turnover", "top line", "topline", "net sales", "sales", "operating revenue"],
    "other_income": ["other income"],
    "total_income": ["total income", "total revenue"],
    "total_expenses": ["total expenses", "total expenditure", "total costs", "total cost"],
    "finance_costs": ["finance costs", "finance cost", "interest cost", "interest expense", "interest costs"],
    "employee_benefits_expense": ["employee benefits expense", "employee cost", "employee costs", "staff cost", "staff costs", "employee expenses"],
    "depreciation_and_amortisation": ["depreciation and amortisation", "depreciation and amortization", "depreciation", "d&a"],
    "profit_before_tax": ["profit before tax", "pbt", "pre-tax profit", "pretax profit"],
    "total_tax_expense": ["total tax expense", "tax expense", "income tax expense", "tax outgo", "taxes"],
    "net_profit_for_period": ["net profit", "profit after tax", "pat", "profit for the period", "net income", "bottom line", "net earnings"],
    "total_comprehensive_income": ["total comprehensive income", "tci"],
    "basic_eps": ["basic eps", "basic earnings per share", "eps", "earnings per share"],
    "diluted_eps": ["diluted eps", "diluted earnings per share"],
    "other_comprehensive_income": ["other comprehensive income", "oci"],
    "current_tax": ["current tax"],
    "deferred_tax": ["deferred tax"],
}

RATIO_PATTERNS = [
    ("net_profit_margin", re.compile(r"net\s+profit\s+margin|pat\s+margin|net\s+margin|profit\s+margin", re.I)),
    ("effective_tax_rate", re.compile(r"effective\s+tax\s+rate|tax\s+rate", re.I)),
    ("other_income_share_of_pbt", re.compile(r"other\s+income\s+(?:as\s+a\s+)?(?:share|percentage|proportion)\s+of\s+(?:profit\s+before\s+tax|pbt)", re.I)),
]

QUARTER_WORDS = {
    "june": (6, 30),
    "september": (9, 30),
    "sept": (9, 30),
    "sep": (9, 30),
    "december": (12, 31),
    "march": (3, 31),
}


@dataclass
class CompanyIndex:
    aliases: list[tuple[str, str]]

    @classmethod
    def from_store(cls, store: object) -> CompanyIndex:
        rows = store.q("SELECT company_id, symbol, name, aliases FROM companies")  # type: ignore[attr-defined]
        out: list[tuple[str, str]] = []
        for r in rows:
            names = set(r["aliases"] or [])
            names.add(r["name"])
            names.add(r["symbol"])
            for n in names:
                a = _norm(n)
                if len(a) >= 2:
                    out.append((a, r["company_id"]))
        out.sort(key=lambda t: -len(t[0]))
        return cls(out)

    def find(self, text: str) -> list[str]:
        t = f" {_norm(text)} "
        found: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for alias, cid in self.aliases:
            start = 0
            needle = f" {alias} "
            while True:
                i = t.find(needle, start)
                if i < 0:
                    break
                span = (i, i + len(needle))
                if all(span[1] <= a or span[0] >= b for a, b in taken):
                    taken.append(span)
                    found.append((i, cid))
                start = i + 1
        found.sort()
        seen: list[str] = []
        for _, cid in found:
            if cid not in seen:
                seen.append(cid)
        return seen


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", s.lower().replace("&", " & "))).strip()


def _item_index() -> list[tuple[str, str]]:
    li = line_items()
    out: list[tuple[str, str]] = []
    for item in li.item_names:
        names = list(EXTRA_NAMES.get(item, [])) + list(li.items[item].get("names", []))
        for n in names:
            out.append((_norm(n), item))
    out.sort(key=lambda t: -len(t[0]))
    return out


_ITEM_INDEX: list[tuple[str, str]] | None = None


def find_items(text: str) -> list[str]:
    global _ITEM_INDEX
    if _ITEM_INDEX is None:
        _ITEM_INDEX = _item_index()
    t = f" {_norm(text)} "
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for alias, item in _ITEM_INDEX:
        needle = f" {alias} "
        i = t.find(needle)
        if i >= 0 and all(i + len(needle) <= a or i >= b for a, b in taken):
            taken.append((i, i + len(needle)))
            found.append((i, item))
    found.sort()
    out: list[str] = []
    for _, it in found:
        if it not in out:
            out.append(it)
    return out


def _fy_end(fy: int) -> date:
    return date(fy, 3, 31)


def find_periods(text: str, default_year: int | None = None) -> list[PeriodKey]:
    low = text.lower()
    out: list[PeriodKey] = []

    def add(k: PeriodKey) -> None:
        if k not in out:
            out.append(k)

    cum_months = 3
    if re.search(r"half[\s\-]*year|six\s+months|\bh1\b|first\s+half|\bh2\b|second\s+half", low):
        cum_months = 6
    elif re.search(r"nine\s+months|9\s*m\b|\b9m\b", low):
        cum_months = 9
    elif re.search(r"full[\s\-]*year|year\s+ended|\bfy\s?\d{2}\b(?!\s*q)|financial\s+year|twelve\s+months", low) and not re.search(r"quarter", low):
        cum_months = 12

    for dm in parse_dates(text):
        d = dm.value
        window = low[max(0, dm.start - 40) : dm.start]
        if re.search(r"year\s+ended\s*$|year\s+ending\s*$", window):
            add(PeriodKey(d, 12))
        elif re.search(r"half[\s\-]*year(?:ly)?\s+ended\s*$|six\s+months\s+ended\s*$", window):
            add(PeriodKey(d, 6))
        elif re.search(r"nine\s+months\s+ended\s*$", window):
            add(PeriodKey(d, 9))
        else:
            add(PeriodKey(d, 3))

    for m in re.finditer(r"q([1-4])\s*(?:of\s*)?(?:fy)?[\s\-']*(\d{2,4})", low):
        q, yy = int(m.group(1)), int(m.group(2))
        fy = 2000 + yy if yy < 100 else yy
        mm = {1: (fy - 1, 6), 2: (fy - 1, 9), 3: (fy - 1, 12), 4: (fy, 3)}[q]
        add(PeriodKey(month_end(*mm), 3))
    for m in re.finditer(r"(?:h|half\s*)([12])\s*(?:of\s*)?(?:fy)?[\s\-']*(\d{2,4})", low):
        h, yy = int(m.group(1)), int(m.group(2))
        fy = 2000 + yy if yy < 100 else yy
        add(PeriodKey(month_end(fy - 1, 9), 6) if h == 1 else PeriodKey(month_end(fy, 3), 6))
    for m in re.finditer(r"(?:first|second)\s+half\s+of\s+(?:fy)?[\s\-']*(\d{2,4})", low):
        yy = int(m.group(1))
        fy = 2000 + yy if yy < 100 else yy
        add(PeriodKey(month_end(fy - 1, 9), 6) if "first" in m.group(0) else PeriodKey(month_end(fy, 3), 6))
    for m in re.finditer(r"\b(june|september|sept|sep|december|march)\s+quarter(?:\s+of)?(?:\s+(?:fy)?[\s\-']*(\d{2,4}))?", low):
        mon, day = QUARTER_WORDS[m.group(1)]
        yy = m.group(2)
        if yy:
            n = int(yy)
            fy = 2000 + n if n < 100 else n
            year = fy if mon == 3 else fy - 1
        else:
            year = default_year or gym_config().profile.target_period_end.year
            if mon == 3:
                year = (default_year or gym_config().profile.target_period_end.year)
        add(PeriodKey(month_end(year, mon), 3))
    for m in re.finditer(r"\bfy[\s\-']*(\d{2,4})\b", low):
        n = int(m.group(1))
        fy = 2000 + n if n < 100 else n
        if re.search(r"q[1-4]\s*(?:of\s*)?fy[\s\-']*" + m.group(1), low):
            continue
        if re.search(r"(?:first|second)\s+half\s+of\s+fy[\s\-']*" + m.group(1), low):
            continue
        add(PeriodKey(_fy_end(fy), 12 if cum_months == 12 or "full" in low or "year" in low else cum_months))
    return out


def find_scope(text: str) -> str | None:
    low = text.lower()
    st = bool(re.search(r"stand[\s\-]?alone|non[\s\-]?consolidated", low))
    co = bool(re.search(r"consolidated|\bgroup\b", low)) and not re.search(r"non[\s\-]?consolidated", low)
    if st and not co:
        return "standalone"
    if co and not st:
        return "consolidated"
    return None


def parse_question(text: str, index: CompanyIndex) -> Intent:
    low = text.lower()
    companies = index.find(text)
    items = find_items(text)
    periods = find_periods(text)
    scope = find_scope(text)
    ratio_name = next((n for n, pat in RATIO_PATTERNS if pat.search(text)), None)
    operation = "lookup"
    metric = None
    premise = None
    if re.search(r"\brank\b|\border\b.*\bby\b|highest to lowest|largest to smallest|from highest|which of .* (?:had|reported) the (?:highest|largest|biggest)", low):
        operation = "rank"
        metric = ratio_name or (items[0] if items else None)
    elif re.search(r"what\s+(?:percentage|share|proportion|%)\s+of\b.*\b(?:came|was|came\s+in|accounted)", low) or re.search(r"how much of it came", low):
        operation = "cross_period"
    elif re.search(r"\bwhy\b.*\b(?:fall|fell|drop|dropped|decline|declined|decrease|decreased)\b", low) or re.search(r"\b(?:fell|declined|dropped)\s+(?:sharply|steeply|significantly)\b", low):
        operation = "premise_check"
        premise = "decline"
    elif ratio_name:
        operation = "ratio"
    elif re.search(r"year[\s\-]*on[\s\-]*year|\byoy\b|y-o-y|grow|growth|change(?:d)?\s+(?:by|from)|percentage\s+change|increase|decrease|compared\s+(?:with|to)\s+the\s+same", low):
        operation = "growth_yoy"
    if operation == "rank":
        items = [i for i in items if i != metric] if metric in items else items
    if operation in ("ratio", "rank") and ratio_name:
        items = []
    return Intent(
        companies=companies,
        scope=scope,  # type: ignore[arg-type]
        items=items,
        periods=[str(p) for p in periods],
        operation=operation,  # type: ignore[arg-type]
        ratio_name=ratio_name,
        metric=metric,
        premise_claim=premise,
    )


def slots_match(parsed: Intent, gold: Intent) -> bool:
    if parsed.operation != gold.operation:
        return False
    if sorted(parsed.companies) != sorted(gold.companies) and gold.operation != "rank":
        return False
    if gold.operation == "rank" and set(parsed.companies) != set(gold.companies):
        return False
    if (gold.scope or None) != (parsed.scope or None):
        return False
    if gold.operation == "rank":
        return (parsed.metric or "") == (gold.metric or "")
    if gold.ratio_name != parsed.ratio_name:
        return False
    if sorted(parsed.items) != sorted(gold.items):
        return False
    gp = {PeriodKey.parse(p) for p in gold.periods}
    pp = {PeriodKey.parse(p) for p in parsed.periods}
    return gp <= pp and (len(pp) <= len(gp) + 1)


def fiscal_label_for(end: date) -> str:
    return fiscal_year_label(end)


def month_name_to_num(name: str) -> int:
    return month_number(name)


def label_norm(s: str) -> str:
    return normalize_label(s)
