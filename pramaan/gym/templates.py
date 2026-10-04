from __future__ import annotations

import calendar
import random

from pramaan.core.periods import PeriodKey, fiscal_year_label

ITEM_VARIANTS: dict[str, list[str]] = {
    "revenue_from_operations": ["revenue from operations", "revenue", "net sales", "turnover", "top line"],
    "other_income": ["other income"],
    "total_income": ["total income"],
    "total_expenses": ["total expenses", "total expenditure"],
    "finance_costs": ["finance costs", "interest expense"],
    "employee_benefits_expense": ["employee benefits expense", "employee cost"],
    "depreciation_and_amortisation": ["depreciation and amortisation", "depreciation"],
    "profit_before_tax": ["profit before tax", "PBT"],
    "total_tax_expense": ["total tax expense", "tax expense"],
    "net_profit_for_period": ["net profit", "profit after tax", "PAT", "net profit for the period"],
    "total_comprehensive_income": ["total comprehensive income"],
    "basic_eps": ["basic EPS", "basic earnings per share"],
    "diluted_eps": ["diluted EPS", "diluted earnings per share"],
    "other_comprehensive_income": ["other comprehensive income"],
}

RATIO_LABELS: dict[str, list[str]] = {
    "net_profit_margin": ["net profit margin", "PAT margin"],
    "effective_tax_rate": ["effective tax rate"],
    "other_income_share_of_pbt": ["other income as a share of profit before tax"],
}

SCOPE_VARIANTS = {
    "consolidated": ["consolidated", "consolidated"],
    "standalone": ["standalone", "standalone"],
}


def _dmy(d: object) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"  # type: ignore[attr-defined]


def _mdy(d: object) -> str:
    return f"{calendar.month_name[d.month]} {d.day}, {d.year}"  # type: ignore[attr-defined]


def period_variants(key: PeriodKey, allow_short: bool = True) -> list[str]:
    d = key.end
    fy = fiscal_year_label(d)
    out: list[str] = []
    if key.months == 3:
        out += [f"the quarter ended {_dmy(d)}", f"the quarter ended {_mdy(d)}"]
        q = {6: 1, 9: 2, 12: 3, 3: 4}.get(d.month)
        if q and allow_short:
            out += [f"Q{q} {fy}", f"the {calendar.month_name[d.month]} quarter of {fy}"]
    elif key.months == 6:
        out += [f"the six months ended {_dmy(d)}", f"the half year ended {_dmy(d)}"]
        if allow_short:
            out += [f"H1 {fy}", f"the first half of {fy}"]
    elif key.months == 9:
        out += [f"the nine months ended {_dmy(d)}"]
    elif key.months == 12:
        out += [f"the year ended {_dmy(d)}"]
        if allow_short:
            out += [f"the full year {fy}"]
    return out


T1 = [
    "What was {company}'s {scope} {item} for {period}? Answer in {unit}.",
    "Tell me {company}'s {scope} {item} for {period}, in {unit}.",
    "How much {item} did {company} report on a {scope} basis for {period}? Give the figure in {unit}.",
    "For {period}, what was the {scope} {item} of {company}? Please answer in {unit}.",
]
T2 = [
    "By what percentage did {company}'s {scope} {item} change year on year in {period}?",
    "What was the year-on-year growth in {company}'s {scope} {item} for {period}, in percent?",
    "How much did {company}'s {scope} {item} grow or shrink versus the same quarter last year in {period}? Answer in percent.",
]
T3 = [
    "What was {company}'s {scope} {ratio} for {period}? Answer in percent.",
    "Compute {company}'s {scope} {ratio} for {period} as a percentage.",
]
T4 = [
    "What percentage of {company}'s {scope} {item} for {cum} came in {quarter}?",
    "Of {company}'s {scope} {item} for {cum}, what share in percent came in {quarter}?",
]
T5 = [
    "Rank {companies} by {metric} for {period}, highest first.",
    "Order {companies} from highest to lowest {metric} for {period}.",
]
T6B_FALSE = [
    "Why did {company}'s {scope} {item} fall in {period}?",
    "What caused the decline in {company}'s {scope} {item} for {period}?",
]
T6B_CTRL = [
    "{company}'s {scope} {item} fell sharply year on year in {period}. By what percentage did it change?",
]


def join_companies(names: list[str]) -> str:
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]
