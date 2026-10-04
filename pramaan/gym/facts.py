from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pramaan.core.periods import PeriodKey


@dataclass(frozen=True)
class Fact:
    company_id: str
    scope: str
    item: str
    period: PeriodKey
    value_inr: float
    filing_id: str
    revision: int


class FactBook:
    def __init__(self, store: Any) -> None:
        rows = store.q(
            "SELECT f.company_id, x.scope, x.line_item, x.period_end, x.duration_months, x.value_inr, x.filing_id, x.revision, f.period_end AS filing_period "
            "FROM xbrl_facts x JOIN filings f USING (filing_id) WHERE x.is_latest"
        )
        self._d: dict[tuple[str, str, str, date, int], Fact] = {}
        for r in rows:
            key = (r["company_id"], r["scope"], r["line_item"], r["period_end"], r["duration_months"])
            fact = Fact(
                r["company_id"],
                r["scope"],
                r["line_item"],
                PeriodKey(r["period_end"], r["duration_months"]),
                r["value_inr"],
                r["filing_id"],
                r["revision"],
            )
            prev = self._d.get(key)
            own = r["filing_period"] == r["period_end"]
            if prev is None or own:
                self._d[key] = fact

    def get(self, company_id: str, scope: str, item: str, period: PeriodKey) -> Fact | None:
        return self._d.get((company_id, scope, item, period.end, period.months))

    def value(self, company_id: str, scope: str, item: str, period: PeriodKey) -> float | None:
        f = self.get(company_id, scope, item, period)
        return f.value_inr if f else None

    def scopes(self, company_id: str, period: PeriodKey) -> list[str]:
        out = []
        for scope in ("consolidated", "standalone"):
            if self.get(company_id, scope, "revenue_from_operations", period):
                out.append(scope)
        return out
