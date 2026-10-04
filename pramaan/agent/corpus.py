from __future__ import annotations

from datetime import datetime
from typing import Any

from rapidfuzz import fuzz, process

from pramaan.core.errors import AsOfViolation
from pramaan.core.periods import PeriodKey
from pramaan.data.store import Store
from pramaan.gym.slotparse import CompanyIndex, _norm


class Corpus:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.cidx = CompanyIndex.from_store(store)
        self.companies = {c["company_id"]: c for c in store.q("SELECT * FROM companies")}
        self._names = {_norm(c["name"]): cid for cid, c in self.companies.items()}
        self._symbols = {_norm(c["symbol"]): cid for cid, c in self.companies.items()}
        self._filing_cache: dict[str, list[dict[str, Any]]] = {}
        self._page_cache: dict[str, dict[str, Any]] = {}

    def company_name(self, company_id: str) -> str:
        c = self.companies.get(company_id)
        return c["name"] if c else company_id

    def resolve_company(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        found = self.cidx.find(text)
        out: list[dict[str, Any]] = []
        for cid in found[:limit]:
            c = self.companies[cid]
            out.append({"company_id": cid, "symbol": c["symbol"], "name": c["name"], "score": 100.0})
        if out:
            return out
        q = _norm(text)
        pool = {**self._names, **self._symbols}
        for choice, score, _ in process.extract(q, list(pool), scorer=fuzz.WRatio, limit=limit):
            if score >= 85:
                cid = pool[choice]
                c = self.companies[cid]
                out.append({"company_id": cid, "symbol": c["symbol"], "name": c["name"], "score": float(score)})
        return out

    def visible_filings(self, company_id: str, as_of: datetime) -> list[dict[str, Any]]:
        if company_id not in self._filing_cache:
            self._filing_cache[company_id] = self.store.q(
                "SELECT filing_id, company_id, period_end, board_meeting_date, pdf_published_at, role, split, n_pages, pdf_source "
                "FROM filings WHERE company_id = ? AND pdf_path IS NOT NULL ORDER BY period_end DESC",
                [company_id],
            )
        return [f for f in self._filing_cache[company_id] if f["pdf_published_at"] is not None and f["pdf_published_at"] < as_of]

    def candidate_filings(self, company_id: str, period: PeriodKey, as_of: datetime) -> list[dict[str, Any]]:
        vis = self.visible_filings(company_id, as_of)
        pref = {0: 0, 12: 1, 3: 2, 6: 3, 9: 4}
        scored: list[tuple[int, dict[str, Any]]] = []
        for f in vis:
            diff = (f["period_end"].year - period.end.year) * 12 + (f["period_end"].month - period.end.month)
            if diff in pref:
                scored.append((pref[diff], f))
        scored.sort(key=lambda t: (t[0], -t[1]["period_end"].toordinal()))
        return [f for _, f in scored]

    def filing_visible(self, filing_id: str, as_of: datetime) -> bool:
        row = self.store.filing(filing_id)
        return bool(row and row["pdf_published_at"] is not None and row["pdf_published_at"] < as_of)

    def page(self, page_id: str, as_of: datetime | None) -> dict[str, Any]:
        if page_id not in self._page_cache:
            p = self.store.page(page_id)
            if p is None:
                raise KeyError(page_id)
            self._page_cache[page_id] = p
        p = self._page_cache[page_id]
        if as_of is not None and not self.filing_visible(p["filing_id"], as_of):
            raise AsOfViolation(f"{page_id} was not published before {as_of}")
        return p

    def filing(self, filing_id: str) -> dict[str, Any] | None:
        return self.store.filing(filing_id)
