from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import duckdb
import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from pramaan.agent import calc
from pramaan.agent.corpus import Corpus
from pramaan.agent.defenses import POLICY_NOTE, Defense, spotlight
from pramaan.agent.injection import Injection
from pramaan.agent.modules import AgentModules
from pramaan.agent.normalize import (
    NormTable,
    from_parsed_json,
    from_table_read,
    select_value,
    to_answer_units,
)
from pramaan.core.config import PerQuestionBudget, line_items
from pramaan.core.errors import AsOfViolation, BudgetExceeded, ToolError
from pramaan.core.periods import PeriodKey
from pramaan.core.schemas import ValueResult
from pramaan.data.layout import Word
from pramaan.data.store import jload
from pramaan.data.tables import detect_unit
from pramaan.retrieval.cascade import Retriever
from pramaan.retrieval.queries import build_query

_TABLE_CACHE: dict[tuple[str, str], NormTable] = {}
_TABLE_LOCK = threading.Lock()
_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _key_lock(key: tuple[str, str]) -> threading.Lock:
    with _TABLE_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


class CompanyMatch(BaseModel):
    company_id: str
    symbol: str
    name: str
    score: float


class FilingInfo(BaseModel):
    filing_id: str
    company_id: str
    period_end: str
    pdf_published_at: str
    role: str
    n_pages: int


class SearchHit(BaseModel):
    page_id: str
    filing_id: str
    page_no: int
    score: float
    scope: str | None = None
    page_type: str | None = None
    snippet: str = ""


class ToolBudgetExceeded(ToolError):
    pass


class ToolBox:
    def __init__(
        self,
        corpus: Corpus,
        retriever: Retriever,
        modules: AgentModules,
        as_of: datetime,
        reader: str = "llm",
        budget: PerQuestionBudget | None = None,
        stages: str = "rrf+ce",
        pages_per_get: int = 3,
        search_k: int = 8,
        lessons: str = "",
        trace: Callable[[str, dict[str, Any]], None] | None = None,
        row_docs: bool = True,
        injection: Injection | None = None,
        defense: Defense | None = None,
    ) -> None:
        self.injection = injection
        self.defense = defense or Defense()
        self.corpus = corpus
        self.retriever = retriever
        self.modules = modules
        self.as_of = as_of
        self.reader = reader
        self.budget = budget or PerQuestionBudget()
        self.stages = stages
        self.pages_per_get = pages_per_get
        self.search_k = search_k
        self.lessons = lessons
        self.trace = trace or (lambda t, d: None)
        self.row_docs = row_docs
        self.tool_calls = 0
        self.internal_calls = 0
        self.search_log: list[dict[str, Any]] = []
        self.extracted: list[dict[str, Any]] = []
        self.read_errors: list[str] = []
        self.pages_read: list[str] = []
        self.tables_by_page: dict[str, NormTable] = {}

    def bump(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.budget.max_tool_calls:
            raise ToolBudgetExceeded(f"more than {self.budget.max_tool_calls} tool calls")

    def resolve_company(self, name_or_ticker: str) -> CompanyMatch | None:
        hits = self.corpus.resolve_company(name_or_ticker)
        return CompanyMatch(**hits[0]) if hits else None

    def list_filings(self, company_id: str, period_end: str | None = None) -> list[FilingInfo]:
        out = []
        for f in self.corpus.visible_filings(company_id, self.as_of):
            if period_end and f["period_end"].isoformat() != period_end:
                continue
            out.append(
                FilingInfo(
                    filing_id=f["filing_id"],
                    company_id=company_id,
                    period_end=f["period_end"].isoformat(),
                    pdf_published_at=f["pdf_published_at"].isoformat(),
                    role=f["role"],
                    n_pages=int(f["n_pages"] or 0),
                )
            )
        return out

    def search(
        self,
        query: str,
        company_id: str | None = None,
        filing_id: str | None = None,
        k: int | None = None,
        scope: str | None = None,
        exclude_pages: set[str] | None = None,
    ) -> list[SearchHit]:
        self.internal_calls += 1
        t0 = time.perf_counter()
        res = self.retriever.search(
            query,
            company_id=company_id,
            filing_id=filing_id,
            scope=scope,
            as_of=self.as_of,
            k=(k or self.search_k) + len(exclude_pages or ()),
            stages=self.stages,
        )
        hits: list[SearchHit] = []
        for h in res.hits:
            if exclude_pages and h.page_id in exclude_pages:
                continue
            p = self.corpus.page(h.page_id, self.as_of)
            hits.append(
                SearchHit(
                    page_id=h.page_id,
                    filing_id=p["filing_id"],
                    page_no=p["page_no"],
                    score=h.score,
                    scope=p["scope"],
                    page_type=p["page_type"],
                    snippet=(p["text"] or "")[:160].replace("\n", " "),
                )
            )
        if self.injection and self.injection.gold_pages and filing_id:
            gold = [g for g in self.injection.gold_pages if g.startswith(filing_id + "#")]
            if gold:
                have = {h.page_id for h in hits}
                front = []
                for g in gold:
                    if g in (exclude_pages or ()):
                        continue
                    p = self.corpus.page(g, self.as_of)
                    front.append(SearchHit(page_id=g, filing_id=p["filing_id"], page_no=p["page_no"], score=99.0, scope=p["scope"], page_type=p["page_type"], snippet=""))
                hits = front + [h for h in hits if h.page_id not in {f.page_id for f in front}]
                del have
        hits = hits[: k or self.search_k]
        self.search_log.append(
            {
                "query": query,
                "company_id": company_id,
                "filing_id": filing_id,
                "scope": scope,
                "pages": [h.page_id for h in hits],
                "margin": res.margin,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "timings": res.timings_ms,
            }
        )
        self.trace("search", {"query": query, "pages": [h.page_id for h in hits][:5]})
        return hits

    def read_table(self, page_id: str, note: str = "") -> NormTable:
        self.internal_calls += 1
        page = self.corpus.page(page_id, self.as_of)
        key = (page_id, f"{self.reader}:{self.modules.fingerprint()}:{note}:{self.defense.spotlight}:{id(self.corpus.store)}")
        # One reader per page: workers that want the same page wait for the first read and reuse
        # it, so concurrent questions never pay for the same page twice.
        with _key_lock(key):
            return self._read_table_locked(page_id, page, key, note)

    def _read_table_locked(self, page_id: str, page: dict[str, Any], key: tuple[str, str], note: str) -> NormTable:
        with _TABLE_LOCK:
            cached = _TABLE_CACHE.get(key)
        if cached is not None:
            self.tables_by_page[page_id] = cached
            return cached
        rules_tbl = jload(page.get("tbl"), None)
        nt: NormTable | None = None
        if self.reader in ("rules", "hybrid") and rules_tbl:
            cand = from_parsed_json(page_id, rules_tbl, page.get("scope"))
            if self.reader == "rules" or self._rules_ok(cand):
                nt = cand
        if nt is None:
            lessons = "\n".join(x for x in (self.lessons, note) if x)
            words = [Word.from_list(w) for w in jload(page.get("words"), [])]
            page_text = page["text"] or ""
            if self.defense.spotlight:
                page_text = spotlight(page_text)
                lessons = "\n".join(x for x in (lessons, POLICY_NOTE) if x)
            try:
                tr = self.modules.read_table(page_text, lessons)
                nt = from_table_read(page_id, tr, words, page.get("scope"))
            except BudgetExceeded:
                raise
            except Exception as exc:
                self.read_errors.append(f"{page_id}: {exc}")
                nt = NormTable(page_id, None, [], [], "llm", page.get("scope"))
            if nt.unit is None:
                unit, _ = detect_unit(page["text"] or "")
                nt.unit = unit
        self.pages_read.append(page_id)
        self.tables_by_page[page_id] = nt
        self.trace("tool_result", {"tool": "read_table", "page_id": page_id, "rows": len(nt.rows), "source": nt.source})
        with _TABLE_LOCK:
            _TABLE_CACHE[key] = nt
        return nt

    @staticmethod
    def _rules_ok(nt: NormTable) -> bool:
        has_period = sum(1 for c in nt.columns if c.period) >= 1
        return nt.unit is not None and has_period and nt.n_items() >= 6

    def get_value(
        self,
        company: str,
        scope: str | None,
        item: str,
        period: str,
        exclude_pages: set[str] | None = None,
        note: str = "",
    ) -> ValueResult:
        m = self.resolve_company(company)
        if m is None:
            return ValueResult(found=False, item=item, period=period, scope=scope, reason="unknown_company")  # type: ignore[arg-type]
        try:
            pk = PeriodKey.parse(period)
        except ValueError:
            return ValueResult(found=False, company_id=m.company_id, item=item, period=period, scope=scope, reason="bad_period")  # type: ignore[arg-type]
        li = line_items()
        if item not in li.items:
            return ValueResult(found=False, company_id=m.company_id, item=item, period=period, scope=scope, reason="unknown_item")  # type: ignore[arg-type]
        if self.injection and (m.company_id, item, period) in self.injection.gold_values:
            inj = self.injection.gold_values[(m.company_id, item, period)].model_copy(deep=True)
            self.extracted.append(inj.model_dump())
            return inj
        filings = self.corpus.candidate_filings(m.company_id, pk, self.as_of)
        if not filings:
            vis = self.corpus.visible_filings(m.company_id, self.as_of)
            why = "no_filing_visible" if not vis else "no_filing_covers_period"
            return ValueResult(found=False, company_id=m.company_id, item=item, period=period, scope=scope, reason=why)  # type: ignore[arg-type]
        scopes = [scope] if scope else ["consolidated", "standalone"]
        excl = set(exclude_pages or ())
        tried: list[str] = []
        best_margin = 0.0
        for sc in scopes:
            for filing in filings[:2]:
                query = build_query(m.name, sc, [item], [period])
                hits = self.search(query, m.company_id, filing["filing_id"], scope=sc, exclude_pages=excl)
                if hits:
                    best_margin = max(best_margin, self.search_log[-1]["margin"])
                for rank, h in enumerate(hits[: self.pages_per_get]):
                    if h.page_id in excl:
                        continue
                    nt = self.read_table(h.page_id, note)
                    tried.append(h.page_id)
                    if nt.page_scope not in (None, "both", sc) and not any(c.scope == sc for c in nt.columns):
                        continue
                    sel = select_value(nt, item, pk, sc)
                    if sel is None:
                        continue
                    val = to_answer_units(sel, nt.unit)
                    page_unit = nt.unit
                    unit_ans = "INR_per_share" if sel.per_share else "INR_crore"
                    reason = ""
                    if val is None:
                        val = sel.value
                        reason = "unit_unknown"
                    res = ValueResult(
                        found=True,
                        company_id=m.company_id,
                        filing_id=filing["filing_id"],
                        scope=sc,  # type: ignore[arg-type]
                        item=item,
                        period=period,
                        value=round(val, 4),
                        unit=unit_ans,
                        page_id=h.page_id,
                        quote=sel.raw,
                        bbox=sel.box,
                        column_header=str(sel.col.period) if sel.col.period else sel.col.header,
                        row_label=sel.row.label,
                        page_unit=page_unit,
                        reason=reason,
                        retrieval_rank=rank + 1,
                        retrieval_margin=best_margin,
                        pages_read=tried,
                    )
                    self.extracted.append(res.model_dump())
                    return res
        return ValueResult(
            found=False,
            company_id=m.company_id,
            item=item,
            period=period,
            scope=scope,  # type: ignore[arg-type]
            reason="value_not_found_on_retrieved_pages",
            pages_read=tried,
            retrieval_margin=best_margin,
        )

    def check_filed(self, company: str, period: str) -> dict[str, Any]:
        m = self.resolve_company(company)
        if m is None:
            return {"filed": False, "reason": "unknown_company"}
        pk = PeriodKey.parse(period)
        for f in self.corpus.visible_filings(m.company_id, self.as_of):
            if f["period_end"] == pk.end:
                return {"filed": True, "filing_id": f["filing_id"], "published_at": f["pdf_published_at"].isoformat()}
        return {"filed": False, "reason": "not_published_as_of", "as_of": self.as_of.isoformat()}

    def calculate(self, expression: str, variables: dict[str, float]) -> float:
        return calc.calculate(expression, variables)

    def rank(self, entries: list[dict[str, Any]], descending: bool = True) -> list[str]:
        vals = [(e["company"], float(e["value"])) for e in entries if e.get("value") is not None]
        vals.sort(key=lambda t: t[1], reverse=descending)
        return [c for c, _ in vals]

    def query_facts(self, sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
        return run_facts_query(self.extracted, sql, max_rows)


def validate_select(sql: str) -> None:
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ToolError(f"cannot parse SQL: {exc}") from exc
    if not isinstance(tree, exp.Select) and not isinstance(tree, exp.Union):
        raise ToolError("only SELECT statements are allowed")
    banned = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Pragma, exp.Copy, exp.Attach)
    for node in tree.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, banned):
            raise ToolError("statement type not allowed")
        if isinstance(n, exp.Table) and n.name.lower() != "extracted":
            raise ToolError("only the table 'extracted' may be queried")
        if isinstance(n, exp.Anonymous) and n.name.lower() in {"read_csv", "read_parquet", "read_json", "glob", "read_text"}:
            raise ToolError("file functions are not allowed")


def run_facts_query(rows: list[dict[str, Any]], sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
    validate_select(sql)
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            "CREATE TABLE extracted (company_id VARCHAR, filing_id VARCHAR, scope VARCHAR, item VARCHAR, period VARCHAR, value DOUBLE, unit VARCHAR, page_id VARCHAR, quote VARCHAR)"
        )
        for r in rows:
            con.execute(
                "INSERT INTO extracted VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [r.get("company_id"), r.get("filing_id"), r.get("scope"), r.get("item"), r.get("period"), r.get("value"), r.get("unit"), r.get("page_id"), r.get("quote")],
            )
        con.execute("SET enable_external_access=false")
        cur = con.execute(f"SELECT * FROM ({sql.rstrip(';')}) LIMIT {int(max_rows)}")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    except duckdb.Error as exc:
        raise ToolError(f"query failed: {exc}") from exc
    finally:
        con.close()


def clear_table_cache() -> None:
    with _TABLE_LOCK:
        _TABLE_CACHE.clear()
        _KEY_LOCKS.clear()


__all__ = ["AsOfViolation", "ToolBox", "ToolBudgetExceeded", "clear_table_cache", "run_facts_query", "validate_select"]
