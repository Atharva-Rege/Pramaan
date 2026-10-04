from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from pramaan.core.periods import PeriodKey
from pramaan.data.store import Store, jload


@dataclass
class DocRecord:
    doc_id: str
    page_id: str
    kind: str
    filing_id: str
    company_id: str
    scope: str | None
    page_type: str | None
    role: str
    pdf_ts: float
    item: str | None
    text: str
    ce_text: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _period_label(p: str | None) -> str:
    if not p:
        return ""
    try:
        k = PeriodKey.parse(p)
    except ValueError:
        return p
    return f"{k.months}m ended {k.end.day} {calendar.month_abbr[k.end.month]} {k.end.year}"


def table_digest(tbl: dict[str, Any], max_rows: int = 45) -> str:
    cols = tbl.get("columns", [])
    heads = [_period_label(c.get("period")) or f"col{c['index']}" for c in cols]
    lines: list[str] = []
    for row in tbl.get("rows", [])[:max_rows]:
        vals = [v for v in row["values"] if v is not None]
        if not vals:
            continue
        lines.append(f"{row['label']} | " + " | ".join(vals[:4]))
    head = f"unit {tbl.get('unit') or 'unknown'} | columns " + " ; ".join(heads[:4])
    return head + "\n" + "\n".join(lines)


def row_text(company: str, scope: str | None, label: str, values: list[str | None], heads: list[str]) -> str:
    pairs = [f"{h}: {v}" for h, v in zip(heads, values, strict=False) if v is not None]
    return f"{company} {scope or ''} | {label} | " + " | ".join(pairs)


def page_prefix(company: str, scope: str | None, page_type: str | None, period_end: Any) -> str:
    pe = period_end.isoformat() if hasattr(period_end, "isoformat") else str(period_end)
    return f"{company} | {scope or 'unspecified scope'} | {page_type or 'page'} | period {pe}"


def build_docs(store: Store, include_roles: tuple[str, ...] = ("target", "previous_quarter"), row_docs: bool = True) -> list[DocRecord]:
    rows = store.q(
        "SELECT p.page_id, p.filing_id, p.page_no, p.text, p.scope, p.page_type, p.tbl, p.is_results_page, "
        "f.company_id, f.role, f.period_end, f.pdf_published_at, c.name AS company_name "
        "FROM pages p JOIN filings f USING (filing_id) JOIN companies c USING (company_id) "
        "WHERE f.pdf_path IS NOT NULL ORDER BY p.filing_id, p.page_no"
    )
    docs: list[DocRecord] = []
    for r in rows:
        if r["role"] not in include_roles:
            continue
        ts = r["pdf_published_at"]
        pdf_ts = ts.timestamp() if isinstance(ts, datetime) else 0.0
        tbl = jload(r["tbl"], None)
        prefix = page_prefix(r["company_name"], r["scope"], r["page_type"], r["period_end"])
        if r["is_results_page"] and tbl:
            digest = table_digest(tbl)
            body = digest
        else:
            body = (r["text"] or "")[:1800]
        text = f"{prefix}\n{body}"
        docs.append(
            DocRecord(
                doc_id=r["page_id"],
                page_id=r["page_id"],
                kind="page",
                filing_id=r["filing_id"],
                company_id=r["company_id"],
                scope=r["scope"],
                page_type=r["page_type"],
                role=r["role"],
                pdf_ts=pdf_ts,
                item=None,
                text=(r["text"] or "")[:3500] if not (r["is_results_page"] and tbl) else text,
                ce_text=text,
            )
        )
        if row_docs and tbl:
            heads = [_period_label(c.get("period")) or f"col{c['index']}" for c in tbl["columns"]]
            for k, row in enumerate(tbl["rows"]):
                if not any(v is not None for v in row["values"]):
                    continue
                rt = row_text(r["company_name"], r["scope"], row["label"], row["values"], heads)
                docs.append(
                    DocRecord(
                        doc_id=f"{r['page_id']}::r{k}",
                        page_id=r["page_id"],
                        kind="row",
                        filing_id=r["filing_id"],
                        company_id=r["company_id"],
                        scope=r["scope"],
                        page_type=r["page_type"],
                        role=r["role"],
                        pdf_ts=pdf_ts,
                        item=row.get("item"),
                        text=rt,
                        ce_text=rt,
                    )
                )
    return docs
