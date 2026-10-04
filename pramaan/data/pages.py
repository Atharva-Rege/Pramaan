from __future__ import annotations

import re
from dataclasses import dataclass

from pramaan.core.config import line_items
from pramaan.data.tables import ParsedTable

CORE_ITEMS = (
    "revenue_from_operations",
    "total_income",
    "total_expenses",
    "profit_before_tax",
    "total_tax_expense",
    "net_profit_for_period",
    "current_tax",
    "deferred_tax",
    "other_income",
    "finance_costs",
    "basic_eps",
)

_STATEMENT_TITLE = re.compile(
    r"statement\s+of\s+(?:un)?-?audited[\w\s,&\-]*financial\s+results|financial\s+results\s+for\s+the\s+(?:quarter|half|year|period|nine)|"
    r"unaudited\s+(?:standalone|consolidated)[\w\s]*results|audited\s+(?:standalone|consolidated)[\w\s]*results",
    re.I,
)
_SEGMENT = re.compile(r"segment[\s\-]*(?:wise|information|revenue|results|reporting)|segment\s+(?:assets|liabilities)", re.I)
_BALANCE = re.compile(r"statement\s+of\s+assets\s+and\s+liabilities|balance\s+sheet\s+as\s+at", re.I)
_CASHFLOW = re.compile(r"cash\s+flow\s+statement|statement\s+of\s+cash\s+flows", re.I)
_AUDITOR = re.compile(r"independent\s+auditor|limited\s+review\s+report|review\s+report|chartered\s+accountants", re.I)
_PRESENTATION = re.compile(r"ebitda|capex|net debt|free cash flow|market capitali[sz]ation|order inflow|order book", re.I)
_COVER = re.compile(r"outcome\s+of\s+(?:the\s+)?board\s+meeting|pursuant\s+to\s+regulation|we\s+wish\s+to\s+inform|we\s+hereby\s+inform", re.I)


@dataclass
class PageInfo:
    page_type: str
    is_results: bool
    scope: str | None
    n_items: int
    items: list[str]


def _scope_signals(text: str) -> tuple[bool, bool]:
    li = line_items()
    low = text.lower()
    st = any(a in low for a in li.scopes["standalone"]["aliases"])
    co = any(a in low for a in li.scopes["consolidated"]["aliases"] if a != "group")
    return st, co


def classify_page(text: str, table: ParsedTable, prev_scope: str | None, page_no: int) -> PageInfo:
    head = "\n".join(text.splitlines()[:25])
    items = sorted({r.item for r in table.rows if r.item})
    core = [i for i in items if i in CORE_ITEMS]
    has_periods = any(c.period for c in table.columns)
    st, co = _scope_signals(head)
    scope: str | None
    if any(c.scope for c in table.columns):
        scope = "both"
    elif co and not st:
        scope = "consolidated"
    elif st and not co:
        scope = "standalone"
    elif st and co:
        scope = "both"
    else:
        scope = None

    presentation = len(set(m.group(0).lower() for m in _PRESENTATION.finditer(text))) >= 2
    is_statement = len(core) >= 5 and len(table.columns) >= 2 and not presentation
    page_type = "other"
    if is_statement and not _SEGMENT.search(head) or is_statement and len(core) >= 7:
        page_type = "results_statement"
    elif len(core) >= 4 and len(table.columns) >= 2 and not has_periods and prev_scope is not None:
        page_type = "results_statement"
    elif _SEGMENT.search(text) and len(table.rows) >= 4:
        page_type = "segment"
    elif _BALANCE.search(head):
        page_type = "balance_sheet"
    elif _CASHFLOW.search(head):
        page_type = "cash_flow"
    elif _AUDITOR.search(head):
        page_type = "auditor_report"
    elif page_no <= 2 and _COVER.search(text):
        page_type = "cover"
    elif _STATEMENT_TITLE.search(head) and len(table.rows) >= 4:
        page_type = "results_statement"
    elif len(table.rows) < 3:
        page_type = "notes"
    is_results = page_type == "results_statement"
    if is_results and scope is None:
        scope = prev_scope
    return PageInfo(page_type, is_results, scope, len(items), items)
