from __future__ import annotations

import os
import threading
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from pramaan.core.timeutil import utcnow

mcp = MCPServer(
    "pramaan",
    instructions=(
        "Pramaan answers questions about Indian listed companies from their results filings. Every number comes with the page and the exact "
        "printed figure. Prefer `ask` for full questions and cite the returned page and quote when you reply to the user. Not investment advice."
    ),
)

_lock = threading.Lock()
_state: dict[str, Any] = {}


def _rt() -> Any:
    with _lock:
        if "rt" not in _state:
            from pramaan.agent.factory import Runtime
            from pramaan.data.store import Store
            from pramaan.llm.cache import LLMStore
            from pramaan.llm.client import LLMClient
            from pramaan.retrieval.index import RetrievalIndex

            if os.environ.get("PRAMAAN_LLM") == "offline" or not os.environ.get("ANTHROPIC_API_KEY"):
                for r in ("SMALL", "STRONG", "REFLECTION"):
                    os.environ.setdefault(f"PRAMAAN_MODEL_{r}", "offline/rules")
            _state["rt"] = Runtime(Store(read_only=True), RetrievalIndex.load("main"), LLMClient(store=LLMStore()))
        return _state["rt"]


def _toolbox() -> Any:
    from pramaan.agent.factory import tier_lm
    from pramaan.agent.modules import AgentModules
    from pramaan.agent.tools import ToolBox
    from pramaan.core.config import budget_config

    rt = _rt()
    offline = os.environ.get("PRAMAAN_MODEL_SMALL", "").startswith("offline/")
    modules = AgentModules(tier_lm(rt, "small", "mcp"))
    return ToolBox(rt.corpus, rt.retriever, modules, utcnow(), reader="rules" if offline else "llm", budget=budget_config().per_question, stages="bm25+dense+rows" if offline else "rrf+ce")


@mcp.tool()
def resolve_company(name_or_ticker: str) -> list[dict[str, Any]]:
    return _rt().corpus.resolve_company(name_or_ticker)


@mcp.tool()
def list_filings(company_id: str, period_end: str | None = None) -> list[dict[str, Any]]:
    return [f.model_dump() for f in _toolbox().list_filings(company_id, period_end)]


@mcp.tool()
def search(query: str, company_id: str | None = None, filing_id: str | None = None, k: int = 8) -> list[dict[str, Any]]:
    return [h.model_dump() for h in _toolbox().search(query, company_id, filing_id, k)]


@mcp.tool()
def read_table(page_id: str) -> dict[str, Any]:
    nt = _toolbox().read_table(page_id)
    return {
        "page_id": nt.page_id,
        "unit": nt.unit,
        "source": nt.source,
        "columns": [{"index": c.index, "period": str(c.period) if c.period else None, "scope": c.scope} for c in nt.columns],
        "rows": [{"label": r.label, "item": r.item, "values": r.raw} for r in nt.rows],
    }


@mcp.tool()
def get_value(company: str, scope: str, item: str, period: str) -> dict[str, Any]:
    return _toolbox().get_value(company, scope, item, period).model_dump()


@mcp.tool()
def calculate(expression: str, variables: dict[str, float] | None = None) -> float:
    return _toolbox().calculate(expression, variables or {})


@mcp.tool()
def ask(question: str, config: str | None = None) -> dict[str, Any]:
    from pramaan.agent.factory import build_agent

    rt = _rt()
    offline = os.environ.get("PRAMAAN_MODEL_SMALL", "").startswith("offline/")
    cfg = config or ("rules_only" if offline else "cascade")
    with _lock:
        agent = _state.setdefault("agents", {}).setdefault(cfg, build_agent(rt, cfg, production=True))
    ep = agent.run(question, utcnow())
    return {
        "answer": ep.answer.model_dump() if ep.answer else None,
        "verified": bool(ep.verify and ep.verify.status != "hard_fail"),
        "confidence": ep.confidence,
        "cost_usd": ep.cost_usd,
        "config": cfg,
    }


@mcp.resource("filing://{filing_id}/page/{n}")
def filing_page(filing_id: str, n: str) -> str:
    p = _rt().store.page(f"{filing_id}#p{int(n)}")
    return p["text"] if p else "page not found"


@mcp.prompt()
def analyst_question(company: str, question: str) -> str:
    return (
        f"Answer this analyst question about {company}: {question}\n"
        "Use the Pramaan `ask` tool. In your reply give the number with its unit, name the filing page and quote the printed figure it came from, "
        "and say clearly if Pramaan abstained or the filing was not yet published. Do not give investment advice."
    )


def main(transport: str = "stdio") -> None:
    mcp.run(transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
