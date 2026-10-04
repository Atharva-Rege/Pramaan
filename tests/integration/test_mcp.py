from __future__ import annotations

import asyncio
import os
import sys

import pytest


@pytest.mark.slow
def test_mcp_server_lists_tools_over_stdio(ingested, tmp_path):
    mcp_client = pytest.importorskip("mcp.client.stdio")
    from mcp import ClientSession

    home, info, store, summary = ingested
    import shutil

    from pramaan.retrieval.index import RetrievalIndex

    RetrievalIndex.build(store).save("main")
    store.con.execute("CHECKPOINT")
    copy = tmp_path / "home"
    (copy / "data" / "db").mkdir(parents=True)
    shutil.copy2(home / "data" / "db" / "pramaan.duckdb", copy / "data" / "db" / "pramaan.duckdb")
    shutil.copytree(home / "indexes", copy / "indexes")
    env = {**os.environ, "PRAMAAN_HOME": str(copy), "PRAMAAN_LLM": "offline", "PRAMAAN_MODEL_SMALL": "offline/rules", "PRAMAAN_MODEL_STRONG": "offline/rules", "PRAMAAN_MODEL_REFLECTION": "offline/rules"}
    params = mcp_client.StdioServerParameters(command=sys.executable, args=["-m", "pramaan.mcp_server.server"], env=env)

    async def go():
        async with mcp_client.stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            res = await session.call_tool("resolve_company", {"name_or_ticker": "Alpha Industries"})
            return names, res

    names, res = asyncio.run(go())
    assert {"ask", "get_value", "search", "calculate", "read_table", "list_filings", "resolve_company"} <= names
    assert not getattr(res, "is_error", getattr(res, "isError", False))
