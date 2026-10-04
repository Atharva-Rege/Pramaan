from __future__ import annotations

import importlib
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path

from pramaan.core.paths import get_paths


def run_doctor(check_llm: bool, echo: Callable[[str], None] = print) -> bool:
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        echo(f"{'ok  ' if good else 'FAIL'} {name} {detail}")

    line("python", sys.version_info >= (3, 11), platform.python_version())
    for mod in ("pydantic", "duckdb", "lxml", "pdfplumber", "pypdfium2", "bm25s", "faiss", "fastembed", "sentence_transformers", "dspy", "anthropic", "fastapi", "mcp"):
        try:
            m = importlib.import_module(mod)
            line(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as exc:
            line(f"import {mod}", mod in ("fastapi", "mcp", "sentence_transformers"), str(exc)[:80])
    p = get_paths()
    for name, path in (("home", p.home), ("work", p.work)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("x")
            probe.unlink()
            line(f"{name} writable", True, str(path))
        except Exception as exc:
            line(f"{name} writable", False, str(exc))
    line("database", Path(p.db_path).exists(), str(p.db_path))
    line("index", (p.indexes / "main" / "info.json").exists(), str(p.indexes / "main"))
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    echo(f"info ANTHROPIC_API_KEY {'set' if key else 'not set (offline mode only)'}")
    try:
        import httpx

        r = httpx.head("https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv", timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        line("network nsearchives.nseindia.com", r.status_code == 200, str(r.status_code))
    except Exception as exc:
        line("network nsearchives.nseindia.com", False, str(exc)[:80])
    if check_llm:
        ok = _llm_checks(line) and ok
    return ok


def _llm_checks(line: Callable[..., None]) -> bool:
    import dspy

    from pramaan.agent.modules import AgentModules
    from pramaan.core.config import models_config
    from pramaan.llm.client import get_client
    from pramaan.llm.dspy_lm import make_lm

    client = get_client()
    mc = models_config()
    good = True
    for role in mc.roles:
        try:
            res = client.complete(role, messages=[{"role": "user", "content": 'Reply with exactly {"ok": true}'}], max_tokens=60, tag="doctor", use_cache=False)
            line(f"llm role {role}", '"ok"' in res.text, f"{res.model} ${res.cost_usd:.5f} {res.latency_ms}ms")
        except Exception as exc:
            good = False
            line(f"llm role {role}", False, f"{type(exc).__name__}: {str(exc)[:160]}")
    try:
        dspy.configure(lm=make_lm("small", client, "doctor"))
        m = AgentModules(make_lm("small", client, "doctor"))
        it = m.parse_intent("What was TCS's consolidated revenue from operations for the quarter ended 30 June 2026?", "2026-08-01T00:00:00")
        line("dspy intent parse", bool(it.items), str(it.model_dump())[:120])
    except Exception as exc:
        good = False
        line("dspy intent parse", False, f"{type(exc).__name__}: {str(exc)[:160]}")
    return good
