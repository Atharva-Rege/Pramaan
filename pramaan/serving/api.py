from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pramaan.core.config import budget_config
from pramaan.core.paths import REPO_ROOT, get_paths
from pramaan.core.timeutil import utcnow
from pramaan.serving.demo import load_demo, match_demo


class AskBody(BaseModel):
    question: str
    as_of: str | None = None
    config: str | None = None


class State:
    def __init__(self) -> None:
        self.rt: Any = None
        self.agents: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.llm_mode = os.environ.get("PRAMAAN_LLM", "api")
        self.demo_only = os.environ.get("PRAMAAN_DEMO_ONLY") == "1"
        self.demo = load_demo()

    def runtime(self) -> Any:
        with self.lock:
            if self.rt is None:
                from pramaan.agent.factory import Runtime
                from pramaan.data.store import Store
                from pramaan.llm.cache import LLMStore
                from pramaan.llm.client import LLMClient
                from pramaan.retrieval.index import RetrievalIndex

                if self.llm_mode == "offline" or not os.environ.get("ANTHROPIC_API_KEY"):
                    for r in ("SMALL", "STRONG", "REFLECTION"):
                        os.environ.setdefault(f"PRAMAAN_MODEL_{r}", "offline/rules")
                self.rt = Runtime(Store(read_only=not os.environ.get("PRAMAAN_REVIEW_TOKEN")), RetrievalIndex.load("main"), LLMClient(store=LLMStore(), daily_cap=budget_config().demo_daily_usd_cap))
            return self.rt

    def store(self) -> Any:
        """The database alone, for routes that never need the retrieval index or an LLM client.
        A demo host ships a small database and no index, so these routes must not build the runtime."""
        if self.rt is not None:
            return self.rt.store
        with self.lock:
            if getattr(self, "_store", None) is None:
                from pramaan.data.store import Store

                self._store = Store(read_only=True)
            return self._store

    def agent(self, config: str) -> Any:
        from pramaan.agent.factory import build_agent

        with self.lock:
            if config not in self.agents:
                rt = self.runtime()
                self.agents[config] = build_agent(rt, config, production=True, xbrl_lookup=self._xbrl)
            return self.agents[config]

    def _xbrl(self, company_id: str, scope: str, item: str, period: Any) -> float | None:
        rt = self.runtime()
        r = rt.store.one(
            "SELECT x.value_inr FROM xbrl_facts x JOIN filings f USING (filing_id) WHERE f.company_id = ? AND x.scope = ? AND x.line_item = ? AND x.period_end = ? AND x.duration_months = ? AND x.is_latest LIMIT 1",
            [company_id, scope, item, period.end, period.months],
        )
        if not r:
            return None
        return r["value_inr"] if item.endswith("eps") else r["value_inr"] / 1e7


def create_app() -> FastAPI:
    root = FastAPI(title="Pramaan", docs_url=None, redoc_url=None, openapi_url=None)
    app = FastAPI(title="Pramaan API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    st = State()
    app.state.st = st
    limit_per_hour = int(os.environ.get("PRAMAAN_RATE_LIMIT_PER_HOUR", "20"))

    def rate_limit(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        dq = st.hits[ip]
        while dq and now - dq[0] > 3600:
            dq.popleft()
        if len(dq) >= limit_per_hour:
            raise HTTPException(429, "rate limit reached, try the demo questions")
        dq.append(now)

    def review_auth(x_review_token: str | None = Header(default=None)) -> None:
        expected = os.environ.get("PRAMAAN_REVIEW_TOKEN")
        if not expected or x_review_token != expected:
            raise HTTPException(401, "review token required")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "demo_only": st.demo_only, "llm_mode": st.llm_mode, "demo_questions": len(st.demo), "time": utcnow().isoformat()}

    @app.get("/demo/questions")
    def demo_questions() -> list[dict[str, Any]]:
        return [{"qid": e["qid"], "tier": e["tier"], "question": e["question"], "as_of": e["as_of"]} for e in st.demo]

    def sse(kind: str, data: Any) -> str:
        return f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

    @app.post("/ask")
    async def ask(body: AskBody, request: Request, _: None = Depends(rate_limit)) -> StreamingResponse:
        demo_hit = match_demo(body.question, st.demo)
        over_cap = False
        if not st.demo_only:
            try:
                rt = st.runtime()
                over_cap = rt.client.store.day_spend() >= budget_config().demo_daily_usd_cap
            except Exception:
                over_cap = True

        async def replay() -> Any:
            yield sse("mode", {"mode": "demo", "note": "answer replayed from a cached episode"})
            for ev in demo_hit["events"] if demo_hit else []:
                yield sse(ev["type"], ev["data"])
                await asyncio.sleep(0.25)
            yield sse("done", {"mode": "demo"})

        if st.demo_only or over_cap:
            if demo_hit is None:
                raise HTTPException(503, "live answers are paused (daily budget); pick one of the curated demo questions")
            return StreamingResponse(replay(), media_type="text/event-stream")

        q: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        config = body.config or ("cascade" if os.environ.get("ANTHROPIC_API_KEY") and st.llm_mode != "offline" else "rules_only")
        as_of = datetime.fromisoformat(body.as_of) if body.as_of else utcnow()

        def work() -> None:
            try:
                agent = st.agent(config)
                ep = agent.run(body.question, as_of, emit=lambda k, d: q.put((k, d)))
                q.put(("done", {"mode": "live", "cost_usd": ep.cost_usd, "latency_ms": ep.latency_ms, "config": config}))
            except Exception as exc:
                q.put(("error", {"message": str(exc)}))
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()

        async def stream() -> Any:
            yield sse("mode", {"mode": "live", "config": config})
            while True:
                item = await asyncio.get_event_loop().run_in_executor(None, q.get)
                if item is None:
                    break
                yield sse(item[0], item[1])

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/filings")
    def filings(limit: int = 200) -> list[dict[str, Any]]:
        store = st.store()
        rows = store.q(
            "SELECT f.filing_id, c.name, c.symbol, f.period_end, f.pdf_published_at, f.xbrl_published_at, f.n_pages, f.split, f.role FROM filings f JOIN companies c USING (company_id) "
            "WHERE f.pdf_path IS NOT NULL ORDER BY f.pdf_published_at DESC LIMIT ?",
            [limit],
        )
        return json.loads(json.dumps(rows, default=str))

    @app.get("/filings/{filing_id}")
    def filing(filing_id: str) -> dict[str, Any]:
        store = st.store()
        f = store.filing(filing_id)
        if f is None:
            raise HTTPException(404, "unknown filing")
        pages = store.pages_of(filing_id, "page_id, page_no, page_type, scope, is_results_page, unit, text_source")
        meta = json.loads(f["meta"]) if f.get("meta") else {}
        f = {k: v for k, v in f.items() if k != "meta"}
        return json.loads(json.dumps({"filing": f, "pdf_url": meta.get("pdf_url"), "pages": pages}, default=str))

    @app.get("/pages/{page_id:path}/image")
    def page_image(page_id: str, box: str | None = None) -> Response:
        store = st.store()
        p = store.page(page_id)
        if p is None:
            raise HTTPException(404, "unknown page")
        from PIL import Image, ImageDraw

        img = None
        # The stored path is absolute on the machine that ingested the page, so a restored snapshot
        # looks for the render at the page's own place under this deployment's data home first.
        png = Path(p["png_path"]) if p["png_path"] else None
        if png is None or not png.exists():
            png = get_paths().pages / p["filing_id"] / f"p{p['page_no']}.png"
        if png.exists():
            img = Image.open(png).convert("RGB")
        else:
            f = store.filing(p["filing_id"])
            pdf = get_paths().work / f["pdf_path"] if f and f["pdf_path"] else None
            if pdf and pdf.exists():
                from pramaan.data.pdf import render_page_image

                img = render_page_image(pdf, p["page_no"], 110)
        if img is None:
            raise HTTPException(404, "page image not available in this deployment")
        if box:
            x0, y0, x1, y1 = (float(v) for v in box.split(","))
            sx, sy = img.width / float(p["width"] or img.width), img.height / float(p["height"] or img.height)
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([x0 * sx - 2, y0 * sy - 2, x1 * sx + 2, y1 * sy + 2], outline=(220, 38, 38, 255), width=3, fill=(250, 204, 21, 70))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/pages/{page_id:path}/text")
    def page_text(page_id: str) -> dict[str, Any]:
        store = st.store()
        p = store.page(page_id)
        if p is None:
            raise HTTPException(404, "unknown page")
        return {"page_id": page_id, "text": p["text"], "unit": p["unit"], "scope": p["scope"]}

    def results_dir() -> Path:
        return get_paths().results

    @app.get("/lab/{name}")
    def lab(name: str) -> Any:
        root = results_dir()
        safe = name.replace("..", "")
        candidates = sorted(root.glob(f"{safe}/*.json")) if (root / safe).is_dir() else []
        if not candidates and (root / f"{safe}.json").exists():
            return json.loads((root / f"{safe}.json").read_text(encoding="utf-8"))
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in candidates}

    @app.get("/leaderboard")
    def leaderboard() -> dict[str, Any]:
        # Same rules as the README table (pramaan.render): tagged runs keep their tag, exploratory
        # variants are marked, and the specified configurations are listed first.
        from pramaan.render import _all, _exploratory

        out: dict[str, Any] = {}
        for split in ("dev", "test"):
            rows = []
            for name, d in _all("E3", split).items():
                s = d["summary"]
                tagged = " [" in name
                offline = d["meta"].get("provenance") == "offline-heuristic"
                expl = _exploratory(name)
                group = 1 if expl else (2 if tagged or offline else 0)
                rows.append(
                    (
                        (group, -s["verified_accuracy"]),
                        {
                            "config": f"{name} (exploratory)" if expl else name,
                            "n": s["n"],
                            "accuracy": s["accuracy"],
                            "accuracy_ci": s["accuracy_ci"],
                            "verified_accuracy": s["verified_accuracy"],
                            "verified_accuracy_ci": s["verified_accuracy_ci"],
                            "cost_per_question_usd": s["cost_per_question_usd"],
                            "cost_per_verified_correct_usd": s["cost_per_verified_correct_usd"],
                            "latency_ms_p50": s["latency_ms_p50"],
                            "latency_ms_p95": s["latency_ms_p95"],
                            "abstain_rate": s["abstain_rate"],
                            "provenance": d["meta"].get("provenance"),
                            "exploratory": expl,
                        },
                    )
                )
            out[split] = [r for _, r in sorted(rows, key=lambda t: t[0])]
        return out

    @app.get("/season")
    def season() -> dict[str, Any]:
        store = st.store()
        filings = store.q(
            "SELECT f.filing_id, c.name, f.pdf_published_at, f.xbrl_published_at, f.split FROM filings f JOIN companies c USING (company_id) WHERE f.role = 'target' AND f.pdf_path IS NOT NULL ORDER BY f.pdf_published_at"
        )
        run = store.one("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
        per: dict[str, list[int]] = {}
        graded = correct = 0
        if run:
            for e in store.q(
                "SELECT e.grade, q.meta FROM episodes e JOIN questions q USING (qid) WHERE e.run_id = ?", [run["run_id"]]
            ):
                fid = (json.loads(e["meta"]) or {}).get("filing_id")
                ok = bool(json.loads(e["grade"]).get("correct"))
                graded += 1
                correct += int(ok)
                if fid:
                    row = per.setdefault(fid, [0, 0])
                    row[0] += 1
                    row[1] += int(ok)
        for f in filings:
            a, c = per.get(f["filing_id"], [0, 0])
            f["answered"], f["correct"] = a, c
        return json.loads(
            json.dumps(
                {"filings": filings, "graded_episodes": graded, "correct": correct, "accuracy": (correct / graded) if graded else None, "run": run["run_id"] if run else None},
                default=str,
            )
        )

    @app.get("/review/next", dependencies=[Depends(review_auth)])
    def review_next(kind: str = "human") -> dict[str, Any]:
        rt = st.runtime()
        if kind == "human":
            r = rt.store.one(
                "SELECT q.* FROM questions q JOIN human_checks h USING (qid) WHERE h.reviewer = '__sample__' AND q.qid NOT IN (SELECT qid FROM human_checks WHERE reviewer <> '__sample__') LIMIT 1"
            )
        else:
            r = rt.store.one(
                "SELECT q.* FROM questions q WHERE q.qid IN (SELECT qid FROM episodes WHERE json_extract_string(grade, '$.correct') = 'false' AND json_extract_string(grade, '$.citations_valid') = 'true') "
                "AND q.qid NOT IN (SELECT qid FROM adjudications) LIMIT 1"
            )
        if not r:
            return {"done": True}
        from pramaan.gym.grade import question_from_row

        q = question_from_row(r)
        return json.loads(json.dumps({"question": q.model_dump(), "image": f"/pages/{q.gold_pages[0]}/image" if q.gold_pages else None}, default=str))

    class Review(BaseModel):
        qid: str
        question_ok: bool | None = None
        gold_ok: bool | None = None
        verdict: str | None = None
        note: str = ""
        reviewer: str = "reviewer"

    @app.post("/review", dependencies=[Depends(review_auth)])
    def review_submit(body: Review) -> dict[str, Any]:
        store = st.runtime().store
        if store.read_only:
            raise HTTPException(409, "server started without PRAMAAN_REVIEW_TOKEN")
        if body.verdict:
            store.insert_rows("adjudications", [{"qid": body.qid, "verdict": body.verdict, "note": body.note, "reviewer": body.reviewer, "ts": utcnow()}])
        else:
            store.insert_rows("human_checks", [{"qid": body.qid, "question_ok": body.question_ok, "gold_ok": body.gold_ok, "note": body.note, "reviewer": body.reviewer, "ts": utcnow()}])
        return {"ok": True}

    root.mount("/api", app)
    dist = REPO_ROOT / "web" / "dist"
    if dist.exists():
        root.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    else:

        @root.get("/")
        def landing() -> JSONResponse:
            return JSONResponse({"service": "pramaan", "api": "/api/docs", "web": "build the web app with `cd web && npm run build`"})

    root.state.st = st
    return root
