from __future__ import annotations

import json
from typing import Any

from pramaan.agent.factory import Runtime, build_agent
from pramaan.core.paths import get_paths
from pramaan.core.splits import purpose
from pramaan.evals.harness import load_questions
from pramaan.gym.grade import grade

QUOTA = {"T1": 6, "T2": 4, "T3": 3, "T4": 2, "T5": 2, "T6": 3}


def demo_path() -> Any:
    return get_paths().results / "demo" / "episodes.json"


def build_demo(rt: Runtime, config: str = "cascade", n: int = 20, echo: Any = print) -> dict[str, Any]:
    with purpose("dev_eval"):
        qs = load_questions(rt.store, "dev", None, seed=2)
    agent = build_agent(rt, config)
    resolver = lambda name: (rt.corpus.resolve_company(name, limit=1) or [{}])[0].get("company_id")  # noqa: E731
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for q in qs:
        if counts.get(q.tier, 0) >= QUOTA.get(q.tier, 0):
            continue
        events: list[dict[str, Any]] = []
        ep = agent.run(q.text, q.as_of, qid=q.qid, emit=lambda kind, data, ev=events: ev.append({"type": kind, "data": data}))
        g = grade(rt.store, q, ep.answer, resolver)
        if not g.verified_correct:
            continue
        counts[q.tier] = counts.get(q.tier, 0) + 1
        picked.append({"qid": q.qid, "tier": q.tier, "question": q.text, "as_of": q.as_of.isoformat(), "events": events, "answer": ep.answer.model_dump() if ep.answer else None, "cost_usd": ep.cost_usd, "latency_ms": ep.latency_ms})
        echo(f"demo {len(picked)}/{n}: {q.tier} {q.text[:80]}")
        if len(picked) >= n:
            break
    path = demo_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "episodes": picked}, indent=1, default=str), encoding="utf-8")
    return {"episodes": len(picked), "path": str(path)}


def build_demo_from_run(store: Any, run_id: str, n: int = 20, echo: Any = print) -> dict[str, Any]:
    """Demo episodes taken from a stored evaluation run: real, logged traces, no API calls.

    Only verified-correct episodes are kept (the demo is a curated showcase, not a sample), within
    the tier quotas, and episodes that show a retry or an escalation come first so the trace
    panel has something to show. The file records the run it came from."""
    from pramaan.core.schemas import Grade
    from pramaan.data.store import jload

    rows = store.q(
        "SELECT e.qid, e.trajectory, e.answer, e.grade, e.cost_usd, e.latency_ms, q.tier, q.text, q.as_of "
        "FROM episodes e JOIN questions q USING (qid) WHERE e.run_id = ? ORDER BY e.qid",
        [run_id],
    )
    cands = []
    for r in rows:
        if not Grade.model_validate(jload(r["grade"])).verified_correct:
            continue
        events = [{"type": e["type"], "data": e.get("data", {})} for e in jload(r["trajectory"], [])]
        interesting = any(e["type"] in ("reflect", "escalate") for e in events)
        cands.append((not interesting, r, events))
    cands.sort(key=lambda t: t[0])
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for _, r, events in cands:
        if counts.get(r["tier"], 0) >= QUOTA.get(r["tier"], 0):
            continue
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1
        picked.append(
            {
                "qid": r["qid"],
                "tier": r["tier"],
                "question": r["text"],
                "as_of": r["as_of"].isoformat() if hasattr(r["as_of"], "isoformat") else str(r["as_of"]),
                "events": events,
                "answer": jload(r["answer"], None),
                "cost_usd": r["cost_usd"],
                "latency_ms": r["latency_ms"],
            }
        )
        if len(picked) >= n:
            break
    path = demo_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = run_id.split("-dev-")[0].split("-test-")[0]
    path.write_text(json.dumps({"config": config, "run_id": run_id, "source": "stored evaluation run", "episodes": picked}, indent=1, default=str), encoding="utf-8")
    echo(f"demo: {len(picked)} episodes from {run_id}; tiers {counts}; with retry or escalation: {sum(1 for p in picked if any(e['type'] in ('reflect', 'escalate') for e in p['events']))}")
    return {"episodes": len(picked), "path": str(path), "run_id": run_id, "tiers": counts}


def load_demo() -> list[dict[str, Any]]:
    p = demo_path()
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("episodes", [])


def match_demo(question: str, episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    from rapidfuzz import fuzz, process

    if not episodes:
        return None
    best = process.extractOne(question, [e["question"] for e in episodes], scorer=fuzz.token_set_ratio)
    if best and best[1] >= 92:
        return episodes[best[2]]
    return None
