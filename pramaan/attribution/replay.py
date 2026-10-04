from __future__ import annotations

from collections import Counter
from typing import Any

from pramaan.agent.graph import Agent
from pramaan.agent.injection import Injection
from pramaan.core.ids import split_page_id
from pramaan.core.schemas import Question, ValueResult
from pramaan.data.store import Store
from pramaan.evals.harness import EvalRecord
from pramaan.gym.grade import grade
from pramaan.improve.diagnose import STAGES, diagnose

INJECT_STAGES = ("A1", "A2", "A4")


def gold_value_results(store: Store, q: Question) -> dict[tuple[str, str, str], ValueResult]:
    out: dict[tuple[str, str, str], ValueResult] = {}
    for gf in q.gold.facts:
        if not gf.page_id:
            continue
        fid, _ = split_page_id(gf.page_id)
        ev = store.one(
            "SELECT cell_text, box, row_label FROM gold_evidence WHERE filing_id = ? AND scope = ? AND line_item = ? AND page_id = ? ORDER BY ambiguous LIMIT 1",
            [fid, gf.scope, gf.item, gf.page_id],
        )
        page = store.page(gf.page_id)
        per_share = gf.item in ("basic_eps", "diluted_eps")
        out[(gf.company_id, gf.item, gf.period)] = ValueResult(
            found=True,
            company_id=gf.company_id,
            filing_id=fid,
            scope=gf.scope,
            item=gf.item,
            period=gf.period,
            value=gf.value,
            unit="INR_per_share" if per_share else "INR_crore",
            page_id=gf.page_id,
            quote=(ev["cell_text"] if ev else str(gf.value)),
            column_header=gf.period,
            row_label=(ev["row_label"] if ev else ""),
            page_unit=(page["unit"] if page else None),
            retrieval_rank=1,
        )
    return out


def build_injection(store: Store, q: Question, stage: str, corpus: Any) -> Injection:
    inj = Injection()
    if stage == "A1":
        names = [corpus.company_name(c) for c in q.slots.companies]
        inj.intent = q.slots.model_copy(update={"companies": names})
    elif stage in ("A2", "A3"):
        inj.gold_pages = list(q.gold_pages)
    elif stage == "A4":
        inj.gold_values = gold_value_results(store, q)
    return inj


def attribute_records(store: Store, records: list[EvalRecord], corpus: Any) -> list[dict[str, Any]]:
    out = []
    for r in records:
        if r.g.correct and r.g.citations_valid:
            continue
        d = diagnose(r.q, r.ep, r.g, store, corpus)
        out.append({"qid": r.q.qid, "tier": r.q.tier, "stage": d.stage or "A5", "sub": d.sub_stage, "message": (d.messages or [""])[0]})
    return out


def stage_distribution(attrib: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(a["stage"] for a in attrib)
    n = max(1, sum(c.values()))
    return {"counts": {k: c.get(k, 0) for k in STAGES}, "share": {k: c.get(k, 0) / n for k in STAGES}, "n_failed": sum(c.values())}


def counterfactual_replay(
    store: Store,
    agent: Agent,
    records: list[EvalRecord],
    max_episodes: int = 200,
    stages: tuple[str, ...] = INJECT_STAGES,
) -> dict[str, Any]:
    failed = [r for r in records if not r.g.correct][:max_episodes]
    flips: dict[str, list[bool]] = {s: [] for s in stages}
    detail: list[dict[str, Any]] = []
    for r in failed:
        row: dict[str, Any] = {"qid": r.q.qid}
        for stage in stages:
            if r.q.gold.answer_type == "unavailable" and stage != "A1":
                continue
            inj = build_injection(store, r.q, stage, agent.corpus)
            ep = agent.run(r.q.text, r.q.as_of, qid=r.q.qid, injection=inj)
            g = grade(store, r.q, ep.answer, lambda n: (agent.corpus.resolve_company(n, limit=1) or [{}])[0].get("company_id"))
            flips[stage].append(g.correct)
            row[stage] = g.correct
        detail.append(row)
    return {
        "n_failed": len(failed),
        "flip_rate": {s: (sum(v) / len(v) if v else None) for s, v in flips.items()},
        "attempted": {s: len(v) for s, v in flips.items()},
        "detail": detail,
    }
