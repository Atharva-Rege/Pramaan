from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from pramaan import __version__
from pramaan.agent.graph import Agent, confidence_features
from pramaan.agent.tools import clear_table_cache
from pramaan.core.config import config_hash, models_config
from pramaan.core.errors import PramaanError
from pramaan.core.ids import stable_hash
from pramaan.core.paths import get_paths
from pramaan.core.schemas import Episode, Grade, Question
from pramaan.core.splits import check_split_access, splits_hash
from pramaan.core.timeutil import iso, utcnow
from pramaan.data.store import Store, jdump
from pramaan.evals.stats import (
    bootstrap_mean_ci,
    expected_calibration_error,
    percentile,
    risk_coverage,
)
from pramaan.gym.grade import grade, question_from_row
from pramaan.retrieval.rerank import reranker_model_name


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, cwd=get_paths().home).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def load_questions(
    store: Store,
    split: str,
    subset: int | None = None,
    seed: int = 7,
    tiers: list[str] | None = None,
    qids: list[str] | None = None,
) -> list[Question]:
    check_split_access(split, f"load_questions subset={subset}")
    rows = store.q("SELECT * FROM questions WHERE split = ? ORDER BY as_of, qid", [split])
    qs = [question_from_row(r) for r in rows]
    if tiers:
        qs = [q for q in qs if q.tier in tiers]
    if qids:
        want = set(qids)
        qs = [q for q in qs if q.qid in want]
    if subset and subset < len(qs):
        rng = random.Random(seed)
        picked = rng.sample(range(len(qs)), subset)
        qs = [qs[i] for i in sorted(picked)]
    return qs


@dataclass
class EvalRecord:
    q: Question
    ep: Episode
    g: Grade

    @property
    def filing(self) -> str:
        return str(self.q.meta.get("filing_id", self.q.qid))


def run_episodes(
    agent: Agent,
    questions: list[Question],
    store: Store,
    run_id: str,
    workers: int = 4,
    resolver: Callable[[str], str | None] | None = None,
    progress: Callable[[int, int, EvalRecord], None] | None = None,
    persist: bool = True,
    lessons_for: Callable[[Question], str | None] | None = None,
) -> list[EvalRecord]:
    records: list[EvalRecord | None] = [None] * len(questions)

    def one(i: int) -> tuple[int, EvalRecord]:
        q = questions[i]
        ep = agent.run(q.text, q.as_of, qid=q.qid, run_id=run_id, lessons_override=lessons_for(q) if lessons_for else None)
        g = grade(store, q, ep.answer, resolver)
        return i, EvalRecord(q, ep, g)

    done = 0
    if workers <= 1:
        for i in range(len(questions)):
            _, rec = one(i)
            records[i] = rec
            done += 1
            if progress:
                progress(done, len(questions), rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, i) for i in range(len(questions))]
            for fut in as_completed(futs):
                i, rec = fut.result()
                records[i] = rec
                done += 1
                if progress:
                    progress(done, len(questions), rec)
    out = [r for r in records if r is not None]
    if persist:
        persist_episodes(store, run_id, out)
    return out


def persist_episodes(store: Store, run_id: str, records: list[EvalRecord]) -> None:
    store.x("DELETE FROM episodes WHERE run_id = ?", [run_id])
    rows = []
    for r in records:
        rows.append(
            {
                "run_id": run_id,
                "qid": r.q.qid,
                "trajectory": jdump([e.model_dump() for e in r.ep.trace]),
                "answer": jdump(r.ep.answer.model_dump() if r.ep.answer else None),
                "verify": jdump(r.ep.verify.model_dump() if r.ep.verify else None),
                "grade": jdump(r.g.model_dump()),
                "diagnostics": jdump(
                    {
                        "intent": r.ep.intent.model_dump() if r.ep.intent else None,
                        "plan": r.ep.plan.model_dump() if r.ep.plan else None,
                        "values": [v.model_dump() for v in r.ep.value_results],
                        "search": r.ep.search_log,
                        "tier_used": r.ep.tier_used,
                        "escalated": r.ep.escalated,
                        "retries": r.ep.retries,
                        "abstained": r.ep.abstained,
                        "confidence": r.ep.confidence,
                        "node_cost": r.ep.node_cost,
                        "node_latency_ms": r.ep.node_latency_ms,
                        "drafts": {k: v.model_dump() for k, v in r.ep.draft_by_tier.items()},
                        "lessons_used": r.ep.lessons_used,
                        "error": r.ep.error,
                        "features": confidence_features(r.ep, r.ep.answer, r.ep.verify) if r.ep.answer else None,
                    }
                ),
                "cost_usd": r.ep.cost_usd,
                "latency_ms": r.ep.latency_ms,
            }
        )
    store.insert_rows("episodes", rows)


def agent_errors(records: list[EvalRecord]) -> list[EvalRecord]:
    """Episodes that raised an unexpected agent error (a spent budget, a missing key, a bug). The
    agent turns these into abstentions, so a result written from them looks like real accuracy."""
    return [r for r in records if r.ep.error]


def error_summary(records: list[EvalRecord]) -> str:
    errs = agent_errors(records)
    return f"{len(errs)} of {len(records)} episodes raised an agent error, for example {errs[0].ep.error[:100]!r}" if errs else ""


def run_meta(config: str, split: str, llm_mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    mc = models_config()
    return {
        "config": config,
        "split": split,
        "split_hash": splits_hash(),
        "config_hash": config_hash(),
        "git_sha": git_sha(),
        "version": __version__,
        "models": {r: mc.role(r).model for r in mc.roles},
        "reranker": reranker_model_name(),
        "llm_mode": llm_mode,
        "provenance": "offline-heuristic" if llm_mode == "offline" else "api",
        "created_at": iso(utcnow()),
        **(extra or {}),
    }


def summarize(records: list[EvalRecord]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    clusters = [r.filing for r in records]
    acc = [1.0 if r.g.correct else 0.0 for r in records]
    ver = [1.0 if r.g.verified_correct else 0.0 for r in records]
    cost = [r.ep.cost_usd for r in records]
    lat = [r.ep.latency_ms for r in records]
    mean_acc, acc_lo, acc_hi = bootstrap_mean_ci(acc, clusters)
    mean_ver, ver_lo, ver_hi = bootstrap_mean_ci(ver, clusters)
    total_cost = float(sum(cost))
    n_ver = int(sum(ver))
    by_tier: dict[str, dict[str, float]] = {}
    for t in sorted({r.q.tier for r in records}):
        sub = [r for r in records if r.q.tier == t]
        by_tier[t] = {
            "n": float(len(sub)),
            "accuracy": sum(r.g.correct for r in sub) / len(sub),
            "verified_accuracy": sum(r.g.verified_correct for r in sub) / len(sub),
            "abstain_rate": sum(r.ep.abstained for r in sub) / len(sub),
        }
    conf = [r.ep.confidence for r in records]
    rc = risk_coverage(conf, [r.g.correct for r in records])
    t6 = [r for r in records if r.q.tier == "T6"]
    answerable = [r for r in records if r.q.gold.answer_type in ("number", "ranking")]
    false_abstain = [r for r in answerable if r.ep.abstained]
    return {
        "n": n,
        "accuracy": mean_acc,
        "accuracy_ci": [acc_lo, acc_hi],
        "verified_accuracy": mean_ver,
        "verified_accuracy_ci": [ver_lo, ver_hi],
        "citation_valid_rate": sum(r.g.citations_valid for r in records) / n,
        "abstain_rate": sum(r.ep.abstained for r in records) / n,
        "false_abstain_rate": len(false_abstain) / max(1, len(answerable)),
        "t6_accuracy": (sum(r.g.correct for r in t6) / len(t6)) if t6 else None,
        "by_tier": by_tier,
        "cost_total_usd": round(total_cost, 4),
        "cost_per_question_usd": round(total_cost / n, 5),
        "cost_per_verified_correct_usd": round(total_cost / n_ver, 5) if n_ver else None,
        "latency_ms_p50": percentile(lat, 50),
        "latency_ms_p95": percentile(lat, 95),
        "escalation_rate": sum(r.ep.escalated for r in records) / n,
        "retry_rate": sum(1 for r in records if r.ep.retries > 0) / n,
        "errors": sum(1 for r in records if r.ep.error),
        "ece": expected_calibration_error(conf, [r.g.correct for r in records]),
        "aurc": rc["aurc"],
        "selective_accuracy_at_90_coverage": rc["selective_acc_at_90"],
        "risk_coverage": {"coverage": rc["coverage"], "risk": rc["risk"]},
    }


def per_question_rows(records: list[EvalRecord]) -> list[dict[str, Any]]:
    return [
        {
            "qid": r.q.qid,
            "tier": r.q.tier,
            "filing": r.filing,
            "correct": r.g.correct,
            "verified": r.g.verified_correct,
            "citations_valid": r.g.citations_valid,
            "abstained": r.ep.abstained,
            "confidence": r.ep.confidence,
            "cost_usd": r.ep.cost_usd,
            "latency_ms": r.ep.latency_ms,
            "tier_used": r.ep.tier_used,
            "escalated": r.ep.escalated,
            "error_kind": r.g.error_kind,
        }
        for r in records
    ]


def write_result(exp: str, name: str, payload: dict[str, Any]) -> str:
    out_dir = get_paths().results / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return str(path)


def make_run_id(config: str, split: str) -> str:
    return f"{config}-{split}-{stable_hash([config, split, time.time()], 6)}"


def register_run(store: Store, run_id: str, config: str, split: str, cost: float, meta: dict[str, Any]) -> None:
    store.upsert(
        "runs",
        ["run_id"],
        {
            "run_id": run_id,
            "config_hash": meta.get("config_hash"),
            "git_sha": meta.get("git_sha"),
            "split_hash": meta.get("split_hash"),
            "started_at": utcnow(),
            "cost_usd": cost,
            "name": config,
            "split": split,
            "meta": jdump(meta),
        },
    )


def evaluate(
    agent: Agent,
    config: str,
    split: str,
    store: Store,
    exp: str = "E3",
    subset: int | None = None,
    workers: int = 4,
    llm_mode: str = "api",
    tiers: list[str] | None = None,
    seed: int = 7,
    progress: Callable[[int, int, EvalRecord], None] | None = None,
    extra_meta: dict[str, Any] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    from pramaan.core.splits import purpose

    corpus = agent.corpus
    # A run pays for its own table reads: the in-process cache is shared across configs that
    # use the same reader, which hid read costs from every config after the first.
    clear_table_cache()

    def resolver(name_: str) -> str | None:
        hits = corpus.resolve_company(name_, limit=1)
        return hits[0]["company_id"] if hits else None

    why = "final_eval" if split == "test" else "dev_eval"
    with purpose(why):
        qs = load_questions(store, split, subset, seed, tiers)
        if not qs:
            raise PramaanError(f"no questions in split {split}; run `pramaan gym build` first")
        run_id = make_run_id(config, split)
        t0 = time.time()
        records = run_episodes(agent, qs, store, run_id, workers, resolver, progress)
    if agent_errors(records):
        raise PramaanError(f"{config} on {split}: {error_summary(records)}; no result written")
    meta = run_meta(config, split, llm_mode, {"subset": subset, "n_questions_available": len(qs), "wall_seconds": round(time.time() - t0, 1), **(extra_meta or {})})
    summary = summarize(records)
    register_run(store, run_id, config, split, summary["cost_total_usd"], meta)
    payload = {"meta": meta, "run_id": run_id, "summary": summary, "per_question": per_question_rows(records)}
    path = write_result(exp, name or f"{config}_{split}", payload)
    payload["path"] = path
    return payload
