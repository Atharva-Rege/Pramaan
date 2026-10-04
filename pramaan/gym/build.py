from __future__ import annotations

import random
from collections import Counter
from typing import Any

from pramaan.core.config import gym_config
from pramaan.core.ids import stable_hash
from pramaan.core.periods import PeriodKey
from pramaan.core.splits import purpose
from pramaan.data.store import Store, jdump
from pramaan.gym.evidence import compute_filing_evidence
from pramaan.gym.generate import GymBuilder, select_questions, write_questions


def cache_evidence(store: Store, echo: Any = print) -> int:
    cfg = gym_config()
    cur = PeriodKey(cfg.profile.target_period_end, 3)
    store.x("DELETE FROM gold_evidence")
    total = 0
    for f in store.filings("role = 'target' AND pdf_path IS NOT NULL"):
        rows = compute_filing_evidence(store, f["filing_id"], cur)
        store.insert_rows("gold_evidence", rows)
        total += len(rows)
    echo(f"gold evidence rows: {total}")
    return total


def evidence_coverage(store: Store) -> dict[str, Any]:
    from pramaan.core.config import line_items

    li = line_items()
    cfg = gym_config()
    cur = PeriodKey(cfg.profile.target_period_end, 3)
    elig = set(cfg.t1_items) | {"revenue_from_operations", "net_profit_for_period"}
    agg = {"digital": [0, 0], "ocr_layer": [0, 0], "no_results_page": [0, 0]}
    per_item: dict[str, list[int]] = {}
    for f in store.filings("role = 'target' AND pdf_path IS NOT NULL"):
        fid = f["filing_id"]
        if store.count("pages", "filing_id = ?", [fid]) == 0:
            continue
        rp = store.q("SELECT ocr_layer, text_source FROM pages WHERE filing_id = ? AND is_results_page", [fid])
        kind = "no_results_page" if not rp else ("ocr_layer" if any(r["ocr_layer"] or r["text_source"] != "text_layer" for r in rp) else "digital")
        facts = store.q(
            "SELECT scope, line_item, value_inr FROM xbrl_facts WHERE filing_id = ? AND is_latest AND period_end = ? AND duration_months = ?",
            [fid, cur.end, cur.months],
        )
        got = {(r["scope"], r["line_item"]) for r in store.q("SELECT scope, line_item FROM gold_evidence WHERE filing_id = ?", [fid])}
        for x in facts:
            if x["line_item"] not in elig or abs(x["value_inr"]) < 1e-9 or li.kind(x["line_item"]) is None:
                continue
            hit = (x["scope"], x["line_item"]) in got
            agg[kind][0] += int(hit)
            agg[kind][1] += 1
            pi = per_item.setdefault(x["line_item"], [0, 0])
            pi[0] += int(hit)
            pi[1] += 1
    return {
        "by_page_kind": {k: {"located": v[0], "facts": v[1], "rate": (v[0] / v[1]) if v[1] else None} for k, v in agg.items()},
        "by_item": {k: {"located": v[0], "facts": v[1], "rate": v[0] / v[1]} for k, v in sorted(per_item.items())},
    }


def build_gym(store: Store, sizes: dict[str, int] | None = None, echo: Any = print) -> dict[str, Any]:
    cfg = gym_config()
    with purpose("gym_build"):
        gb = GymBuilder(store, cfg)
        cands = gb.build_candidates()
        pool = Counter((c.split, c.tier) for c in cands)
        echo(f"candidates: {len(cands)}")
        want = sizes or cfg.profile.sizes
        picks = select_questions(cands, want, cfg.tier_mix, cfg.seed)
        n = write_questions(store, picks)
    counts = Counter((q.split, q.tier) for q in picks)
    by_split = Counter(q.split for q in picks)
    echo(f"questions written: {n} {dict(by_split)}")
    sample_ids = sample_for_human_check(store, cfg.human_check_sample, cfg.seed)
    return {
        "profile": cfg.profile.name,
        "requested": want,
        "written": n,
        "by_split": dict(by_split),
        "by_split_tier": {f"{s}/{t}": c for (s, t), c in sorted(counts.items())},
        "candidate_pool": {f"{s}/{t}": c for (s, t), c in sorted(pool.items())},
        "human_check_sample": len(sample_ids),
        "approx_evidence_share": sum(1 for q in picks if q.meta.get("approx")) / max(1, len(picks)),
    }


def sample_for_human_check(store: Store, n: int, seed: int) -> list[str]:
    rows = store.q("SELECT qid FROM questions WHERE split IN ('dev', 'train') ORDER BY qid")
    rng = random.Random(int(stable_hash([seed, "human"], 8), 16))
    ids = [r["qid"] for r in rows]
    rng.shuffle(ids)
    chosen = ids[:n]
    store.x("DELETE FROM human_checks WHERE reviewer = '__sample__'")
    store.insert_rows(
        "human_checks",
        [{"qid": q, "question_ok": None, "gold_ok": None, "note": "", "reviewer": "__sample__", "ts": None} for q in chosen],
    )
    return chosen


def gym_stats(store: Store) -> dict[str, Any]:
    rows = store.q("SELECT split, tier, count(*) AS n FROM questions GROUP BY split, tier ORDER BY split, tier")
    return {"counts": rows, "total": store.count("questions"), "meta": jdump({})}
