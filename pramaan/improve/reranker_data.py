from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from pramaan.core.config import retrieval_config
from pramaan.core.ids import stable_hash
from pramaan.core.paths import get_paths
from pramaan.core.splits import SplitAccessError, purpose
from pramaan.data.store import Store, jload
from pramaan.gym.grade import question_from_row
from pramaan.retrieval.docs import page_prefix, table_digest
from pramaan.retrieval.queries import question_query


def _page_text(store: Store, page_id: str, cache: dict[str, str]) -> str:
    if page_id in cache:
        return cache[page_id]
    r = store.one(
        "SELECT p.text, p.tbl, p.scope, p.page_type, p.is_results_page, f.period_end, c.name FROM pages p JOIN filings f USING (filing_id) JOIN companies c USING (company_id) WHERE p.page_id = ?",
        [page_id],
    )
    if r is None:
        return ""
    prefix = page_prefix(r["name"], r["scope"], r["page_type"], r["period_end"])
    tbl = jload(r["tbl"], None)
    body = table_digest(tbl) if r["is_results_page"] and tbl else (r["text"] or "")[:1800]
    cache[page_id] = f"{prefix}\n{body}"
    return cache[page_id]


def _assert_train_only(store: Store, all_pages: dict[str, list[str]], used_pages: set[str]) -> None:
    """Hard check behind the filter: no exported page may belong to a dev or test target filing."""
    page_filing = {pid: fid for fid, pids in all_pages.items() for pid in pids}
    held_out = {
        r["filing_id"]
        for r in store.q("SELECT filing_id FROM filings WHERE role = 'target' AND split IN ('dev', 'test')")
    }
    bad = sorted(pid for pid in used_pages if page_filing.get(pid) in held_out)
    if bad:
        raise SplitAccessError(f"reranker data includes {len(bad)} pages of dev or test filings, e.g. {bad[:3]}")


def export_reranker_data(store: Store, out_dir: Path | None = None, seed: int = 11) -> dict[str, Any]:
    cfg = retrieval_config().training
    n_neg = int(cfg.get("hard_negatives_per_positive", 7))
    holdout_frac = float(cfg.get("holdout_fraction", 0.1))
    out_dir = out_dir or get_paths().exports
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with purpose("reranker_export"):
        rows = store.q("SELECT * FROM questions WHERE split = 'train' AND tier IN ('T1', 'T2', 'T3', 'T4', 'T6') AND array_length(gold_pages) > 0")
    names = {c["company_id"]: c["name"] for c in store.q("SELECT company_id, name FROM companies")}
    companies = sorted({(question_from_row(r).meta.get("company_ids") or [""])[0] for r in rows})
    rng.shuffle(companies)
    held = set(companies[: max(1, int(len(companies) * holdout_frac))])
    results_pages = store.q(
        "SELECT p.page_id, p.filing_id, p.scope, f.company_id, f.role, f.split FROM pages p JOIN filings f USING (filing_id) WHERE p.is_results_page AND f.pdf_path IS NOT NULL"
    )
    by_filing: dict[str, list[dict[str, Any]]] = {}
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for p in results_pages:
        by_filing.setdefault(p["filing_id"], []).append(p)
        by_scope.setdefault(p["scope"] or "", []).append(p)
    all_pages: dict[str, list[str]] = {}
    for r in store.q("SELECT page_id, filing_id FROM pages"):
        all_pages.setdefault(r["filing_id"], []).append(r["page_id"])
    cache: dict[str, str] = {}
    flat: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    used_pages: set[str] = set()
    for r in rows:
        q = question_from_row(r)
        cid = (q.meta.get("company_ids") or [""])[0]
        fid = q.meta.get("filing_id")
        query = question_query(q, names.get(cid, cid))
        gold = set(q.gold_pages)
        split_tag = "holdout" if cid in held else "train"
        negs: list[tuple[str, str]] = []
        for pid in all_pages.get(fid, []):
            if pid not in gold:
                negs.append((pid, "same_filing"))
        other_scope = [p["page_id"] for p in by_filing.get(fid, []) if p["page_id"] not in gold]
        negs += [(p, "same_filing_results") for p in other_scope]
        same_company_prev = [p for p in results_pages if p["company_id"] == cid and p["filing_id"] != fid]
        negs += [(p["page_id"], "other_period") for p in same_company_prev[:6]]
        scope = q.slots.scope or ""
        # Other companies' pages must come from train filings only: a page of a dev or test
        # filing is part of that split, so it may not enter reranker training (spec rule 9).
        pool = [p for p in by_scope.get(scope, []) if p["company_id"] != cid and p["split"] == "train"]
        rng.shuffle(pool)
        negs += [(p["page_id"], "other_company") for p in pool[:6]]
        seen: set[str] = set()
        dedup = []
        for pid, kind in negs:
            if pid not in seen and pid not in gold:
                seen.add(pid)
                dedup.append((pid, kind))
        rng.shuffle(dedup)
        hard_first = sorted(dedup, key=lambda t: {"same_filing_results": 0, "other_period": 1, "other_company": 2, "same_filing": 3}[t[1]])
        chosen = hard_first[: n_neg]
        gid = stable_hash([q.qid], 8)
        used_pages.update(q.gold_pages)
        used_pages.update(pid for pid, _ in chosen)
        for pos in q.gold_pages:
            flat.append({"group": gid, "qid": q.qid, "query": query, "text": _page_text(store, pos, cache), "label": 1, "kind": "positive", "split": split_tag})
        for pid, kind in chosen:
            flat.append({"group": gid, "qid": q.qid, "query": query, "text": _page_text(store, pid, cache), "label": 0, "kind": kind, "split": split_tag})
        grouped.append(
            {
                "qid": q.qid,
                "query": query,
                "positive": [_page_text(store, p, cache) for p in q.gold_pages],
                "negatives": [_page_text(store, pid, cache) for pid, _ in chosen],
                "split": split_tag,
            }
        )
    _assert_train_only(store, all_pages, used_pages)
    flat_df = pd.DataFrame(flat)
    grp_df = pd.DataFrame(grouped)
    flat_path, grp_path = out_dir / "reranker_pairs.parquet", out_dir / "reranker_groups.parquet"
    flat_df.to_parquet(flat_path, index=False)
    grp_df.to_parquet(grp_path, index=False)
    return {
        "pairs": len(flat_df),
        "groups": len(grp_df),
        "positives": int((flat_df["label"] == 1).sum()) if len(flat_df) else 0,
        "holdout_groups": int((grp_df["split"] == "holdout").sum()) if len(grp_df) else 0,
        "pairs_path": str(flat_path),
        "groups_path": str(grp_path),
    }
