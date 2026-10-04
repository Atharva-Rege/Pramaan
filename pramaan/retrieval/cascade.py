from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from pramaan.core.config import retrieval_config
from pramaan.retrieval.index import KindIndex, RetrievalIndex, embed_query
from pramaan.retrieval.rerank import Reranker


@dataclass
class Hit:
    page_id: str
    score: float
    stages: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]
    n_candidates: int = 0
    margin: float = 0.0

    def page_ids(self) -> list[str]:
        return [h.page_id for h in self.hits]


STAGE_PRESETS: dict[str, dict[str, bool]] = {
    "bm25": {"bm25_page": True},
    "dense": {"dense_page": True},
    "bm25+dense": {"bm25_page": True, "dense_page": True, "rrf": True},
    "bm25+dense+rows": {"bm25_page": True, "dense_page": True, "bm25_row": True, "dense_row": True, "rrf": True},
    "rrf+ce": {"bm25_page": True, "dense_page": True, "bm25_row": True, "dense_row": True, "rrf": True, "ce": True},
    "rrf+rows+ce+llm": {"bm25_page": True, "dense_page": True, "bm25_row": True, "dense_row": True, "rrf": True, "ce": True, "llm": True},
    "rrf+ce_norows": {"bm25_page": True, "dense_page": True, "rrf": True, "ce": True},
}


def rrf(lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, pid in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _pages_from(idx: KindIndex, hits: list[tuple[int, float]]) -> list[str]:
    seen: list[str] = []
    for i, _ in hits:
        pid = idx.docs[i].page_id
        if pid not in seen:
            seen.append(pid)
    return seen


class Retriever:
    def __init__(
        self,
        index: RetrievalIndex,
        reranker: Reranker | None = None,
        llm_rerank: Callable[[str, list[tuple[str, str]]], list[str]] | None = None,
    ) -> None:
        self.index = index
        self.cfg = retrieval_config()
        self.reranker = reranker
        self.llm_rerank = llm_rerank

    def _reranker(self) -> Reranker:
        if self.reranker is None:
            self.reranker = Reranker()
        return self.reranker

    def search(
        self,
        query: str,
        *,
        company_id: str | None = None,
        filing_id: str | None = None,
        scope: str | None = None,
        as_of: datetime | None = None,
        k: int = 10,
        stages: str | dict[str, bool] = "rrf+ce",
        roles: tuple[str, ...] | None = None,
        exclude_filings: set[str] | None = None,
    ) -> SearchResult:
        st = STAGE_PRESETS[stages] if isinstance(stages, str) else stages
        timings: dict[str, float] = {}
        as_of_ts = as_of.timestamp() if as_of else None
        r1, r2, r3 = self.cfg.r1, self.cfg.r2, self.cfg.r3
        lists: list[list[str]] = []
        names: list[str] = []
        qvec: np.ndarray | None = None
        page_mask = self.index.pages.mask(company_id, filing_id, scope, as_of_ts, roles, exclude_filings)
        row_mask = (
            self.index.rows.mask(company_id, filing_id, scope, as_of_ts, roles, exclude_filings)
            if self.index.rows is not None
            else None
        )
        n_cand = int(page_mask.sum())

        t0 = time.perf_counter()
        if st.get("bm25_page"):
            lists.append(_pages_from(self.index.pages, self.index.pages.bm25_search(query, page_mask, r1["bm25_k"])))
            names.append("bm25_page")
        if st.get("bm25_row") and self.index.rows is not None and row_mask is not None:
            lists.append(_pages_from(self.index.rows, self.index.rows.bm25_search(query, row_mask, r1["bm25_k"])))
            names.append("bm25_row")
        timings["bm25"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if st.get("dense_page") or st.get("dense_row"):
            qvec = embed_query(query)
        if st.get("dense_page") and qvec is not None:
            lists.append(_pages_from(self.index.pages, self.index.pages.dense_search(qvec, page_mask, r1["dense_k"])))
            names.append("dense_page")
        if st.get("dense_row") and qvec is not None and self.index.rows is not None and row_mask is not None:
            lists.append(_pages_from(self.index.rows, self.index.rows.dense_search(qvec, row_mask, r1["dense_k"])))
            names.append("dense_row")
        timings["dense"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if len(lists) > 1 or st.get("rrf"):
            fused = rrf(lists, int(r2["rrf_k"]))
        elif lists:
            n = len(lists[0])
            fused = {pid: float(n - i) for i, pid in enumerate(lists[0])}
        else:
            fused = {}
        ranked = sorted(fused, key=lambda p: -fused[p])[: int(r2["top"])]
        timings["fusion"] = (time.perf_counter() - t0) * 1000
        hits = [Hit(p, fused[p], {"fusion": fused[p]}) for p in ranked]

        if st.get("ce") and hits:
            t0 = time.perf_counter()
            texts = [self.index.pages.docs[self.index.pages.page_to_doc[h.page_id]].ce_text for h in hits]
            scores = self._reranker().score(query, [h.page_id for h in hits], texts)
            for h, sc in zip(hits, scores, strict=True):
                h.stages["ce"] = float(sc)
                h.score = float(sc)
            hits.sort(key=lambda h: -h.score)
            hits = hits[: int(r3["top"])]
            timings["ce"] = (time.perf_counter() - t0) * 1000

        if st.get("llm") and self.llm_rerank is not None and hits:
            t0 = time.perf_counter()
            cands = [(h.page_id, self.index.pages.docs[self.index.pages.page_to_doc[h.page_id]].ce_text[:900]) for h in hits[:10]]
            order = self.llm_rerank(query, cands)
            pos = {pid: i for i, pid in enumerate(order)}
            hits = sorted(hits, key=lambda h: pos.get(h.page_id, 99))
            for i, h in enumerate(hits):
                h.stages["llm_rank"] = float(i)
            timings["llm"] = (time.perf_counter() - t0) * 1000

        hits = hits[:k]
        margin = float(hits[0].score - hits[1].score) if len(hits) > 1 else (float(hits[0].score) if hits else 0.0)
        return SearchResult(hits=hits, timings_ms=timings, n_candidates=n_cand, margin=margin)


def stage_names() -> list[str]:
    return list(STAGE_PRESETS)


def make_filters(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}
