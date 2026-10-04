from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np

from pramaan.core.config import retrieval_config
from pramaan.core.paths import get_paths
from pramaan.data.store import Store
from pramaan.retrieval.docs import DocRecord, build_docs

_TOKEN = re.compile(r"[a-z]+|\d[\d,]*\.?\d*")
_lock = threading.Lock()
_EMBEDDER: Any = None


def tokenize(text: str) -> list[str]:
    return [t.replace(",", "") for t in _TOKEN.findall(text.lower())]


def embedder() -> Any:
    global _EMBEDDER
    with _lock:
        if _EMBEDDER is None:
            from fastembed import TextEmbedding

            _EMBEDDER = TextEmbedding(retrieval_config().dense.get("model", "BAAI/bge-small-en-v1.5"))
    return _EMBEDDER


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = list(embedder().embed(texts, batch_size=batch_size))
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def embed_query(text: str) -> np.ndarray:
    prefix = retrieval_config().dense.get("query_prefix", "")
    return embed_texts([prefix + text])[0]


@dataclass
class KindIndex:
    kind: str
    docs: list[DocRecord]
    bm25: bm25s.BM25
    faiss_index: Any
    emb: np.ndarray
    pdf_ts: np.ndarray
    company_idx: np.ndarray
    filing_idx: np.ndarray
    scope_idx: np.ndarray
    companies: list[str]
    filings: list[str]
    scopes: list[str]
    page_to_doc: dict[str, int]

    @classmethod
    def build(cls, kind: str, docs: list[DocRecord], emb: np.ndarray | None = None) -> KindIndex:
        retr = bm25s.BM25()
        retr.index([tokenize(d.text) or ["empty"] for d in docs], show_progress=False)
        if emb is None:
            emb = embed_texts([d.text[:1200] for d in docs]) if docs else np.zeros((0, 384), dtype=np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        if len(docs):
            index.add(emb)
        return cls._finish(kind, docs, retr, index, emb)

    @classmethod
    def _finish(cls, kind: str, docs: list[DocRecord], retr: bm25s.BM25, index: Any, emb: np.ndarray) -> KindIndex:
        companies = sorted({d.company_id for d in docs})
        filings = sorted({d.filing_id for d in docs})
        scopes = ["", "standalone", "consolidated", "both"]
        cmap = {c: i for i, c in enumerate(companies)}
        fmap = {f: i for i, f in enumerate(filings)}
        smap = {s: i for i, s in enumerate(scopes)}
        page_to_doc = {d.page_id: i for i, d in enumerate(docs) if d.kind == "page"}
        return cls(
            kind=kind,
            docs=docs,
            bm25=retr,
            faiss_index=index,
            emb=emb,
            pdf_ts=np.asarray([d.pdf_ts for d in docs], dtype=np.float64),
            company_idx=np.asarray([cmap[d.company_id] for d in docs], dtype=np.int32),
            filing_idx=np.asarray([fmap[d.filing_id] for d in docs], dtype=np.int32),
            scope_idx=np.asarray([smap.get(d.scope or "", 0) for d in docs], dtype=np.int8),
            companies=companies,
            filings=filings,
            scopes=scopes,
            page_to_doc=page_to_doc,
        )

    def mask(
        self,
        company_id: str | None = None,
        filing_id: str | None = None,
        scope: str | None = None,
        as_of_ts: float | None = None,
        roles: tuple[str, ...] | None = None,
        exclude_filings: set[str] | None = None,
    ) -> np.ndarray:
        m = np.ones(len(self.docs), dtype=bool)
        if as_of_ts is not None:
            m &= self.pdf_ts < as_of_ts
        if company_id is not None:
            if company_id in self.companies:
                m &= self.company_idx == self.companies.index(company_id)
            else:
                m &= False
        if filing_id is not None:
            if filing_id in self.filings:
                m &= self.filing_idx == self.filings.index(filing_id)
            else:
                m &= False
        if scope in ("standalone", "consolidated"):
            m &= (self.scope_idx == self.scopes.index(scope)) | (self.scope_idx == self.scopes.index("both")) | (self.scope_idx == 0)
        if roles is not None:
            allowed = np.asarray([d.role in roles for d in self.docs], dtype=bool)
            m &= allowed
        if exclude_filings:
            ex = np.asarray([d.filing_id in exclude_filings for d in self.docs], dtype=bool)
            m &= ~ex
        return m

    def bm25_search(self, query: str, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks or not mask.any():
            return []
        scores = self.bm25.get_scores(toks)
        scores = np.where(mask, scores, -1.0)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def dense_search(self, qvec: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not mask.any():
            return []
        ids = np.nonzero(mask)[0].astype(np.int64)
        if len(ids) <= 5000:
            sims = self.emb[ids] @ qvec
            order = np.argsort(-sims)[:k]
            return [(int(ids[i]), float(sims[i])) for i in order]
        sel = faiss.IDSelectorBatch(ids)
        params = faiss.SearchParameters()
        params.sel = sel
        d, i = self.faiss_index.search(qvec[None, :], k, params=params)
        return [(int(j), float(s)) for j, s in zip(i[0], d[0], strict=True) if j >= 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
        np.save(directory / "emb.npy", self.emb)
        self.bm25.save(str(directory / "bm25"), corpus=None)

    @classmethod
    def load(cls, kind: str, directory: Path) -> KindIndex:
        docs: list[DocRecord] = []
        with open(directory / "docs.jsonl", encoding="utf-8") as f:
            for line in f:
                docs.append(DocRecord(**json.loads(line)))
        emb = np.load(directory / "emb.npy")
        retr = bm25s.BM25.load(str(directory / "bm25"), load_corpus=False)
        index = faiss.IndexFlatIP(emb.shape[1])
        if len(docs):
            index.add(emb)
        return cls._finish(kind, docs, retr, index, emb)


@dataclass
class RetrievalIndex:
    pages: KindIndex
    rows: KindIndex | None

    @classmethod
    def build(cls, store: Store, row_docs: bool = True) -> RetrievalIndex:
        docs = build_docs(store, row_docs=row_docs)
        page_docs = [d for d in docs if d.kind == "page"]
        row_docs_l = [d for d in docs if d.kind == "row"]
        pages = KindIndex.build("page", page_docs)
        rows = KindIndex.build("row", row_docs_l) if row_docs and row_docs_l else None
        return cls(pages, rows)

    def save(self, name: str = "main") -> Path:
        root = get_paths().indexes / name
        self.pages.save(root / "pages")
        if self.rows is not None:
            self.rows.save(root / "rows")
        (root / "info.json").write_text(
            json.dumps({"pages": len(self.pages.docs), "rows": len(self.rows.docs) if self.rows else 0}),
            encoding="utf-8",
        )
        return root

    @classmethod
    def load(cls, name: str = "main") -> RetrievalIndex:
        root = get_paths().indexes / name
        pages = KindIndex.load("page", root / "pages")
        rows = KindIndex.load("row", root / "rows") if (root / "rows").exists() else None
        return cls(pages, rows)
