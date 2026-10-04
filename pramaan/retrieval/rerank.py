from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from pramaan.core.config import retrieval_config


def reranker_model_name(model: str | Path | None = None) -> str:
    """The cross-encoder a run will use: explicit argument, then PRAMAAN_RERANKER_PATH, then
    the config's fine-tuned path, then the off-the-shelf model. Recorded in every result file
    because the environment override is not part of the config hash."""
    import os

    cfg = retrieval_config().r3
    return str(model or os.environ.get("PRAMAAN_RERANKER_PATH") or cfg.get("finetuned_path") or cfg["model"])


class Reranker:
    def __init__(self, model: str | Path | None = None, batch_size: int | None = None, max_length: int | None = None) -> None:
        cfg = retrieval_config().r3
        self.model_name = reranker_model_name(model)
        self.batch_size = batch_size or int(cfg.get("batch_size", 32))
        self.max_length = max_length or int(cfg.get("max_length", 384))
        self._model: Any = None
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], float] = {}

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def score(self, query: str, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        todo: list[int] = []
        for i, d in enumerate(doc_ids):
            hit = self._cache.get((query, d))
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
        if todo:
            model = self._load()
            with self._lock:
                preds = model.predict(
                    [(query, texts[i]) for i in todo],
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            for i, p in zip(todo, preds, strict=True):
                out[i] = float(p)
                self._cache[(query, doc_ids[i])] = float(p)
        return out
