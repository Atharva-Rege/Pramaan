from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if any(p in gold for p in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    for i, p in enumerate(ranked):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def summarize(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}


def evaluate_ranking(ranked: Sequence[str], gold: set[str]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked, gold, 1),
        "recall@3": recall_at_k(ranked, gold, 3),
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "mrr": reciprocal_rank(ranked, gold),
        "ndcg@10": ndcg_at_k(ranked, gold, 10),
    }
