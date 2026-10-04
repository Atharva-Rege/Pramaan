from __future__ import annotations

from typing import Any

from pramaan.core.config import agent_config, budget_config, models_config
from pramaan.core.errors import NeedsConfirmation
from pramaan.data.store import Store

TOK = {
    "intake": (1300, 160),
    "plan": (1700, 320),
    "compose": (1500, 380),
    "read_table": (3600, 1900),
    "reflect": (2200, 350),
}


def _price(model: str, tin: float, tout: float) -> float:
    p = models_config().price(model)
    return (tin * p.input + tout * p.output) / 1e6


def estimate(config: str, store: Store, split: str, n_questions: int | None = None, cache_hit_rate_pages: float = 0.0) -> dict[str, Any]:
    mc = models_config()
    n_all = store.count("questions", "split = ?", [split])
    n = min(n_questions, n_all) if n_questions else n_all
    pages = store.one("SELECT count(DISTINCT p) AS n FROM (SELECT unnest(gold_pages) AS p FROM questions WHERE split = ?)", [split])
    distinct_pages = min(pages["n"] if pages else n, n * 2)
    out: dict[str, Any] = {"config": config, "split": split, "n_questions": n}
    if config in ("B0_naive_rag", "B0_strong"):
        role = "small" if config == "B0_naive_rag" else "strong"
        cost = n * _price(mc.role(role).model, 3200, 420)
        out["usd"] = cost
        return out
    v = agent_config().variant(config)
    total = 0.0
    for i, tier in enumerate(v.tiers):
        if tier == "offline":
            continue
        model = mc.role(tier).model
        share = 1.0 if i == 0 else 0.3
        per_q = sum(_price(model, *TOK[k]) for k in ("intake", "plan", "compose")) * share
        reads = distinct_pages * _price(model, *TOK["read_table"]) * (1 - cache_hit_rate_pages) * share * 1.4
        total += n * per_q + reads
        if v.reflect and i == 0:
            total += n * 0.2 * _price(model, *TOK["reflect"])
    out["usd"] = total
    return out


def require_confirmation(est_usd: float, yes: bool) -> None:
    b = budget_config()
    if est_usd > b.confirm_above_usd and not yes:
        raise NeedsConfirmation(f"estimated cost ${est_usd:.2f} is above ${b.confirm_above_usd:.2f}; re-run with --yes to proceed")
