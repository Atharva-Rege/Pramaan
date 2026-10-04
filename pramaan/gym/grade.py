from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from scipy.stats import kendalltau

from pramaan.core.config import gym_config
from pramaan.core.numbers import find_numbers, parse_number
from pramaan.core.schemas import Answer, Citation, Grade, Question
from pramaan.data.store import Store
from pramaan.data.tables import UNIT_FACTOR_CRORE

UNIT_FACTORS = (1e-7, 1e-4, 1e-2, 1e-1, 1.0)


def _ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def money_correct(pred: float, gold: float, display_unit_crore: float) -> bool:
    tol = max(gym_config().tolerances["money_floor_crore"], display_unit_crore / 2) + 1e-9
    return abs(pred - gold) <= tol


def percent_correct(pred: float, gold: float) -> bool:
    return abs(pred - gold) <= gym_config().tolerances["percent_points"] + 1e-9


def per_share_correct(pred: float, gold: float) -> bool:
    return abs(pred - gold) <= gym_config().tolerances["per_share"] + 1e-9


def value_correct(q: Question, ans: Answer) -> tuple[bool, str]:
    g = q.gold
    if g.value is None or ans.value is None:
        return False, "no_value"
    if ans.unit is not None and g.unit is not None and ans.unit != g.unit:
        return False, "wrong_unit"
    if g.unit == "percent":
        ok = percent_correct(ans.value, g.value)
    elif g.unit == "INR_per_share":
        ok = per_share_correct(ans.value, g.value)
    else:
        ok = money_correct(ans.value, g.value, g.display_unit_crore)
    return ok, "" if ok else "wrong_value"


def ranking_correct(q: Question, ans: Answer, resolver: Callable[[str], str | None]) -> tuple[bool, float | None]:
    gold = q.gold.ranking
    resolved = [resolver(r) for r in ans.ranking]
    if len(resolved) != len(gold) or any(r is None for r in resolved):
        return False, None
    pos = {c: i for i, c in enumerate(gold)}
    try:
        pred_ranks = [pos[r] for r in resolved if r is not None]
    except KeyError:
        return False, None
    exact = pred_ranks == list(range(len(gold)))
    tau: float | None = None
    if len(gold) >= 2:
        res = kendalltau(list(range(len(gold))), pred_ranks)
        tau = float(res.statistic) if res.statistic == res.statistic else None
    return exact, tau


def citation_valid(
    store: Store,
    c: Citation,
    value_crore: float | None,
    per_share: bool,
    display_unit_crore: float,
) -> tuple[bool, str]:
    page = store.page(c.page_id) if c.page_id else None
    if page is None:
        return False, "unknown_page"
    text = page["text"] or ""
    quote = c.quote.strip()
    if not quote:
        return False, "empty_quote"
    if _ws(quote) not in _ws(text):
        return False, "quote_not_on_page"
    if value_crore is None:
        return True, ""
    toks = find_numbers(quote)
    qv = toks[0].value if toks else parse_number(quote)
    if qv is None:
        return False, "quote_not_numeric"
    if per_share:
        return (abs(qv - value_crore) <= 0.0051 or abs(abs(qv) - abs(value_crore)) <= 0.0051), ""
    tol = max(0.01, display_unit_crore / 2) + 1e-9
    page_unit = page.get("unit")
    if page_unit and page_unit in UNIT_FACTOR_CRORE:
        f = UNIT_FACTOR_CRORE[page_unit]
        return (abs(qv * f - value_crore) <= tol), "" if abs(qv * f - value_crore) <= tol else "value_mismatch_under_page_unit"
    for f in UNIT_FACTORS:
        if abs(qv * f - value_crore) <= tol:
            return True, ""
    return False, "value_mismatch"


def citations_valid(store: Store, q: Question, ans: Answer) -> tuple[bool, str]:
    if ans.answer_type in ("unavailable", "premise_false"):
        for c in ans.citations:
            ok, why = citation_valid(store, c, None, False, 0.01)
            if not ok:
                return False, why
        return True, ""
    if not ans.citations:
        return False, "no_citations"
    by_id = {c.id: c for c in ans.citations}
    per_share = (q.gold.unit == "INR_per_share")
    checked = 0
    for inp in ans.inputs:
        c = by_id.get(inp.citation_id or "")
        if c is None:
            return False, f"input_{inp.name}_uncited"
        ok, why = citation_valid(store, c, inp.value, per_share and len(ans.inputs) == 1, q.gold.display_unit_crore)
        if not ok:
            return False, why
        checked += 1
    if checked == 0:
        for c in ans.citations:
            ok, why = citation_valid(store, c, None, False, q.gold.display_unit_crore)
            if not ok:
                return False, why
    return True, ""


def grade(
    store: Store,
    q: Question,
    ans: Answer | None,
    resolver: Callable[[str], str | None] | None = None,
) -> Grade:
    if ans is None:
        return Grade(correct=False, citations_valid=False, verified_correct=False, abstained=False, error_kind="no_answer")
    abstained = ans.answer_type == "unavailable"
    g = q.gold
    kind = ""
    tau: float | None = None
    if g.answer_type in ("unavailable", "premise_false"):
        correct = ans.answer_type == g.answer_type
        kind = "" if correct else ("should_abstain" if g.answer_type == "unavailable" else "missed_false_premise")
    elif g.answer_type == "ranking":
        if ans.answer_type != "ranking":
            correct, kind = False, "wrong_type"
        else:
            correct, tau = ranking_correct(q, ans, resolver or (lambda s: s))
            kind = "" if correct else "wrong_ranking"
    else:
        if ans.answer_type == "unavailable":
            correct, kind = False, "false_abstain"
        elif ans.answer_type != "number":
            correct, kind = False, "wrong_type"
        else:
            correct, kind = value_correct(q, ans)
    valid, why = citations_valid(store, q, ans)
    return Grade(
        correct=correct,
        citations_valid=valid,
        verified_correct=correct and valid,
        abstained=abstained,
        kendall_tau=tau,
        error_kind=kind,
        detail=why,
    )


def question_from_row(row: dict[str, Any]) -> Question:
    from pramaan.data.store import jload

    return Question(
        qid=row["qid"],
        split=row["split"],
        tier=row["tier"],
        as_of=row["as_of"],
        text=row["text"],
        slots=jload(row["slots"]),
        gold=jload(row["gold"]),
        gold_pages=list(row["gold_pages"] or []),
        gold_boxes=jload(row["gold_boxes"], {}),
        template_id=row["template_id"] or "",
        human_checked=bool(row["human_checked"]),
        meta=jload(row["meta"], {}),
    )
