from __future__ import annotations

import pytest

from pramaan.core.schemas import Answer, AnswerInput, Citation, Gold, Intent, Question
from pramaan.data.store import jload
from pramaan.gym.grade import grade, money_correct, percent_correct


def q_with(gold: Gold, **kw) -> Question:
    from datetime import datetime

    return Question(qid="q", split="dev", tier="T1", as_of=datetime(2026, 8, 1), text="t", slots=Intent(), gold=gold, **kw)


def test_money_and_percent_tolerances():
    assert money_correct(309.77, 309.77, 0.01)
    assert money_correct(309.78, 309.77, 0.01)
    assert not money_correct(309.80, 309.77, 0.01)
    assert money_correct(1000.4, 1000.0, 1.0)
    assert percent_correct(12.34, 12.3)
    assert not percent_correct(12.5, 12.3)


def test_unavailable_and_premise_labels(built):
    store = built[2]
    q = q_with(Gold(answer_type="unavailable"))
    assert grade(store, q, Answer(answer_type="unavailable")).correct
    g = grade(store, q, Answer(answer_type="number", value=1.0, unit="INR_crore"))
    assert not g.correct and g.error_kind == "should_abstain"
    q2 = q_with(Gold(answer_type="premise_false"))
    assert grade(store, q2, Answer(answer_type="premise_false")).correct
    assert not grade(store, q2, Answer(answer_type="number", value=3.0)).correct


def test_ranking_order_and_tau(built):
    store = built[2]
    q = q_with(Gold(answer_type="ranking", ranking=["A", "B", "C"]))
    res = lambda s: s  # noqa: E731
    ok = grade(store, q, Answer(answer_type="ranking", ranking=["A", "B", "C"]), res)
    assert ok.correct and ok.kendall_tau == pytest.approx(1.0)
    swapped = grade(store, q, Answer(answer_type="ranking", ranking=["B", "A", "C"]), res)
    assert not swapped.correct and swapped.kendall_tau < 1.0


def _real_cell(store):
    f = store.filings("role = 'target' AND pdf_path IS NOT NULL")[0]
    page = next(p for p in store.pages_of(f["filing_id"]) if p["is_results_page"])
    row = next(r for r in jload(page["tbl"])["rows"] if r["item"] == "revenue_from_operations")
    return page, row["values"][0]


def test_citation_validity_requires_quote_on_page_and_matching_value(built):
    store = built[2]
    page, quote = _real_cell(store)
    unit_factor = {"crore": 1.0, "lakh": 0.01}[page["unit"]]
    value = round(float(quote.replace(",", "")) * unit_factor, 4)
    gold = Gold(answer_type="number", value=round(value, 2), unit="INR_crore", display_unit_crore=0.01)
    q = q_with(gold)
    good = Answer(
        answer_type="number",
        value=round(value, 2),
        unit="INR_crore",
        inputs=[AnswerInput(name="rev", value=value, citation_id="c1")],
        citations=[Citation(id="c1", page_id=page["page_id"], quote=quote)],
    )
    g = grade(store, q, good)
    assert g.correct and g.citations_valid and g.verified_correct
    wrong_quote = good.model_copy(update={"citations": [Citation(id="c1", page_id=page["page_id"], quote="999,999.99")]})
    assert not grade(store, q, wrong_quote).citations_valid
    wrong_page = good.model_copy(update={"citations": [Citation(id="c1", page_id=page["page_id"].rsplit("#p", 1)[0] + "#p1", quote=quote)]})
    assert not grade(store, q, wrong_page).citations_valid
    scaled = good.model_copy(update={"inputs": [AnswerInput(name="rev", value=value * 100, citation_id="c1")]})
    assert not grade(store, q, scaled).citations_valid
    uncited = good.model_copy(update={"citations": [], "inputs": [AnswerInput(name="rev", value=value, citation_id=None)]})
    g2 = grade(store, q, uncited)
    assert g2.correct and not g2.verified_correct
