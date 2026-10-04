from __future__ import annotations

from datetime import datetime

from pramaan.core.schemas import (
    Answer,
    Episode,
    Gold,
    GoldFact,
    Grade,
    Intent,
    Question,
    ValueResult,
)
from pramaan.improve.diagnose import diagnose


def test_diagnoses_a_hundredfold_unit_error(built):
    store = built[2]
    row = store.q("SELECT * FROM questions WHERE tier = 'T1' LIMIT 1")[0]
    from pramaan.gym.grade import question_from_row

    q = question_from_row(row)
    gf = q.gold.facts[0]
    ep = Episode(qid=q.qid, answer=Answer(answer_type="number", value=gf.value * 100, unit="INR_crore"))
    ep.value_results = [
        ValueResult(found=True, company_id=gf.company_id, scope=gf.scope, item=gf.item, period=gf.period, value=gf.value * 100, page_id=gf.page_id or "", quote="1", page_unit="lakh", pages_read=[gf.page_id or ""])
    ]
    ep.search_log = [{"query": "x", "pages": [gf.page_id]}]
    g = Grade(correct=False, citations_valid=True, verified_correct=False, error_kind="wrong_value")
    d = diagnose(q, ep, g, store)
    assert d.stage == "A4" and d.sub_stage == "unit"
    assert "100x too large" in d.summary()
    assert d.per_module.get("ReadTable")


def test_diagnoses_missing_retrieval(built):
    store = built[2]
    from pramaan.gym.grade import question_from_row

    q = question_from_row(store.q("SELECT * FROM questions WHERE tier = 'T1' LIMIT 1")[0])
    ep = Episode(qid=q.qid, answer=Answer(answer_type="unavailable"), abstained=True)
    ep.search_log = [{"query": "something", "pages": ["other#p1"]}]
    g = Grade(correct=False, citations_valid=False, verified_correct=False, abstained=True, error_kind="false_abstain")
    d = diagnose(q, ep, g, store)
    assert d.stage == "A2" and "never retrieved" in d.summary()


def test_correct_answer_has_no_feedback(built):
    store = built[2]
    q = Question(qid="q", split="dev", tier="T1", as_of=datetime(2026, 8, 1), text="t", slots=Intent(), gold=Gold(answer_type="number", value=1.0, unit="INR_crore", facts=[GoldFact(company_id="c", scope="consolidated", item="other_income", period="2026-06-30/3m", value=1.0)]))
    d = diagnose(q, Episode(qid="q"), Grade(correct=True, citations_valid=True, verified_correct=True), store)
    assert d.correct and not d.messages
