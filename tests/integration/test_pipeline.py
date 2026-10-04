from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from pramaan.core.config import gym_config
from pramaan.core.errors import AsOfViolation
from pramaan.core.periods import PeriodKey
from pramaan.core.splits import purpose


def test_ingest_found_everything(ingested):
    home, info, store, summary = ingested
    assert summary["with_pdf"] == len(info)
    assert summary["results_pages"] == 2 * len(info)
    assert summary["facts"] > 400
    assert set(summary["splits"]) >= {"train", "dev"}


def test_gold_evidence_covers_digital_pages(built):
    home, info, store, summary, gym = built
    from pramaan.gym.build import evidence_coverage

    cov = evidence_coverage(store)["by_page_kind"]["digital"]
    assert cov["facts"] > 0 and cov["rate"] >= 0.85


def test_gym_has_every_tier_and_valid_gold(built):
    home, info, store, summary, gym = built
    tiers = {r["tier"] for r in store.q("SELECT DISTINCT tier FROM questions")}
    assert {"T1", "T2", "T3", "T5", "T6"} <= tiers
    row = store.q("SELECT * FROM questions WHERE tier = 'T1' LIMIT 1")[0]
    assert row["gold_pages"]
    assert store.page(row["gold_pages"][0]) is not None


def _agent(built, config="rules_only"):
    for r in ("SMALL", "STRONG", "REFLECTION"):
        os.environ[f"PRAMAAN_MODEL_{r}"] = "offline/rules"
    from pramaan.agent.factory import Runtime, build_agent
    from pramaan.llm.cache import LLMStore
    from pramaan.llm.client import LLMClient
    from pramaan.retrieval.index import RetrievalIndex

    home, info, store, summary, gym = built
    idx = RetrievalIndex.build(store)
    rt = Runtime(store, idx, LLMClient(store=LLMStore(home / "cache" / "llm.sqlite"), enforce_budget=False))
    return rt, build_agent(rt, config, stages="bm25+dense+rows")


def test_offline_agent_accuracy_floor_on_dev(built):
    from pramaan.evals.harness import evaluate

    rt, agent = _agent(built)
    out = evaluate(agent, "rules_only", "dev", rt.store, exp="itest", workers=1, llm_mode="offline", name="itest_dev")
    s = out["summary"]
    assert s["n"] >= 10
    assert s["accuracy"] >= 0.8
    assert s["t6_accuracy"] in (None, 1.0) or s["t6_accuracy"] >= 0.8
    assert s["errors"] == 0


def test_asof_guard_blocks_future_documents(built):
    rt, agent = _agent(built)
    f = rt.store.filings("role = 'target' AND pdf_path IS NOT NULL")[0]
    page = rt.store.pages_of(f["filing_id"], "page_id")[0]["page_id"]
    before = f["pdf_published_at"] - timedelta(hours=1)
    with pytest.raises(AsOfViolation):
        rt.corpus.page(page, before)
    assert rt.corpus.visible_filings(f["company_id"], before) == []
    res = rt.retriever.search("revenue from operations", company_id=f["company_id"], as_of=before, stages="bm25+dense+rows")
    assert res.hits == []


def test_replay_is_deterministic_with_warm_cache(built):
    rt, agent = _agent(built)
    with purpose("dev_eval"):
        from pramaan.evals.harness import load_questions

        q = load_questions(rt.store, "dev", 3)
    a = [agent.run(x.text, x.as_of, qid=x.qid) for x in q]
    b = [agent.run(x.text, x.as_of, qid=x.qid) for x in q]
    assert [e.answer.model_dump() for e in a] == [e.answer.model_dump() for e in b]


def test_future_company_returns_unavailable(built):
    rt, agent = _agent(built)
    f = rt.store.filings("role = 'target' AND pdf_path IS NOT NULL")[0]
    cname = rt.corpus.company_name(f["company_id"])
    ep = agent.run(f"What was {cname}'s consolidated revenue from operations for the quarter ended 30 June 2026? Answer in ₹ crore.", f["pdf_published_at"] - timedelta(days=2))
    assert ep.answer is not None and ep.answer.answer_type == "unavailable"


def test_verifier_catches_unit_error(built):
    from pramaan.core.schemas import Answer, AnswerInput, Citation, Intent, ValueResult
    from pramaan.verify.checks import VerifyContext, run_checks

    rt, agent = _agent(built)
    f = rt.store.filings("role = 'target' AND pdf_path IS NOT NULL")[0]
    pid = next(p for p in rt.store.pages_of(f["filing_id"]) if p["is_results_page"])
    from pramaan.data.store import jload

    row = next(r for r in jload(pid["tbl"])["rows"] if r["item"] == "revenue_from_operations")
    quote = row["values"][0]
    cit = Citation(id="c1", filing_id=f["filing_id"], page_id=pid["page_id"], quote=quote, line_item="revenue_from_operations", period="2026-06-30/3m", scope=pid["scope"])
    vr = ValueResult(found=True, company_id=f["company_id"], filing_id=f["filing_id"], scope=pid["scope"], item="revenue_from_operations", period="2026-06-30/3m", value=1.0, unit="INR_crore", page_id=pid["page_id"], quote=quote, page_unit=pid["unit"])
    wrong = Answer(answer_type="number", value=1.0, unit="INR_crore", inputs=[AnswerInput(name="rev", value=1.0, citation_id="c1")], citations=[cit])
    ctx = VerifyContext(Intent(), wrong, {"c1": vr}, {"c1": cit}, {}, rt.corpus, f["pdf_published_at"] + timedelta(days=1))
    rep = run_checks(ctx)
    assert rep.status == "hard_fail" and any(c.id == "V1" and c.status == "hard_fail" for c in rep.checks)


def test_gold_period_is_target(built):
    cfg = gym_config()
    assert PeriodKey(cfg.profile.target_period_end, 3).months == 3
    assert datetime(2026, 1, 1) < datetime.utcnow()


def test_spent_budget_stops_the_run_instead_of_writing_abstentions(built, tmp_path):
    # A spent budget used to be swallowed by the plan/compose fallbacks and graded as abstentions.
    from pramaan.agent.factory import Runtime, build_agent
    from pramaan.core.errors import PramaanError
    from pramaan.evals.harness import evaluate
    from pramaan.llm.cache import LLMStore
    from pramaan.llm.client import LLMClient

    rt, _ = _agent(built)
    client = LLMClient(store=LLMStore(tmp_path / "cold.sqlite"), enforce_budget=True)
    client.guard.total_cap = 0.0
    rt2 = Runtime(rt.store, rt.index, client)
    agent = build_agent(rt2, "rules_only", stages="bm25+dense+rows")
    with pytest.raises(PramaanError, match="agent error"):
        evaluate(agent, "rules_only", "dev", rt2.store, exp="itest", subset=3, workers=1, llm_mode="offline", name="itest_budget")
