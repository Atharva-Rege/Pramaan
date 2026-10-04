from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pramaan.agent.corpus import Corpus
from pramaan.agent.graph import Agent
from pramaan.agent.modules import AgentModules, load_instruction_file
from pramaan.agent.offline import OfflineBackend
from pramaan.core.config import agent_config, budget_config
from pramaan.data.store import Store
from pramaan.llm.client import LLMClient, get_client
from pramaan.llm.dspy_lm import PramaanLM, make_lm
from pramaan.retrieval.cascade import Retriever
from pramaan.retrieval.index import RetrievalIndex
from pramaan.retrieval.rerank import Reranker


class Runtime:
    def __init__(self, store: Store, index: RetrievalIndex, client: LLMClient, corpus: Corpus | None = None) -> None:
        self.store = store
        self.index = index
        self.client = client
        self.corpus = corpus or Corpus(store)
        self.retriever = Retriever(index, Reranker())
        self.offline = OfflineBackend(self.corpus)
        self.client.set_backend("offline", self.offline)


def tier_lm(rt: Runtime, tier: str, tag: str) -> PramaanLM:
    if tier == "offline":
        return PramaanLM(model="offline/rules", client=rt.client, tag=tag)
    return make_lm(tier, rt.client, tag)


def build_agent(
    rt: Runtime,
    config: str,
    instructions: dict[str, str] | Path | None = None,
    memory: Any = None,
    calibrator: Any = None,
    stages: str | None = None,
    production: bool = False,
    xbrl_lookup: Any = None,
    defense: Any = None,
) -> Agent:
    acfg = agent_config()
    variant = acfg.variant(config)
    mapping: dict[str, str] | None = None
    if isinstance(instructions, Path):
        mapping = load_instruction_file(instructions)
    elif instructions:
        mapping = instructions
    modules: dict[str, AgentModules] = {}
    for tier in dict.fromkeys(variant.tiers):
        m = AgentModules(tier_lm(rt, tier, f"agent.{tier}"))
        if mapping:
            m.load_instructions(mapping)
        modules[tier] = m
    return Agent(
        corpus=rt.corpus,
        retriever=rt.retriever,
        modules=modules,
        variant=variant,
        name=config,
        reader=acfg.reader,
        stages=stages or os.environ.get("PRAMAAN_STAGES", "rrf+ce"),
        budget=budget_config().per_question,
        memory=memory,
        calibrator=calibrator,
        production=production,
        xbrl_lookup=xbrl_lookup,
        pages_per_get=acfg.reader_pages_per_get,
        search_k=acfg.search_k,
        reflect_rounds=int(acfg.cascade.get("reflect_rounds_small", 1)),
        defense=defense,
    )


def load_runtime(index_name: str = "main", build_if_missing: bool = False, client: LLMClient | None = None, read_only: bool = False) -> Runtime:
    store = Store(read_only=read_only)
    try:
        index = RetrievalIndex.load(index_name)
    except Exception:
        if not build_if_missing:
            raise
        index = RetrievalIndex.build(store)
        index.save(index_name)
    return Runtime(store, index, client or get_client())


def episode_summary(ep: Any) -> str:
    return json.dumps({"qid": ep.qid, "answer": ep.answer.model_dump() if ep.answer else None, "cost": ep.cost_usd})
