from __future__ import annotations

import json
import re
from typing import Any

from pramaan.agent.corpus import Corpus
from pramaan.agent.planning import plan_from_intent
from pramaan.core.schemas import Intent, Plan
from pramaan.data.store import jload
from pramaan.gym.slotparse import parse_question
from pramaan.llm.types import LLMRequest, RawResponse

_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond with)", re.S)
_OUT = re.compile(r"Your output fields are:\n1\. `(\w+)`")


def _fields(user: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _FIELD.finditer(user + "\n\nRespond with")}


class OfflineBackend:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._by_text: dict[str, str] | None = None

    def _page_index(self) -> dict[str, str]:
        if self._by_text is None:
            rows = self.corpus.store.q("SELECT page_id, text FROM pages WHERE tbl IS NOT NULL AND tbl <> ''")
            self._by_text = {(r["text"] or "").strip(): r["page_id"] for r in rows}
        return self._by_text

    def complete(self, req: LLMRequest) -> RawResponse:
        system = req.system or ""
        user = req.messages[-1]["content"] if req.messages else ""
        m = _OUT.search(system)
        field = m.group(1) if m else "answer"
        inputs = _fields(user)
        if field == "intent":
            body = self._intent(inputs)
        elif field == "plan":
            body = self._plan(inputs)
        elif field == "table":
            body = self._table(inputs)
        elif field == "answer":
            body = self._answer(inputs)
        else:
            body = "{}"
        text = f"[[ ## {field} ## ]]\n{body}\n\n[[ ## completed ## ]]"
        return RawResponse(text=text, model="offline/rules", input_tokens=max(1, len(user) // 4), output_tokens=max(1, len(body) // 4), stop_reason="end_turn")

    def _intent(self, inp: dict[str, str]) -> str:
        it = parse_question(inp.get("question", ""), self.corpus.cidx)
        d = it.model_dump()
        d["companies"] = [self.corpus.company_name(c) for c in it.companies]
        return json.dumps(d)

    def _plan(self, inp: dict[str, str]) -> str:
        if "previous_plan" in inp:
            return inp["previous_plan"]
        intent = Intent.model_validate(json.loads(inp["intent"]))
        return plan_from_intent(intent).model_dump_json()

    def _table(self, inp: dict[str, str]) -> str:
        pid = self._page_index().get(inp.get("page_text", "").strip())
        empty = {"unit_header": None, "unit": None, "scope": None, "columns": [], "rows": []}
        if pid is None:
            return json.dumps(empty)
        page = self.corpus.store.page(pid)
        tbl = jload(page["tbl"], None) if page else None
        if not tbl:
            return json.dumps(empty)
        cols = []
        for c in tbl["columns"]:
            p = c.get("period")
            cols.append({"header": "", "period_end": p.split("/")[0] if p else None, "months": int(p.split("/")[1][:-1]) if p else None, "scope": c.get("scope")})
        rows = [{"label": r["label"], "values": r["values"]} for r in tbl["rows"]]
        return json.dumps({"unit_header": tbl.get("unit_text"), "unit": tbl.get("unit"), "scope": page["scope"] if page["scope"] in ("standalone", "consolidated") else None, "columns": cols, "rows": rows})

    def _answer(self, inp: dict[str, str]) -> str:
        intent = Intent.model_validate(json.loads(inp["intent"]))
        results = json.loads(inp["results"])
        return json.dumps(compose_from_results(intent, results))


def compose_from_results(intent: Intent, results: dict[str, Any]) -> dict[str, Any]:
    steps: dict[str, Any] = results.get("steps", {})
    gets = {k: v for k, v in steps.items() if v.get("tool") == "get_value"}
    missing = [v for v in gets.values() if not v.get("found")]
    if missing:
        reasons = {v.get("reason", "") for v in missing}
        why = "filing not published as of the question time" if reasons <= {"no_filing_visible", "no_filing_covers_period"} else "the value could not be located in the filings"
        return {"answer_type": "unavailable", "value": None, "unit": None, "ranking": [], "inputs": [], "computation": None, "confidence": 0.9 if reasons <= {"no_filing_visible"} else 0.4, "explanation": f"Unavailable: {why}.", "reason": why}
    calc_steps = [v for v in steps.values() if v.get("tool") == "calculate"]
    rank_steps = [v for v in steps.values() if v.get("tool") == "rank"]
    if intent.operation == "rank" and rank_steps:
        r = rank_steps[0]
        inputs = [{"name": e["company"], "value": e["value"], "citation_id": e.get("citation_id")} for e in r.get("entries", []) if e.get("value") is not None]
        return {"answer_type": "ranking", "value": None, "unit": None, "ranking": r["ordered"], "inputs": inputs, "computation": None, "confidence": 0.8, "explanation": "Ranked by the retrieved values.", "reason": None}
    if intent.operation == "premise_check" and calc_steps:
        g = calc_steps[0]["value"]
        claim = intent.premise_claim
        if (claim == "decline" and g > 0) or (claim == "increase" and g < 0):
            return {"answer_type": "premise_false", "value": None, "unit": None, "ranking": [], "inputs": [], "computation": None, "confidence": 0.8, "explanation": f"The premise is false: the year on year change was {g:.1f} percent.", "reason": "premise contradicted by the filing"}
    if calc_steps:
        c = calc_steps[0]
        inputs = []
        for name, sid in c["variables"].items():
            inputs.append({"name": name, "value": steps[sid]["value"], "citation_id": steps[sid].get("citation_id")})
        return {"answer_type": "number", "value": round(c["value"], 1), "unit": "percent", "ranking": [], "inputs": inputs, "computation": c["expression"], "confidence": 0.8, "explanation": "Computed from the cited figures.", "reason": None}
    if gets:
        v = next(iter(gets.values()))
        return {"answer_type": "number", "value": round(v["value"], 2), "unit": v.get("unit"), "ranking": [], "inputs": [{"name": v["item"], "value": v["value"], "citation_id": v.get("citation_id")}], "computation": None, "confidence": 0.8, "explanation": "Read from the cited table.", "reason": None}
    return {"answer_type": "unavailable", "value": None, "unit": None, "ranking": [], "inputs": [], "computation": None, "confidence": 0.1, "explanation": "No plan steps produced a value.", "reason": "empty plan"}


__all__ = ["OfflineBackend", "Plan", "compose_from_results"]
