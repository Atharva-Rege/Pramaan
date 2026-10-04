from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import dspy
from pydantic import ValidationError

from pramaan.agent.signatures import ComposeAnswer, MakePlan, ParseIntent, ReadTable, Reflect
from pramaan.core.ids import stable_hash
from pramaan.core.schemas import Answer, Intent, Plan, TableRead

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


class ModuleParseError(ValueError):
    pass


def extract_json(text: str) -> Any:
    if text is None:
        raise ModuleParseError("empty output")
    t = _FENCE.sub("", text.strip()).strip()
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ModuleParseError(f"no JSON object in output: {t[:120]!r}")
    end = max(t.rfind("}"), t.rfind("]"))
    chunk = t[start : end + 1]
    for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ModuleParseError(f"invalid JSON: {chunk[:160]!r}")


def _coerce_intent(d: dict[str, Any]) -> Intent:
    d = dict(d)
    for k in ("companies", "items", "periods"):
        v = d.get(k)
        if v is None:
            d[k] = []
        elif isinstance(v, str):
            d[k] = [v]
    if d.get("scope") not in ("standalone", "consolidated"):
        d["scope"] = None
    if d.get("operation") not in ("lookup", "growth_yoy", "ratio", "cross_period", "rank", "unavailable_check", "premise_check"):
        d["operation"] = "lookup"
    d["periods"] = [str(p) for p in d["periods"]]
    return Intent.model_validate(d)


def _coerce_answer(d: dict[str, Any]) -> Answer:
    d = dict(d)
    if d.get("answer_type") not in ("number", "ranking", "unavailable", "premise_false"):
        d["answer_type"] = "number" if d.get("value") is not None else "unavailable"
    if d.get("unit") not in ("INR_crore", "percent", "INR_per_share"):
        d["unit"] = None
    if isinstance(d.get("value"), str):
        try:
            d["value"] = float(d["value"].replace(",", ""))
        except ValueError:
            d["value"] = None
    d["ranking"] = [str(x) for x in (d.get("ranking") or [])]
    inputs = []
    for i in d.get("inputs") or []:
        if isinstance(i, dict) and i.get("value") is not None:
            try:
                inputs.append({"name": str(i.get("name", "x")), "value": float(i["value"]), "citation_id": i.get("citation_id")})
            except (TypeError, ValueError):
                continue
    d["inputs"] = inputs
    d["citations"] = []
    try:
        d["confidence"] = float(d.get("confidence") or 0.0)
    except (TypeError, ValueError):
        d["confidence"] = 0.0
    d["explanation"] = str(d.get("explanation") or "")
    return Answer.model_validate(d)


class AgentModules:
    def __init__(self, lm: Any, predictors: dict[str, Any] | None = None) -> None:
        self.lm = lm
        p = predictors or {}
        self.p_intent = p.get("parse_intent") or dspy.Predict(ParseIntent)
        self.p_plan = p.get("make_plan") or dspy.Predict(MakePlan)
        self.p_read = p.get("read_table") or dspy.Predict(ReadTable)
        self.p_compose = p.get("compose") or dspy.Predict(ComposeAnswer)
        self.p_reflect = p.get("reflect") or dspy.Predict(Reflect)

    def _call(self, predictor: Any, field: str, **inputs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(2):
            kw = dict(inputs)
            if attempt == 1 and "lessons" in kw:
                kw["lessons"] = (kw["lessons"] + "\nReturn only one valid JSON object, no prose.").strip()
            with dspy.context(lm=self.lm):
                pred = predictor(**kw)
            raw = getattr(pred, field, "")
            try:
                return extract_json(raw)
            except ModuleParseError as exc:
                last = exc
        raise ModuleParseError(str(last))

    def parse_intent(self, question: str, as_of: str, lessons: str = "") -> Intent:
        d = self._call(self.p_intent, "intent", question=question, as_of=as_of, lessons=lessons)
        try:
            return _coerce_intent(d)
        except ValidationError as exc:
            raise ModuleParseError(str(exc)) from exc

    def make_plan(self, intent: Intent, feedback: str = "", lessons: str = "") -> Plan:
        d = self._call(self.p_plan, "plan", intent=intent.model_dump_json(), feedback=feedback, lessons=lessons)
        return _plan_from(d)

    def read_table(self, page_text: str, lessons: str = "") -> TableRead:
        d = self._call(self.p_read, "table", page_text=page_text, lessons=lessons)
        if not isinstance(d, dict):
            raise ModuleParseError("table output is not an object")
        try:
            return TableRead.model_validate(d)
        except ValidationError:
            d = {k: v for k, v in d.items() if k in TableRead.model_fields}
            d["rows"] = [r for r in d.get("rows", []) if isinstance(r, dict) and "label" in r]
            return TableRead.model_validate(d)

    def compose(self, question: str, intent: Intent, results: dict[str, Any], lessons: str = "") -> Answer:
        d = self._call(
            self.p_compose,
            "answer",
            question=question,
            intent=intent.model_dump_json(),
            results=json.dumps(results, ensure_ascii=False),
            lessons=lessons,
        )
        try:
            return _coerce_answer(d)
        except ValidationError as exc:
            raise ModuleParseError(str(exc)) from exc

    def reflect(self, question: str, intent: Intent, plan: Plan, failures: str, lessons: str = "") -> Plan:
        d = self._call(
            self.p_reflect,
            "plan",
            question=question,
            intent=intent.model_dump_json(),
            previous_plan=plan.model_dump_json(),
            failures=failures,
            lessons=lessons,
        )
        return _plan_from(d)

    def predictors(self) -> dict[str, Any]:
        return {
            "parse_intent": self.p_intent,
            "make_plan": self.p_plan,
            "read_table": self.p_read,
            "compose": self.p_compose,
            "reflect": self.p_reflect,
        }

    def fingerprint(self) -> str:
        return stable_hash([getattr(self.lm, "model", ""), self.instructions()], 10)

    def instructions(self) -> dict[str, str]:
        return {k: str(p.signature.instructions) for k, p in self.predictors().items()}

    def load_instructions(self, mapping: dict[str, str]) -> None:
        for k, p in self.predictors().items():
            if k in mapping:
                p.signature = p.signature.with_instructions(mapping[k])


def _plan_from(d: Any) -> Plan:
    if isinstance(d, list):
        d = {"steps": d}
    steps = []
    for i, s in enumerate(d.get("steps", [])):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if tool not in ("resolve_company", "check_filed", "get_value", "calculate", "rank"):
            continue
        steps.append({"id": str(s.get("id") or f"s{i + 1}"), "tool": tool, "args": s.get("args") or {}})
    return Plan.model_validate({"steps": steps})


def save_instructions(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=1), encoding="utf-8")


def load_instruction_file(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
