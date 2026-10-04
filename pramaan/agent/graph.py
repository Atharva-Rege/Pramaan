from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pramaan.agent import calc
from pramaan.agent.corpus import Corpus
from pramaan.agent.defenses import Defense, PolicyLog
from pramaan.agent.injection import Injection
from pramaan.agent.modules import AgentModules, ModuleParseError
from pramaan.agent.normalize import NormTable
from pramaan.agent.planning import plan_from_intent
from pramaan.agent.tools import ToolBox, ToolBudgetExceeded
from pramaan.core.config import AgentVariant, PerQuestionBudget, budget_config
from pramaan.core.errors import AsOfViolation, BudgetExceeded, ToolError
from pramaan.core.schemas import (
    Answer,
    Citation,
    Episode,
    Intent,
    Plan,
    PlanStep,
    TraceEvent,
    ValueResult,
    VerifyReport,
)
from pramaan.llm.client import call_scope
from pramaan.obs import new_trace_id, span
from pramaan.retrieval.cascade import Retriever
from pramaan.verify.checks import VerifyContext, run_checks

Emit = Callable[[str, dict[str, Any]], None]


@dataclass
class ExecResult:
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    values: dict[str, ValueResult] = field(default_factory=dict)
    citations: dict[str, Citation] = field(default_factory=dict)
    value_by_step: dict[str, ValueResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    pages_read: list[str] = field(default_factory=list)
    tables: dict[str, NormTable] = field(default_factory=dict)

    def results_json(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "citations": {
                k: {"page_id": c.page_id, "quote": c.quote, "line_item": c.line_item, "period": c.period, "scope": c.scope}
                for k, c in self.citations.items()
            },
            "errors": self.errors,
        }


class Agent:
    def __init__(
        self,
        corpus: Corpus,
        retriever: Retriever,
        modules: dict[str, AgentModules],
        variant: AgentVariant,
        name: str,
        reader: str = "llm",
        stages: str = "rrf+ce",
        budget: PerQuestionBudget | None = None,
        memory: Any = None,
        calibrator: Any = None,
        production: bool = False,
        xbrl_lookup: Any = None,
        pages_per_get: int = 3,
        search_k: int = 8,
        reflect_rounds: int = 1,
        parallel: bool = True,
        defense: Defense | None = None,
    ) -> None:
        self.defense = defense or Defense()
        self.corpus = corpus
        self.retriever = retriever
        self.modules = modules
        self.variant = variant
        self.name = name
        self.reader = variant.reader or reader
        self.stages = stages
        self.budget = budget or budget_config().per_question
        self.memory = memory
        self.calibrator = calibrator
        self.production = production
        self.xbrl_lookup = xbrl_lookup
        self.pages_per_get = pages_per_get
        self.search_k = search_k
        self.reflect_rounds = reflect_rounds
        self.parallel = parallel

    def run(
        self,
        question: str,
        as_of: datetime,
        qid: str = "",
        emit: Emit | None = None,
        run_id: str = "",
        lessons_override: str | None = None,
        injection: Injection | None = None,
    ) -> Episode:
        ep = Episode(run_id=run_id, qid=qid, config=self.name)
        self._injection = injection
        t_start = time.perf_counter()
        trace_id = new_trace_id()

        def event(kind: str, node: str = "", cost: float = 0.0, **data: Any) -> None:
            data = {**data, "trace_id": trace_id}
            ep.trace.append(TraceEvent(type=kind, node=node, data=data, t_ms=int((time.perf_counter() - t_start) * 1000), cost_usd=cost))
            if emit:
                emit(kind, data)

        with call_scope(run_id) as total:
            try:
                self._run(ep, question, as_of, event, lessons_override)
            except Exception as exc:
                ep.error = f"{type(exc).__name__}: {exc}"
                if ep.answer is None:
                    ep.answer = Answer(answer_type="unavailable", reason="agent error", explanation=f"The agent failed: {exc}")
                    ep.abstained = True
                event("error", message=ep.error)
        ep.cost_usd = round(total.cost_usd, 6)
        # Wall clock plus the recorded original latency of calls served from the LLM disk cache:
        # the cache is an evaluation shortcut, so latency is reported as if every call were live
        # (an upper bound when independent steps ran in parallel).
        cached_ms = sum(r.latency_ms for r in total.results if r.cache_hit)
        ep.latency_ms = int((time.perf_counter() - t_start) * 1000) + int(cached_ms)
        if ep.answer is not None:
            event("answer", answer=ep.answer.model_dump(), confidence=ep.confidence, abstained=ep.abstained)
        return ep

    def _node(self, ep: Episode, name: str, fn: Callable[[], Any], event: Callable[..., None]) -> Any:
        t0 = time.perf_counter()
        with call_scope() as sc, span(f"node.{name}", qid=ep.qid, config=self.name):
            try:
                return fn()
            finally:
                ep.node_cost[name] = round(ep.node_cost.get(name, 0.0) + sc.cost_usd, 6)
                ep.node_latency_ms[name] = ep.node_latency_ms.get(name, 0) + int((time.perf_counter() - t0) * 1000)

    def _run(self, ep: Episode, question: str, as_of: datetime, event: Callable[..., None], lessons_override: str | None) -> None:
        tiers = self.variant.tiers
        lessons_text = lessons_override if lessons_override is not None else self._lessons(ep, question, event)
        m0 = self.modules[tiers[0]]
        inj = getattr(self, "_injection", None)
        try:
            if inj is not None and inj.intent is not None:
                intent = inj.intent
            else:
                intent = self._node(ep, "intake", lambda: m0.parse_intent(question, as_of.isoformat(), lessons_text), event)
        except BudgetExceeded:  # a spent budget must stop the episode as an error, not become a fallback answer
            raise
        except (ModuleParseError, Exception) as exc:
            ep.answer = Answer(answer_type="unavailable", reason="intent parsing failed", explanation=str(exc))
            ep.abstained = True
            event("intent", error=str(exc))
            return
        ep.intent = intent
        event("intent", node="intake", intent=intent.model_dump())

        feedback = ""
        exclude: set[str] = set()
        final: tuple[Answer, VerifyReport | None, str] | None = None
        last_failure = ""
        for ti, tier in enumerate(tiers):
            modules = self.modules[tier]
            if ti > 0:
                ep.escalated = True
                event("escalate", node="escalate", to=tier, reason=last_failure)
            try:
                plan = self._node(
                    ep,
                    "plan",
                    lambda m=modules, f=feedback: self._make_plan(m, intent, f, lessons_text),
                    event,
                )
            except BudgetExceeded:  # a spent budget must stop the episode as an error, not become a fallback answer
                raise
            except Exception as exc:
                plan = plan_from_intent(intent, self.variant.calculator)
                event("plan", node="plan", fallback=str(exc))
            ep.plan = plan
            event("plan", node="plan", tier=tier, plan=plan.model_dump())
            reflect_left = self.reflect_rounds if (self.variant.reflect and ti == 0) else 0
            while True:
                tools = ToolBox(
                    self.corpus,
                    self.retriever,
                    modules,
                    as_of,
                    reader=self.reader,
                    budget=self.budget,
                    stages=self.stages,
                    pages_per_get=self.pages_per_get,
                    search_k=self.search_k,
                    lessons=lessons_text,
                    trace=lambda t, d: event(t, node="execute", **d),
                    row_docs=self.variant.row_docs,
                    injection=inj,
                    defense=self.defense,
                )
                tools.policy = PolicyLog()
                tools.intent = intent
                res = self._node(ep, "execute", lambda t=tools, p=plan, ex=set(exclude): self._execute(t, p, ex, event), event)
                ep.value_results = list(res.values.values())
                ep.search_log = list(tools.search_log)
                ep.policy = {"attempted": list(tools.policy.attempted), "blocked": list(tools.policy.blocked), "executed": list(tools.policy.executed)}
                ep.table_text = " ".join(r.label + " " + " ".join(v for v in r.raw if v) for nt in res.tables.values() for r in nt.rows)[:20000]
                try:
                    answer = self._node(
                        ep,
                        "compose",
                        lambda m=modules, r=res: self._compose(m, question, intent, r, lessons_text),
                        event,
                    )
                except BudgetExceeded:  # a spent budget must stop the episode as an error, not become a fallback answer
                    raise
                except Exception as exc:
                    answer = Answer(answer_type="unavailable", reason="composition failed", explanation=str(exc))
                ep.draft_by_tier[tier] = answer
                ep.tier_used = tier
                report: VerifyReport | None = None
                if self.variant.verify:
                    report = self._node(ep, "verify", lambda r=res, a=answer, t=tools: self._verify(intent, a, r, t, as_of), event)
                    ep.verify = report
                    event("verify", node="verify", report=report.model_dump())
                # A hard fail always counts as a failure. With escalate_on_abstain, so does a tier
                # that gave up ("unavailable" with a soft fail) when a stronger tier is left.
                gave_up = (
                    self.variant.escalate_on_abstain
                    and ti < len(tiers) - 1
                    and report is not None
                    and report.status == "soft_fail"
                    and answer.answer_type == "unavailable"
                )
                if report is None or (report.status != "hard_fail" and not gave_up):
                    final = (answer, report, tier)
                    break
                last_failure = report.summary()
                if self.variant.exclude_failed_pages:
                    # Ban the pages that produced a failed answer. This also bans the right page when
                    # the failure was a verifier false reject or a read slip on it; see exclude_failed_pages.
                    for vr in res.values.values():
                        if vr.found and vr.page_id:
                            exclude.add(vr.page_id)
                if reflect_left > 0:
                    reflect_left -= 1
                    ep.retries += 1
                    failures = last_failure + " | " + "; ".join(res.errors[:3])
                    try:
                        plan = self._node(ep, "reflect", lambda m=modules, p=plan, f=failures: m.reflect(question, intent, p, f, lessons_text), event)
                    except BudgetExceeded:  # a spent budget must stop the episode as an error, not become a fallback answer
                        raise
                    except Exception as exc:
                        plan = plan_from_intent(intent, self.variant.calculator)
                        event("reflect", node="reflect", fallback=str(exc))
                    event("reflect", node="reflect", plan=plan.model_dump(), failures=failures)
                    continue
                feedback = f"Previous attempt failed verification: {last_failure}." + (
                    f" Pages already tried: {sorted(exclude)[:4]}." if exclude else ""
                )
                break
            if final is not None:
                break
        if final is None:
            ep.abstained = True
            last = ep.draft_by_tier.get(ep.tier_used)
            ep.answer = Answer(
                answer_type="unavailable",
                reason=f"verification failed: {last_failure}",
                explanation="Pramaan could not produce an answer that passed its checks, so it abstains.",
                citations=last.citations if last else [],
                confidence=0.05,
            )
            ep.confidence = 0.05
            return
        answer, report, tier = final
        ep.answer = answer
        ep.abstained = answer.answer_type == "unavailable"
        ep.confidence = self._confidence(ep, answer, report)
        ep.answer.confidence = ep.confidence
        for c in answer.citations:
            event("evidence", node="compose", citation=c.model_dump())

    def _lessons(self, ep: Episode, question: str, event: Callable[..., None]) -> str:
        if self.memory is None:
            return ""
        try:
            mems = self.memory.retrieve(question)
        except BudgetExceeded:  # a spent budget must stop the episode as an error, not become a fallback answer
            raise
        except Exception:
            return ""
        ep.lessons_used = [m["mem_id"] for m in mems]
        text = self.memory.format(mems)
        event("lessons", node="intake", lessons=[m["lesson"] for m in mems])
        return text

    def _make_plan(self, modules: AgentModules, intent: Intent, feedback: str, lessons: str) -> Plan:
        plan = modules.make_plan(intent, feedback, lessons)
        if not plan.steps:
            return plan_from_intent(intent, self.variant.calculator)
        if not self.variant.calculator:
            plan = Plan(steps=[s for s in plan.steps if s.tool != "calculate"])
        return plan

    def _execute(self, tools: ToolBox, plan: Plan, exclude: set[str], event: Callable[..., None]) -> ExecResult:
        res = ExecResult()
        counter = [0]

        def run_get(step: PlanStep) -> tuple[PlanStep, ValueResult | Exception]:
            a = step.args
            try:
                violation = self._policy_violation(tools, step)
                if violation:
                    tools.policy.attempted.append(violation)
                    if self.defense.tool_policy:
                        tools.policy.blocked.append(violation)
                        return step, ToolError(f"blocked by tool policy: {violation}")
                    tools.policy.executed.append(violation)
                tools.bump()
                event("tool_call", node="execute", tool="get_value", step=step.id, args=a)
                v = tools.get_value(
                    str(a.get("company", "")),
                    a.get("scope"),
                    str(a.get("item", "")),
                    str(a.get("period", "")),
                    exclude_pages=exclude,
                    note=str(a.get("note", "") or ""),
                )
                return step, v
            except (ToolBudgetExceeded, ToolError, AsOfViolation, KeyError, ValueError) as exc:
                return step, exc

        steps = list(plan.steps)
        i = 0
        while i < len(steps):
            batch = []
            while i < len(steps) and steps[i].tool == "get_value":
                batch.append(steps[i])
                i += 1
            if batch:
                if self.parallel and len(batch) > 1:
                    ctx = contextvars.copy_context()
                    with ThreadPoolExecutor(max_workers=min(4, len(batch))) as ex:
                        outs = list(ex.map(lambda s, c=ctx: c.copy().run(run_get, s), batch))
                else:
                    outs = [run_get(s) for s in batch]
                for step, out in outs:
                    self._record_value(res, step, out, counter, event)
                continue
            step = steps[i]
            i += 1
            try:
                tools.bump()
                event("tool_call", node="execute", tool=step.tool, step=step.id, args=step.args)
                if step.tool == "calculate":
                    self._do_calc(res, step)
                elif step.tool == "rank":
                    self._do_rank(res, step, tools)
                elif step.tool == "check_filed":
                    out = tools.check_filed(str(step.args.get("company", "")), str(step.args.get("period", "")))
                    res.steps[step.id] = {"tool": "check_filed", **out}
                elif step.tool == "resolve_company":
                    m = tools.resolve_company(str(step.args.get("name_or_ticker") or step.args.get("company") or ""))
                    res.steps[step.id] = {"tool": "resolve_company", "match": m.model_dump() if m else None}
            except (ToolBudgetExceeded, ToolError, KeyError, ValueError, ZeroDivisionError) as exc:
                res.errors.append(f"{step.id}: {exc}")
                res.steps[step.id] = {"tool": step.tool, "error": str(exc)}
        res.pages_read = list(tools.pages_read)
        res.errors.extend(tools.read_errors[:3])
        res.tables = dict(tools.tables_by_page)
        return res

    def _policy_violation(self, tools: ToolBox, step: PlanStep) -> str | None:
        intent = getattr(tools, "intent", None)
        if intent is None:
            return None
        company = str(step.args.get("company", ""))
        ok_ids = set()
        for name in intent.companies:
            for h in self.corpus.resolve_company(name, limit=1):
                ok_ids.add(h["company_id"])
        got = self.corpus.resolve_company(company, limit=1)
        if ok_ids and (not got or got[0]["company_id"] not in ok_ids):
            return f"company '{company}' is not in the question"
        item = step.args.get("item")
        allowed_items = set(intent.items) | ({intent.metric} if intent.metric else set())
        if intent.ratio_name:
            from pramaan.agent.planning import RATIO_INPUTS

            allowed_items |= set(RATIO_INPUTS.get(intent.ratio_name, ()))
        if intent.metric:
            from pramaan.agent.planning import RATIO_INPUTS

            allowed_items |= set(RATIO_INPUTS.get(intent.metric, ()))
        if allowed_items and item not in allowed_items:
            return f"item '{item}' is not in the question"
        return None

    def _record_value(self, res: ExecResult, step: PlanStep, out: ValueResult | Exception, counter: list[int], event: Callable[..., None]) -> None:
        a = step.args
        if isinstance(out, Exception):
            res.errors.append(f"{step.id}: {out}")
            res.steps[step.id] = {"tool": "get_value", "found": False, "reason": str(out), "company": a.get("company"), "item": a.get("item"), "period": a.get("period")}
            return
        entry: dict[str, Any] = {
            "tool": "get_value",
            "found": out.found,
            "company": a.get("company"),
            "scope": out.scope,
            "item": out.item,
            "period": out.period,
            "reason": out.reason,
        }
        if out.found:
            counter[0] += 1
            cid = f"c{len(res.citations) + 1}"
            cit = Citation(
                id=cid,
                filing_id=out.filing_id,
                page_id=out.page_id,
                quote=out.quote,
                line_item=out.item,
                period=out.period,
                scope=out.scope,
                bbox=out.bbox,
            )
            res.citations[cid] = cit
            res.values[cid] = out
            res.value_by_step[step.id] = out
            entry.update({"value": out.value, "unit": out.unit, "citation_id": cid, "quote": out.quote, "page_id": out.page_id, "row_label": out.row_label})
            event("tool_result", node="execute", tool="get_value", step=step.id, value=out.value, page_id=out.page_id, quote=out.quote)
        else:
            res.errors.append(f"{step.id}: {out.reason}")
            res.values[f"x{step.id}"] = out
            event("tool_result", node="execute", tool="get_value", step=step.id, found=False, reason=out.reason)
        res.steps[step.id] = entry

    def _do_calc(self, res: ExecResult, step: PlanStep) -> None:
        a = step.args
        variables: dict[str, float] = {}
        for name, ref in (a.get("variables") or {}).items():
            src = res.steps.get(str(ref))
            if src is None or src.get("value") is None:
                raise ToolError(f"variable {name} refers to unavailable step {ref}")
            variables[name] = float(src["value"])
        val = calc.calculate(str(a.get("expression", "")), variables)
        res.steps[step.id] = {"tool": "calculate", "expression": a.get("expression"), "variables": dict(a.get("variables") or {}), "inputs": variables, "value": round(val, 6)}

    def _do_rank(self, res: ExecResult, step: PlanStep, tools: ToolBox) -> None:
        entries = []
        for e in step.args.get("entries", []):
            src = res.steps.get(str(e.get("ref")))
            entries.append({"company": e.get("company"), "value": None if src is None else src.get("value"), "citation_id": None if src is None else src.get("citation_id")})
        if any(e["value"] is None for e in entries):
            raise ToolError("rank entries missing values")
        res.steps[step.id] = {"tool": "rank", "ordered": tools.rank(entries, bool(step.args.get("descending", True))), "entries": entries}

    def _compose(self, modules: AgentModules, question: str, intent: Intent, res: ExecResult, lessons: str) -> Answer:
        ans = modules.compose(question, intent, res.results_json(), lessons)
        used: list[Citation] = []
        for inp in ans.inputs:
            if inp.citation_id in res.citations:
                c = res.citations[inp.citation_id]
                if all(c.id != u.id for u in used):
                    used.append(c)
            else:
                inp.citation_id = None
        ans.citations = used
        return ans

    def _verify(self, intent: Intent, answer: Answer, res: ExecResult, tools: ToolBox, as_of: datetime) -> VerifyReport:
        ctx = VerifyContext(
            flags={"cell_rule": self.defense.cell_rule, "relax_ratio_units": self.variant.relax_ratio_units},
            intent=intent,
            answer=answer,
            values=res.values,
            citations=res.citations,
            tables=res.tables,
            corpus=self.corpus,
            as_of=as_of,
            production=self.production,
            xbrl_lookup=self.xbrl_lookup,
        )
        if answer.answer_type == "unavailable":
            return self._verify_unavailable(answer, res)
        return run_checks(ctx)

    def _verify_unavailable(self, answer: Answer, res: ExecResult) -> VerifyReport:
        from pramaan.core.schemas import CheckResult

        missing = [v for v in res.values.values() if not v.found]
        reasons = {v.reason for v in missing}
        if missing and reasons <= {"no_filing_visible", "no_filing_covers_period"}:
            return VerifyReport(status="pass", checks=[CheckResult(id="V0", status="pass", message="required filing is not visible as of the question time")])
        if missing:
            return VerifyReport(status="soft_fail", checks=[CheckResult(id="V0", status="soft_fail", message="value not found: " + ", ".join(sorted(reasons)))])
        return VerifyReport(status="soft_fail", checks=[CheckResult(id="V0", status="soft_fail", message="abstained without a retrieval failure")])

    def _confidence(self, ep: Episode, answer: Answer, report: VerifyReport | None) -> float:
        feats = confidence_features(ep, answer, report)
        if self.calibrator is not None:
            return float(self.calibrator.predict_proba(feats))
        return heuristic_confidence(feats)


def confidence_features(ep: Episode, answer: Answer, report: VerifyReport | None) -> dict[str, float]:
    status = {c.id: c.status for c in (report.checks if report else [])}
    margin = max((v.retrieval_margin for v in ep.value_results if v.found), default=0.0)
    rank = max((v.retrieval_rank or 1 for v in ep.value_results if v.found), default=1)
    return {
        "n_soft": float(sum(1 for s in status.values() if s == "soft_fail")),
        "v1_pass": float(status.get("V1") == "pass"),
        "v2_pass": float(status.get("V2") == "pass"),
        "v3_pass": float(status.get("V3") == "pass"),
        "v4_pass": float(status.get("V4") == "pass"),
        "v4_tested": float(status.get("V4") not in (None, "skip")),
        "v5_pass": float(status.get("V5") == "pass"),
        "margin": float(margin),
        "rank": float(rank),
        "strong": float(ep.tier_used == "strong"),
        "escalated": float(ep.escalated),
        "retries": float(ep.retries),
        "abstain": float(answer.answer_type == "unavailable"),
        "llm_conf": float(answer.confidence or 0.0),
    }


def heuristic_confidence(f: dict[str, float]) -> float:
    c = 0.55 + 0.1 * f["v1_pass"] + 0.1 * f["v2_pass"] + 0.05 * f["v3_pass"] + 0.1 * f["v4_pass"] + 0.05 * f["v5_pass"]
    c -= 0.1 * f["n_soft"] + 0.05 * f["escalated"] + 0.05 * f["retries"] + 0.03 * (f["rank"] - 1)
    return max(0.01, min(0.99, c))
