from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Scope = Literal["standalone", "consolidated"]
AnswerUnit = Literal["INR_crore", "percent", "INR_per_share"]
AnswerType = Literal["number", "ranking", "unavailable", "premise_false"]
Operation = Literal[
    "lookup",
    "growth_yoy",
    "ratio",
    "cross_period",
    "rank",
    "unavailable_check",
    "premise_check",
]
Split = Literal["train", "dev", "test", "redteam"]
Tier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
TextSource = Literal["text_layer", "tesseract", "doctr", "vision_api"]


class Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Intent(Loose):
    companies: list[str] = Field(default_factory=list)
    scope: Scope | None = None
    items: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    operation: Operation = "lookup"
    ratio_name: str | None = None
    metric: str | None = None
    premise_claim: str | None = None


class PlanStep(Loose):
    id: str
    tool: Literal["resolve_company", "check_filed", "get_value", "calculate", "rank"]
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(Loose):
    steps: list[PlanStep] = Field(default_factory=list)


class TableColumn(Loose):
    header: str = ""
    period_end: str | None = None
    months: int | None = None
    scope: Scope | None = None


class TableRow(Loose):
    label: str
    values: list[str | None] = Field(default_factory=list)
    box: list[float] | None = None


class TableRead(Loose):
    unit_header: str | None = None
    unit: str | None = None
    scope: Scope | None = None
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)


class Citation(Loose):
    id: str
    filing_id: str = ""
    page_id: str = ""
    quote: str = ""
    line_item: str | None = None
    period: str | None = None
    scope: Scope | None = None
    bbox: list[float] | None = None


class AnswerInput(Loose):
    name: str
    value: float
    citation_id: str | None = None


class Answer(Loose):
    answer_type: AnswerType = "number"
    value: float | None = None
    unit: AnswerUnit | None = None
    ranking: list[str] = Field(default_factory=list)
    inputs: list[AnswerInput] = Field(default_factory=list)
    computation: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    reason: str | None = None


class ValueResult(Loose):
    found: bool
    company_id: str = ""
    filing_id: str = ""
    scope: Scope | None = None
    item: str = ""
    period: str = ""
    value: float | None = None
    unit: AnswerUnit | None = None
    page_id: str = ""
    quote: str = ""
    bbox: list[float] | None = None
    column_header: str = ""
    row_label: str = ""
    page_unit: str | None = None
    reason: str = ""
    retrieval_rank: int | None = None
    retrieval_margin: float = 0.0
    pages_read: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class CheckResult(Loose):
    id: str
    status: Literal["pass", "soft_fail", "hard_fail", "skip"]
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class VerifyReport(Loose):
    status: Literal["pass", "soft_fail", "hard_fail"]
    checks: list[CheckResult] = Field(default_factory=list)

    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status in ("soft_fail", "hard_fail")]

    def summary(self) -> str:
        bad = self.failed()
        if not bad:
            return "all checks passed"
        return "; ".join(f"{c.id}: {c.message}" for c in bad)


class TraceEvent(Loose):
    type: str
    node: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    t_ms: int = 0
    cost_usd: float = 0.0


class GoldFact(Loose):
    company_id: str
    scope: Scope
    item: str
    period: str
    value: float
    page_id: str | None = None


class Gold(Loose):
    answer_type: AnswerType
    value: float | None = None
    unit: AnswerUnit | None = None
    ranking: list[str] = Field(default_factory=list)
    display_unit_crore: float = 0.01
    inputs: dict[str, float] = Field(default_factory=dict)
    facts: list[GoldFact] = Field(default_factory=list)
    note: str = ""


class Question(Loose):
    qid: str
    split: Split
    tier: Tier
    as_of: datetime
    text: str
    slots: Intent
    gold: Gold
    gold_pages: list[str] = Field(default_factory=list)
    gold_boxes: dict[str, list[list[float]]] = Field(default_factory=dict)
    template_id: str = ""
    human_checked: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Episode(Loose):
    run_id: str = ""
    qid: str
    config: str = ""
    answer: Answer | None = None
    verify: VerifyReport | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    intent: Intent | None = None
    plan: Plan | None = None
    value_results: list[ValueResult] = Field(default_factory=list)
    search_log: list[dict[str, Any]] = Field(default_factory=list)
    tier_used: str = ""
    retries: int = 0
    escalated: bool = False
    abstained: bool = False
    cost_usd: float = 0.0
    latency_ms: int = 0
    node_cost: dict[str, float] = Field(default_factory=dict)
    node_latency_ms: dict[str, int] = Field(default_factory=dict)
    lessons_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None
    draft_by_tier: dict[str, Answer] = Field(default_factory=dict)
    policy: dict[str, list[str]] = Field(default_factory=dict)
    table_text: str = ""


class Grade(Loose):
    correct: bool
    citations_valid: bool
    verified_correct: bool
    abstained: bool = False
    kendall_tau: float | None = None
    error_kind: str = ""
    detail: str = ""


class Diagnostics(Loose):
    correct: bool
    citations_valid: bool
    stage: str = ""
    sub_stage: str = ""
    messages: list[str] = Field(default_factory=list)
    per_module: dict[str, list[str]] = Field(default_factory=dict)

    def summary(self) -> str:
        if self.correct and self.citations_valid:
            return "Correct and the citations are valid."
        lines = list(self.messages) or ["Incorrect answer."]
        return " ".join(lines)

    def for_module(self, name: str) -> str:
        msgs = self.per_module.get(name)
        if msgs:
            return " ".join(msgs)
        if self.correct and self.citations_valid:
            return "This module's output contributed to a correct, well-cited answer."
        return "No fault was attributed to this module. " + self.summary()


def utc_now_naive() -> datetime:
    from pramaan.core.timeutil import utcnow

    return utcnow()
