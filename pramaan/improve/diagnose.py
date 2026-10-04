from __future__ import annotations

import math
from typing import Any

from pramaan.agent.corpus import Corpus
from pramaan.core.config import line_items
from pramaan.core.periods import PeriodKey, describe_period
from pramaan.core.schemas import Diagnostics, Episode, GoldFact, Grade, Question, ValueResult
from pramaan.data.store import Store, jload
from pramaan.gym.grade import grade
from pramaan.gym.slotparse import slots_match

STAGES = {
    "A1": "intent",
    "A2": "retrieval",
    "A3": "rerank",
    "A4": "read",
    "A5": "plan_or_compute",
    "A6": "compose_or_citation",
    "A7": "verifier",
}


def _human(item: str) -> str:
    return line_items().human_name(item) if item in line_items().items else item


def _match_value(gf: GoldFact, results: list[ValueResult]) -> ValueResult | None:
    cands = [v for v in results if v.item == gf.item and v.period == gf.period and (not v.company_id or v.company_id == gf.company_id)]
    found = [v for v in cands if v.found]
    return (found or cands or [None])[0]


def _power_of_ten(got: float, want: float) -> int | None:
    if want == 0 or got == 0 or (got < 0) != (want < 0):
        return None
    r = abs(got / want)
    e = math.log10(r)
    k = round(e)
    if k != 0 and abs(e - k) < 0.03:
        return int(k)
    return None


def _locate_cell(store: Store, page_id: str, raw: str, label: str) -> tuple[str | None, str | None, str | None]:
    page = store.page(page_id)
    tbl = jload(page["tbl"], None) if page else None
    if not tbl:
        return None, None, None
    best: tuple[str | None, str | None, str | None] = (None, None, None)
    for row in tbl["rows"]:
        for j, v in enumerate(row["values"]):
            if v == raw:
                period = tbl["columns"][j]["period"] if j < len(tbl["columns"]) else None
                if row["label"].lower()[:20] == label.lower()[:20]:
                    return row["label"], period, row.get("item")
                best = best if best[0] else (row["label"], period, row.get("item"))
    return best


def diagnose(q: Question, ep: Episode, g: Grade, store: Store, corpus: Corpus | None = None) -> Diagnostics:
    if g.correct and g.citations_valid:
        return Diagnostics(correct=True, citations_valid=True)
    msgs: list[str] = []
    per: dict[str, list[str]] = {"ParseIntent": [], "MakePlan": [], "ReadTable": [], "ComposeAnswer": [], "Reflect": []}
    stage = ""
    sub = ""

    def add(st: str, text: str, module: str | None = None, subtype: str = "") -> None:
        nonlocal stage, sub
        msgs.append(text)
        if not stage:
            stage, sub = st, subtype
        if module:
            per[module].append(text)

    corpus = corpus or Corpus(store)
    gold = q.gold
    ans = ep.answer
    if ep.intent is not None and corpus is not None:
        got_ids = []
        for name in ep.intent.companies:
            hits = corpus.resolve_company(name, limit=1)
            got_ids.append(hits[0]["company_id"] if hits else name)
        parsed = ep.intent.model_copy(update={"companies": got_ids})
        if not slots_match(parsed, q.slots):
            diffs = []
            if sorted(parsed.items) != sorted(q.slots.items):
                diffs.append(f"items {parsed.items} but the question asks for {q.slots.items}")
            if (parsed.scope or None) != (q.slots.scope or None):
                diffs.append(f"scope {parsed.scope} but the question says {q.slots.scope}")
            if sorted(parsed.periods) != sorted(q.slots.periods):
                diffs.append(f"periods {parsed.periods} but the question asks for {q.slots.periods}")
            if parsed.operation != q.slots.operation:
                diffs.append(f"operation {parsed.operation} but this is a {q.slots.operation} question")
            if sorted(parsed.companies) != sorted(q.slots.companies):
                diffs.append(f"companies {parsed.companies} but the question names {q.slots.companies}")
            if diffs:
                add("A1", "Intent parsing error: " + "; ".join(diffs) + ".", "ParseIntent", "intent")
    searched: dict[str, int] = {}
    for entry in ep.search_log:
        for r, pid in enumerate(entry["pages"]):
            searched[pid] = min(searched.get(pid, 99), r)
    read: set[str] = set()
    for v in ep.value_results:
        read.update(v.pages_read)

    if gold.answer_type == "unavailable":
        if ans is not None and ans.answer_type != "unavailable":
            used = [c.page_id for c in ans.citations][:2]
            add("A7", "The filing for this period was not published as of the question time, so the right answer is unavailable, but the agent answered from " + (", ".join(used) or "another document") + ". Check that the cited filing's period equals the period asked.", "ComposeAnswer", "false_accept")
    elif gold.answer_type == "premise_false":
        if ans is not None and ans.answer_type != "premise_false":
            add("A5", "The question assumes a decline, but the figures show growth; compare the current and prior year values before explaining.", "MakePlan", "premise")
    else:
        for gf in gold.facts:
            vr = _match_value(gf, ep.value_results)
            gp = gf.page_id or ""
            label = f"{_human(gf.item)} ({describe_period(PeriodKey.parse(gf.period))}, {gf.scope})"
            if gp and gp not in searched and (vr is None or vr.page_id != gp):
                qtxt = next((e["query"] for e in ep.search_log), "")
                add("A2", f"Gold page {gp} for {label} was never retrieved; the search query was '{qtxt}'. Include the scope and the statement title in the query.", None, "retrieval")
                continue
            if gp and gp in searched and gp not in read and (vr is None or vr.page_id != gp):
                add("A3", f"Gold page {gp} for {label} was retrieved at rank {searched[gp] + 1} but never read; only the top pages are read.", None, "rerank")
                continue
            if vr is None or not vr.found:
                add("A4", f"No value was extracted for {label} although the page was read; the table row or column was not recognised.", "ReadTable", "not_found")
                continue
            if vr.value is None:
                continue
            tol = max(0.011, abs(gf.value) * 1e-6)
            if abs(vr.value - gf.value) <= tol or abs(vr.value - gf.value) <= 0.0051 * max(1.0, abs(gf.value) / 100):
                continue
            k = _power_of_ten(vr.value, gf.value)
            unit = vr.page_unit or "unknown"
            if k is not None:
                direction = "too large" if k > 0 else "too small"
                add("A4", f"Value is {10 ** abs(k)}x {direction} for {label}; the page header says the unit is {unit} and it was converted incorrectly.", "ReadTable", "unit")
                continue
            lab, per_col, row_item = _locate_cell(store, vr.page_id, vr.quote, vr.row_label)
            if per_col and per_col != gf.period:
                add("A4", f"Used the {per_col} column for {_human(gf.item)}; the question asks for {gf.period}.", "ReadTable", "wrong_column")
            elif row_item and row_item != gf.item:
                add("A4", f"Read the '{lab}' row as {_human(gf.item)}; that row is a different line item.", "ReadTable", "wrong_row")
            else:
                add("A4", f"Extracted {vr.value} for {label} but the filing prints {gf.value}; check for an OCR digit error.", "ReadTable", "ocr_digit")
        if not stage and ans is not None:
            if ans.answer_type == "unavailable":
                pass
            elif g.correct is False:
                expr = ans.computation or "none"
                add("A5", f"Inputs matched the filing but the final value {ans.value} is wrong; the computation used was '{expr}'. Use the formula for this question type exactly.", "MakePlan", "compute")
            elif not g.citations_valid:
                add("A6", f"The value is right but the citation is invalid ({g.detail}). Cite the page and the exact printed number for every input.", "ComposeAnswer", "citation")

    if ans is not None and ans.answer_type == "unavailable" and gold.answer_type in ("number", "ranking"):
        drafts = ep.draft_by_tier
        right = [t for t, d in drafts.items() if grade(store, q, d, None).correct and gold.answer_type == "number"]
        if right:
            report = ep.verify.summary() if ep.verify else "checks failed"
            msgs.insert(0, f"False reject: the {right[0]} draft was correct but verification rejected it ({report}).")
            stage, sub = "A7", "false_reject"
        elif not stage:
            add("A4", "The agent abstained because no value could be extracted from the retrieved pages.", "ReadTable", "not_found")
    elif ans is not None and ep.verify is not None and ep.verify.status != "hard_fail" and not g.correct and stage != "A7":
        sub = sub or ""
        msgs.append("The verifier accepted this wrong answer (false accept).")
    if not msgs:
        msgs = ["The answer is incorrect."]
        stage = stage or "A5"
    return Diagnostics(correct=g.correct, citations_valid=g.citations_valid, stage=stage, sub_stage=sub, messages=msgs, per_module=per)


def stage_label(code: str) -> str:
    return STAGES.get(code, code)


def to_jsonable(d: Diagnostics) -> dict[str, Any]:
    return d.model_dump()
