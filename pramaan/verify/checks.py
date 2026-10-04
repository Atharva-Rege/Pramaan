from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pramaan.agent import calc
from pramaan.agent.corpus import Corpus
from pramaan.agent.normalize import NormTable, parse_period, select_value, to_answer_units
from pramaan.core.config import line_items
from pramaan.core.numbers import find_numbers, parse_number
from pramaan.core.periods import PeriodKey, parse_dates
from pramaan.core.schemas import Answer, CheckResult, Citation, Intent, ValueResult, VerifyReport
from pramaan.data.tables import UNIT_FACTOR_CRORE

HARD_REL = 0.10


@dataclass
class VerifyContext:
    intent: Intent
    answer: Answer
    values: dict[str, ValueResult]
    citations: dict[str, Citation]
    tables: dict[str, NormTable]
    corpus: Corpus
    as_of: datetime
    production: bool = False
    xbrl_lookup: Any = None
    flags: dict[str, Any] = field(default_factory=dict)


def _ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _quote_value(quote: str) -> tuple[float | None, int]:
    toks = find_numbers(quote)
    if toks:
        return toks[0].value, toks[0].decimals
    v = parse_number(quote)
    return v, 2


def v1_grounding(ctx: VerifyContext) -> CheckResult:
    a = ctx.answer
    if a.answer_type in ("unavailable", "premise_false") and not a.inputs:
        return CheckResult(id="V1", status="skip", message="no numeric claims")
    problems: list[str] = []
    for inp in a.inputs:
        c = ctx.citations.get(inp.citation_id or "")
        if c is None:
            problems.append(f"input {inp.name} has no valid citation")
            continue
        page = ctx.corpus.page(c.page_id, ctx.as_of) if c.page_id else None
        if page is None:
            problems.append(f"citation {c.id} points to an unknown page")
            continue
        if _ws(c.quote) not in _ws(page["text"] or ""):
            problems.append(f"quote '{c.quote}' is not on page {c.page_id}")
            continue
        qv, dec = _quote_value(c.quote)
        vr = ctx.values.get(c.id)
        if qv is None:
            problems.append(f"quote '{c.quote}' is not numeric")
            continue
        per_share = bool(vr and vr.unit == "INR_per_share")
        if per_share:
            if abs(abs(qv) - abs(inp.value)) > 0.0051:
                problems.append(f"quote {c.quote} differs from input {inp.name}={inp.value}")
            continue
        unit = (vr.page_unit if vr else None) or page.get("unit")
        f = UNIT_FACTOR_CRORE.get(unit or "")
        if f is None:
            continue
        prec = f * 10 ** (-dec)
        tol = max(0.0051, prec / 2 + 1e-9)
        if abs(qv * f - inp.value) > tol:
            problems.append(f"quote {c.quote} under unit {unit} is {qv * f:.4f} crore but input {inp.name}={inp.value}")
    if problems:
        return CheckResult(id="V1", status="hard_fail", message="; ".join(problems[:3]), details={"problems": problems})
    return CheckResult(id="V1", status="pass")


def v2_arithmetic(ctx: VerifyContext) -> CheckResult:
    a = ctx.answer
    if a.answer_type == "ranking":
        return _v2_ranking(ctx)
    if a.answer_type == "unavailable":
        return CheckResult(id="V2", status="skip")
    if a.answer_type == "premise_false":
        return _v2_premise(ctx)
    if a.value is None:
        return CheckResult(id="V2", status="hard_fail", message="number answer without a value")
    env = {i.name: i.value for i in a.inputs}
    if a.computation:
        try:
            got = calc.calculate(a.computation, env)
        except Exception as exc:
            return CheckResult(id="V2", status="hard_fail", message=f"computation failed: {exc}")
    elif len(a.inputs) == 1:
        got = a.inputs[0].value
    elif not a.inputs:
        return CheckResult(id="V2", status="hard_fail", message="value has no inputs")
    else:
        return CheckResult(id="V2", status="hard_fail", message="several inputs but no computation")
    tol = max(0.0051 if a.unit != "percent" else 0.051, 1e-3 * abs(got))
    if abs(got - a.value) > tol:
        return CheckResult(id="V2", status="hard_fail", message=f"recomputed {got:.4f} but answer says {a.value}", details={"recomputed": got})
    return CheckResult(id="V2", status="pass", details={"recomputed": got})


def _v2_ranking(ctx: VerifyContext) -> CheckResult:
    a = ctx.answer
    vals = {i.name.lower(): i.value for i in a.inputs}
    order = [vals.get(r.lower()) for r in a.ranking]
    if any(o is None for o in order):
        return CheckResult(id="V2", status="soft_fail", message="ranking inputs missing for some companies")
    ok = all(order[i] >= order[i + 1] for i in range(len(order) - 1))  # type: ignore[operator]
    return CheckResult(id="V2", status="pass" if ok else "hard_fail", message="" if ok else "ranking order contradicts its inputs")


def _growth(values: dict[str, ValueResult]) -> float | None:
    cur = prev = None
    for v in values.values():
        if not v.found or v.value is None:
            continue
        k = PeriodKey.parse(v.period) if v.period else None
        if k is None:
            continue
        if cur is None or k.end > cur[0].end:
            prev, cur = cur, (k, v.value)
        elif prev is None or k.end > prev[0].end:
            prev = (k, v.value)
    if cur is None or prev is None or prev[1] == 0:
        return None
    return (cur[1] - prev[1]) / abs(prev[1]) * 100


def _v2_premise(ctx: VerifyContext) -> CheckResult:
    g = _growth(ctx.values)
    claim = ctx.intent.premise_claim
    if g is None or claim is None:
        return CheckResult(id="V2", status="soft_fail", message="cannot confirm the premise from retrieved values")
    contradicts = (claim == "decline" and g > 0) or (claim == "increase" and g < 0)
    if contradicts:
        return CheckResult(id="V2", status="pass", details={"growth": g})
    return CheckResult(id="V2", status="hard_fail", message=f"premise holds (growth {g:.1f}%) but answer says it is false")


def _unit_cancels(ctx: VerifyContext) -> bool:
    """A percent answer computed from values that all sit on one page does not depend on that
    page's unit header, because every input shares it. Opt-in via the variant flag
    relax_ratio_units; inputs from different pages keep the strict rule (crore against lakh)."""
    if not ctx.flags.get("relax_ratio_units") or ctx.answer.unit != "percent":
        return False
    return len({v.page_id for v in ctx.values.values() if v.found}) == 1


def v3_units(ctx: VerifyContext) -> CheckResult:
    li = line_items()
    problems: list[str] = []
    soft: list[str] = []
    for v in ctx.values.values():
        if not v.found:
            continue
        kind = li.kind(v.item) if v.item in li.items else "money"
        if kind == "money":
            if not v.page_unit:
                if _unit_cancels(ctx):
                    soft.append(f"no unit header for {v.item} on {v.page_id}; ignored because a percent answer from one page does not depend on it")
                else:
                    problems.append(f"no unit header found for {v.item} on {v.page_id}")
                continue
            prev = _prior_magnitude(ctx, v)
            if prev is not None and v.value:
                ratio = abs(v.value) / max(abs(prev), 1e-9)
                if ratio > 60 or ratio < 1 / 60:
                    problems.append(f"{v.item} magnitude {v.value:.2f} crore vs previous filing {prev:.2f} crore (x{ratio:.1f})")
                elif ratio > 15 or ratio < 1 / 15:
                    soft.append(f"{v.item} is x{ratio:.1f} the previous filing")
        else:
            if ctx.answer.unit and ctx.answer.unit == "INR_crore" and len(ctx.values) == 1:
                problems.append("per-share item reported with a crore unit")
    if ctx.answer.unit == "INR_crore":
        for i in ctx.answer.inputs:
            if abs(i.value) > 5e6:
                problems.append(f"input {i.name}={i.value} is implausible in crore")
    if problems:
        return CheckResult(id="V3", status="hard_fail", message="; ".join(problems[:3]), details={"problems": problems})
    if soft:
        return CheckResult(id="V3", status="soft_fail", message="; ".join(soft[:2]))
    return CheckResult(id="V3", status="pass")


def _prior_magnitude(ctx: VerifyContext, v: ValueResult) -> float | None:
    if not v.company_id or not v.period or not v.scope:
        return None
    try:
        pk = PeriodKey.parse(v.period)
    except ValueError:
        return None
    key = (v.company_id, v.scope, v.item, str(pk.end))
    cache: dict[Any, float | None] = ctx.flags.setdefault("prior_mag", {})
    if key in cache:
        return cache[key]
    out: float | None = None
    for f in ctx.corpus.visible_filings(v.company_id, ctx.as_of):
        if f["period_end"] >= pk.end or f["filing_id"] == v.filing_id:
            continue
        for p in ctx.corpus.store.q(
            "SELECT tbl, scope, unit FROM pages WHERE filing_id = ? AND is_results_page", [f["filing_id"]]
        ):
            if p["scope"] not in (v.scope, "both") or not p["tbl"]:
                continue
            from pramaan.agent.normalize import from_parsed_json
            from pramaan.data.store import jload

            nt = from_parsed_json("", jload(p["tbl"]), p["scope"])
            cols = [c for c in nt.columns if c.period]
            if not cols:
                continue
            sel = select_value(nt, v.item, cols[0].period, v.scope)  # type: ignore[arg-type]
            if sel:
                out = to_answer_units(sel, nt.unit)
                break
        if out is not None:
            break
    cache[key] = out
    return out


def v4_identities(ctx: VerifyContext) -> CheckResult:
    li = line_items()
    seen_pages = {v.page_id: v for v in ctx.values.values() if v.found and v.page_id}
    problems: list[str] = []
    hard = False
    checked = 0
    for pid, vr in seen_pages.items():
        nt = ctx.tables.get(pid)
        if nt is None or not vr.period:
            continue
        pk = PeriodKey.parse(vr.period)
        for ident in li.identities:
            if vr.scope and vr.scope not in ident["scopes"]:
                continue
            names = [ident["lhs"], *[r.lstrip("-") for r in ident["rhs"]]]
            if vr.item not in names:
                continue
            vals: dict[str, float] = {}
            for n in names:
                sel = select_value(nt, n, pk, vr.scope)
                if sel is None:
                    break
                cv = to_answer_units(sel, nt.unit)
                if cv is None:
                    break
                vals[n] = cv
            else:
                checked += 1
                rhs = sum((-vals[r[1:]] if r.startswith("-") else vals[r]) for r in ident["rhs"])
                lhs = vals[ident["lhs"]]
                diff = abs(lhs - rhs)
                tol = max(li.tol["abs_crore"], li.tol["rel"] * max(abs(lhs), abs(rhs)))
                if ident["id"] == "total_tax":
                    tol = max(tol, 0.25 * abs(lhs))
                if diff > tol:
                    rel = diff / max(abs(lhs), 1e-9)
                    problems.append(f"identity {ident['id']} fails on {pid}: {lhs:.2f} vs {rhs:.2f}")
                    if rel > HARD_REL and ident["id"] != "total_tax":
                        hard = True
    if problems:
        return CheckResult(id="V4", status="hard_fail" if hard else "soft_fail", message="; ".join(problems[:2]), details={"problems": problems})
    return CheckResult(id="V4", status="pass" if checked else "skip", message="" if checked else "no identity could be tested")


def v5_period(ctx: VerifyContext) -> CheckResult:
    problems: list[str] = []
    for v in ctx.values.values():
        if not v.found or not v.period:
            continue
        pk = PeriodKey.parse(v.period)
        if v.column_header and parse_period(v.column_header) is not None and parse_period(v.column_header) != pk:
            problems.append(f"column {v.column_header} is not {v.period}")
            continue
        page = ctx.corpus.page(v.page_id, ctx.as_of) if v.page_id else None
        if page is None:
            continue
        dates = {d.value for d in parse_dates((page["text"] or "")[:6000])}
        if pk.end not in dates:
            problems.append(f"{pk.end} does not appear on page {v.page_id}")
    if problems:
        return CheckResult(id="V5", status="hard_fail" if any("is not" in p for p in problems) else "soft_fail", message="; ".join(problems[:2]))
    return CheckResult(id="V5", status="pass")


def v6_xbrl(ctx: VerifyContext) -> CheckResult:
    if not ctx.production or ctx.xbrl_lookup is None:
        return CheckResult(id="V6", status="skip", message="gym mode or no XBRL")
    problems: list[str] = []
    for v in ctx.values.values():
        if not v.found or v.value is None or not v.period:
            continue
        ref = ctx.xbrl_lookup(v.company_id, v.scope, v.item, PeriodKey.parse(v.period))
        if ref is None:
            continue
        tol = 0.0101
        if abs(ref - v.value) > tol:
            problems.append(f"{v.item}: PDF read {v.value} but exchange XBRL says {ref}")
    if problems:
        return CheckResult(id="V6", status="soft_fail", message="; ".join(problems[:2]), details={"disagreements": problems})
    return CheckResult(id="V6", status="pass")


def v7_cells(ctx: VerifyContext) -> CheckResult:
    if not ctx.flags.get("cell_rule"):
        return CheckResult(id="V7", status="skip", message="cell rule disabled")
    problems: list[str] = []
    for inp in ctx.answer.inputs:
        c = ctx.citations.get(inp.citation_id or "")
        if c is None:
            continue
        nt = ctx.tables.get(c.page_id)
        if nt is None:
            problems.append(f"no table read for {c.page_id}")
            continue
        cells = {_ws(v) for r in nt.rows for v in r.raw if v}
        if _ws(c.quote) not in cells:
            problems.append(f"'{c.quote}' is not a table cell on {c.page_id}")
    if problems:
        return CheckResult(id="V7", status="hard_fail", message="; ".join(problems[:2]), details={"problems": problems})
    return CheckResult(id="V7", status="pass")


def run_checks(ctx: VerifyContext) -> VerifyReport:
    checks = [v1_grounding(ctx), v2_arithmetic(ctx), v3_units(ctx), v4_identities(ctx), v5_period(ctx), v6_xbrl(ctx), v7_cells(ctx)]
    if any(c.status == "hard_fail" for c in checks):
        status = "hard_fail"
    elif any(c.status == "soft_fail" for c in checks):
        status = "soft_fail"
    else:
        status = "pass"
    return VerifyReport(status=status, checks=checks)  # type: ignore[arg-type]
