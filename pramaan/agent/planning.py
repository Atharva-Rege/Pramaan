from __future__ import annotations

from pramaan.core.config import gym_config
from pramaan.core.periods import PeriodKey
from pramaan.core.schemas import Intent, Plan, PlanStep

RATIO_INPUTS = {
    "net_profit_margin": ("net_profit_for_period", "revenue_from_operations"),
    "effective_tax_rate": ("total_tax_expense", "profit_before_tax"),
    "other_income_share_of_pbt": ("other_income", "profit_before_tax"),
}


def default_scope(intent: Intent) -> str:
    return intent.scope or "consolidated"


def plan_from_intent(intent: Intent, use_calculator: bool = True) -> Plan:
    scope = default_scope(intent)
    steps: list[PlanStep] = []

    def gv(sid: str, company: str, item: str, period: str) -> PlanStep:
        return PlanStep(id=sid, tool="get_value", args={"company": company, "scope": scope, "item": item, "period": period})

    company = intent.companies[0] if intent.companies else ""
    periods = intent.periods or [str(PeriodKey(gym_config().profile.target_period_end, 3))]
    cur = periods[0]
    op = intent.operation
    item = intent.items[0] if intent.items else (intent.metric or "revenue_from_operations")
    if op == "lookup":
        steps.append(gv("s1", company, item, cur))
    elif op in ("growth_yoy", "premise_check"):
        prev = str(PeriodKey.parse(cur).shift_years(-1))
        steps += [gv("s1", company, item, cur), gv("s2", company, item, prev)]
        if use_calculator:
            steps.append(PlanStep(id="s3", tool="calculate", args={"expression": "(cur - prev) / abs(prev) * 100", "variables": {"cur": "s1", "prev": "s2"}}))
    elif op == "ratio":
        num, den = RATIO_INPUTS.get(intent.ratio_name or "", ("net_profit_for_period", "revenue_from_operations"))
        steps += [gv("s1", company, num, cur), gv("s2", company, den, cur)]
        if use_calculator:
            steps.append(PlanStep(id="s3", tool="calculate", args={"expression": "num / den * 100", "variables": {"num": "s1", "den": "s2"}}))
    elif op == "cross_period":
        cum = periods[0]
        quarter = periods[1] if len(periods) > 1 else periods[0]
        steps += [gv("s1", company, item, cum), gv("s2", company, item, quarter)]
        if use_calculator:
            steps.append(PlanStep(id="s3", tool="calculate", args={"expression": "quarter / cum * 100", "variables": {"quarter": "s2", "cum": "s1"}}))
    elif op == "rank":
        metric = intent.metric or item
        entries = []
        k = 0
        for c in intent.companies:
            if metric in RATIO_INPUTS:
                num, den = RATIO_INPUTS[metric]
                a, b, r = f"s{k + 1}", f"s{k + 2}", f"s{k + 3}"
                steps += [gv(a, c, num, cur), gv(b, c, den, cur)]
                steps.append(PlanStep(id=r, tool="calculate", args={"expression": "num / den * 100", "variables": {"num": a, "den": b}}))
                entries.append({"company": c, "ref": r})
                k += 3
            else:
                a = f"s{k + 1}"
                steps.append(gv(a, c, metric, cur))
                entries.append({"company": c, "ref": a})
                k += 1
        steps.append(PlanStep(id=f"s{k + 1}", tool="rank", args={"entries": entries, "descending": True}))
    else:
        steps.append(gv("s1", company, item, cur))
    return Plan(steps=steps)
