from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pramaan.core.config import GymConfig, gym_config, line_items
from pramaan.core.ids import question_id, stable_hash
from pramaan.core.periods import PeriodKey
from pramaan.core.schemas import Gold, GoldFact, Intent
from pramaan.data.store import Store, jdump
from pramaan.gym import templates as T
from pramaan.gym.evidence import EvidenceIndex, Located
from pramaan.gym.facts import FactBook
from pramaan.gym.slotparse import CompanyIndex, parse_question, slots_match

MAX_ATTEMPTS = 10
REL_PRECISION = 0.0006


@dataclass
class Candidate:
    tier: str
    split: str
    filing_id: str
    company_ids: list[str]
    text: str
    slots: Intent
    gold: Gold
    gold_pages: list[str]
    gold_boxes: dict[str, list[list[float]]]
    as_of: datetime
    template_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def qid(self) -> str:
        return question_id(
            self.split,
            self.tier,
            {"t": self.template_id, "c": self.company_ids, "s": self.slots.model_dump(), "f": self.filing_id},
        )


def crore(v: float) -> float:
    return v / 1e7


def gfact(cid: str, scope: str, item: str, period: PeriodKey | str, v_inr: float, loc: Located, per_share: bool) -> GoldFact:
    return GoldFact(
        company_id=cid,
        scope=scope,  # type: ignore[arg-type]
        item=item,
        period=str(period),
        value=round(v_inr if per_share else crore(v_inr), 4),
        page_id=loc.page_id,
    )


def precision_crore(loc: Located) -> float:
    return loc.unit_inr / 1e7 * 10 ** (-loc.decimals)


class GymBuilder:
    def __init__(self, store: Store, cfg: GymConfig | None = None) -> None:
        self.store = store
        self.cfg = cfg or gym_config()
        self.li = line_items()
        self.fb = FactBook(store)
        self.cidx = CompanyIndex.from_store(store)
        prof = self.cfg.profile
        self.cur = PeriodKey(prof.target_period_end, 3)
        self.py = PeriodKey(prof.prior_year_period_end, 3)
        self.filings = store.filings("role = 'target' AND split IS NOT NULL AND pdf_path IS NOT NULL")
        self.companies = {c["company_id"]: c for c in store.q("SELECT * FROM companies")}
        self._ev: dict[str, EvidenceIndex] = {}
        self._phrases: dict[str, list[str]] = {}

    def ev(self, filing_id: str) -> EvidenceIndex:
        if filing_id not in self._ev:
            self._ev[filing_id] = EvidenceIndex(self.store, filing_id)
        return self._ev[filing_id]

    def company_phrases(self, cid: str) -> list[str]:
        if cid in self._phrases:
            return self._phrases[cid]
        c = self.companies[cid]
        names = [a for a in (c["aliases"] or []) if len(a) >= 3 and a != c["symbol"]]
        names.append(c["name"])
        ok: list[str] = []
        for n in dict.fromkeys(names):
            if self.cidx.find(f"{n}") == [cid]:
                ok.append(n)
        if not ok:
            ok = [c["name"]]
        ok.sort(key=lambda n: ("&" in n, abs(len(n) - 14)))
        self._phrases[cid] = ok[:4]
        return self._phrases[cid]

    def rng(self, *parts: object) -> random.Random:
        return random.Random(int(stable_hash([self.cfg.seed, *parts], 8), 16))

    def as_of_after(self, rng: random.Random, t: datetime) -> datetime:
        return (t + timedelta(minutes=rng.uniform(10, 36 * 60))).replace(microsecond=0)

    def render(
        self,
        rng: random.Random,
        pool: list[str],
        slots: Intent,
        fill: dict[str, Any],
    ) -> str | None:
        for _ in range(MAX_ATTEMPTS):
            tmpl = T.pick(rng, pool)
            f = {k: (v(rng) if callable(v) else v) for k, v in fill.items()}
            text = tmpl.format(**f)
            if slots_match(parse_question(text, self.cidx), slots):
                return text
        return None

    def item_phrase(self, item: str) -> Any:
        return lambda rng: T.pick(rng, T.ITEM_VARIANTS.get(item, [item.replace("_", " ")]))

    def scope_ok_pages(self, locs: list[Located | None]) -> tuple[list[str], dict[str, list[list[float]]]] | None:
        pages: list[str] = []
        boxes: dict[str, list[list[float]]] = {}
        for loc in locs:
            if loc is None:
                return None
            if loc.page_id not in pages:
                pages.append(loc.page_id)
            if loc.box:
                boxes.setdefault(loc.page_id, []).append(loc.box)
        return pages, boxes

    def build_candidates(self) -> list[Candidate]:
        out: list[Candidate] = []
        for f in self.filings:
            out.extend(self.for_filing(f))
        out.extend(self.cross_company())
        return out

    def for_filing(self, f: dict[str, Any]) -> list[Candidate]:
        cands: list[Candidate] = []
        cid, fid, split = f["company_id"], f["filing_id"], f["split"]
        ev = self.ev(fid)
        if not ev.pages:
            return cands
        for scope in self.fb.scopes(cid, self.cur):
            cands += self.t1(f, ev, scope)
            cands += self.t2(f, ev, scope)
            cands += self.t3(f, ev, scope)
            cands += self.t4(f, ev, scope)
            cands += self.t6(f, ev, scope)
        del cid, fid, split
        return cands

    def _locate(self, ev: EvidenceIndex, cid: str, scope: str, item: str, period: PeriodKey) -> tuple[float, Located] | None:
        v = self.fb.value(cid, scope, item, period)
        if v is None:
            return None
        loc = ev.locate(scope, item, v, period, self.cur)
        if loc is None:
            return None
        return v, loc

    def t1(self, f: dict[str, Any], ev: EvidenceIndex, scope: str) -> list[Candidate]:
        out: list[Candidate] = []
        cid = f["company_id"]
        for item in self.cfg.t1_items:
            hit = self._locate(ev, cid, scope, item, self.cur)
            if hit is None:
                continue
            v, loc = hit
            kind = self.li.kind(item)
            if abs(v) < 1e-9:
                continue
            gold_v = round(v, 2) if kind == "per_share" else round(crore(v), 2)
            unit = "INR_per_share" if kind == "per_share" else "INR_crore"
            slots = Intent(companies=[cid], scope=scope, items=[item], periods=[str(self.cur)], operation="lookup")  # type: ignore[arg-type]
            rng = self.rng("T1", f["filing_id"], scope, item)
            text = self.render(
                rng,
                T.T1,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "item": self.item_phrase(item),
                    "period": lambda r: T.pick(r, T.period_variants(self.cur)),
                    "unit": "₹ per share" if kind == "per_share" else "₹ crore",
                },
            )
            if text is None:
                continue
            pages, boxes = [loc.page_id], ({loc.page_id: [loc.box]} if loc.box else {})
            out.append(
                Candidate(
                    "T1",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    Gold(
                        answer_type="number",
                        value=gold_v,
                        unit=unit,
                        display_unit_crore=max(0.01, precision_crore(loc)) if kind != "per_share" else 0.01,
                        inputs={item: gold_v},
                        facts=[gfact(cid, scope, item, self.cur, v, loc, kind == "per_share")],
                    ),
                    pages,
                    boxes,
                    self.as_of_after(rng, f["pdf_published_at"]),
                    "T1",
                    {"approx": loc.approx, "ambiguous": loc.ambiguous, "item": item, "scope": scope},
                )
            )
        return out

    def t2(self, f: dict[str, Any], ev: EvidenceIndex, scope: str) -> list[Candidate]:
        out: list[Candidate] = []
        cid = f["company_id"]
        tol = self.cfg.tolerances
        for item in self.cfg.t2_items:
            cur = self._locate(ev, cid, scope, item, self.cur)
            prev = self._locate(ev, cid, scope, item, self.py)
            if cur is None or prev is None:
                continue
            (cv, cl), (pv, pl) = cur, prev
            if self.li.kind(item) == "money" and abs(crore(pv)) < tol["min_abs_denominator_crore"]:
                continue
            if abs(pv) < 1e-9 or (cv < 0) != (pv < 0):
                continue
            if self.li.kind(item) == "per_share" and abs(pv) < 0.5:
                continue
            growth = (cv - pv) / abs(pv) * 100
            if abs(growth) > tol["max_abs_growth_percent"]:
                continue
            if not self._precise(cl, cv, item) or not self._precise(pl, pv, item):
                continue
            slots = Intent(companies=[cid], scope=scope, items=[item], periods=[str(self.cur)], operation="growth_yoy")  # type: ignore[arg-type]
            rng = self.rng("T2", f["filing_id"], scope, item)
            text = self.render(
                rng,
                T.T2,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "item": self.item_phrase(item),
                    "period": lambda r: T.pick(r, T.period_variants(self.cur)),
                },
            )
            if text is None:
                continue
            pk = self.scope_ok_pages([cl, pl])
            if pk is None:
                continue
            pages, boxes = pk
            out.append(
                Candidate(
                    "T2",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    Gold(
                        answer_type="number",
                        value=round(growth, 1),
                        unit="percent",
                        inputs={"current": round(crore(cv), 4) if self.li.kind(item) == "money" else cv, "prior": round(crore(pv), 4) if self.li.kind(item) == "money" else pv},
                        facts=[
                            gfact(cid, scope, item, self.cur, cv, cl, self.li.kind(item) == "per_share"),
                            gfact(cid, scope, item, self.py, pv, pl, self.li.kind(item) == "per_share"),
                        ],
                    ),
                    pages,
                    boxes,
                    self.as_of_after(rng, f["pdf_published_at"]),
                    "T2",
                    {"item": item, "scope": scope, "approx": cl.approx or pl.approx},
                )
            )
        return out

    def _precise(self, loc: Located, v: float, item: str) -> bool:
        if self.li.kind(item) == "per_share":
            return abs(v) >= 10 ** (-loc.decimals) / REL_PRECISION * 0.01 or True
        return precision_crore(loc) / max(abs(crore(v)), 1e-9) <= REL_PRECISION * 1.0 + 0.0004

    def t3(self, f: dict[str, Any], ev: EvidenceIndex, scope: str) -> list[Candidate]:
        out: list[Candidate] = []
        cid = f["company_id"]
        for name, spec in self.cfg.ratios.items():
            num_item, den_item = spec["numerator"], spec["denominator"]
            num = self._locate(ev, cid, scope, num_item, self.cur)
            den = self._locate(ev, cid, scope, den_item, self.cur)
            if num is None or den is None:
                continue
            (nv, nl), (dv, dl) = num, den
            if dv <= 0 or abs(crore(dv)) < self.cfg.tolerances["min_abs_denominator_crore"]:
                continue
            if any(self.fb.value(cid, scope, req, self.cur) is None or self.fb.value(cid, scope, req, self.cur) <= 0 for req in spec.get("requires_positive", [])):  # type: ignore[operator]
                continue
            if not self._precise(nl, nv, num_item) or not self._precise(dl, dv, den_item):
                continue
            val = nv / dv * spec["scale"]
            if abs(val) > 1000:
                continue
            slots = Intent(companies=[cid], scope=scope, items=[], periods=[str(self.cur)], operation="ratio", ratio_name=name)  # type: ignore[arg-type]
            rng = self.rng("T3", f["filing_id"], scope, name)
            text = self.render(
                rng,
                T.T3,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "ratio": lambda r, name=name: T.pick(r, T.RATIO_LABELS[name]),
                    "period": lambda r: T.pick(r, T.period_variants(self.cur)),
                },
            )
            if text is None:
                continue
            pk = self.scope_ok_pages([nl, dl])
            if pk is None:
                continue
            pages, boxes = pk
            out.append(
                Candidate(
                    "T3",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    Gold(
                        answer_type="number",
                        value=round(val, 1),
                        unit="percent",
                        inputs={num_item: round(crore(nv), 4), den_item: round(crore(dv), 4)},
                        facts=[gfact(cid, scope, num_item, self.cur, nv, nl, False), gfact(cid, scope, den_item, self.cur, dv, dl, False)],
                    ),
                    pages,
                    boxes,
                    self.as_of_after(rng, f["pdf_published_at"]),
                    "T3",
                    {"ratio": name, "scope": scope},
                )
            )
        return out

    def t4(self, f: dict[str, Any], ev: EvidenceIndex, scope: str) -> list[Candidate]:
        out: list[Candidate] = []
        cid = f["company_id"]
        for q_end, cum_months in ((self.cur.end, 12), (self.cur.end, 6), (self.cur.end, 9)):
            cum_key, q_key = PeriodKey(q_end, cum_months), PeriodKey(q_end, 3)
            if cum_months == 12 and q_end.month != 3:
                continue
            if cum_months == 6 and q_end.month != 9:
                continue
            if cum_months == 9 and q_end.month != 12:
                continue
            outs = self._t4_pair(f, ev, scope, cid, cum_key, q_key)
            out += outs
        if self.cur.end.month == 6:
            prev_q = PeriodKey(self.cur.end.replace(month=3, day=31, year=self.cur.end.year), 3)
            prev_y = PeriodKey(prev_q.end, 12)
            out += self._t4_pair(f, ev, scope, cid, prev_y, prev_q)
        return out

    def _t4_pair(self, f: dict[str, Any], ev: EvidenceIndex, scope: str, cid: str, cum_key: PeriodKey, q_key: PeriodKey) -> list[Candidate]:
        out: list[Candidate] = []
        for item in ("revenue_from_operations", "net_profit_for_period", "profit_before_tax", "total_expenses"):
            cum = self._locate(ev, cid, scope, item, cum_key)
            q = self._locate(ev, cid, scope, item, q_key)
            if cum is None or q is None:
                continue
            (cv, cl), (qv, ql) = cum, q
            if cv <= 0 or abs(crore(cv)) < self.cfg.tolerances["min_abs_denominator_crore"]:
                continue
            if not self._precise(cl, cv, item) or not self._precise(ql, qv, item):
                continue
            share = qv / cv * 100
            if not (-500 < share < 500):
                continue
            slots = Intent(companies=[cid], scope=scope, items=[item], periods=[str(cum_key), str(q_key)], operation="cross_period")  # type: ignore[arg-type]
            rng = self.rng("T4", f["filing_id"], scope, item, str(cum_key))
            text = self.render(
                rng,
                T.T4,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "item": self.item_phrase(item),
                    "cum": lambda r: T.pick(r, T.period_variants(cum_key, allow_short=False)),
                    "quarter": lambda r: T.pick(r, T.period_variants(q_key, allow_short=False)),
                },
            )
            if text is None:
                continue
            pk = self.scope_ok_pages([cl, ql])
            if pk is None:
                continue
            pages, boxes = pk
            out.append(
                Candidate(
                    "T4",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    Gold(
                        answer_type="number",
                        value=round(share, 1),
                        unit="percent",
                        inputs={"cumulative": round(crore(cv), 4), "quarter": round(crore(qv), 4)},
                        facts=[gfact(cid, scope, item, cum_key, cv, cl, False), gfact(cid, scope, item, q_key, qv, ql, False)],
                    ),
                    pages,
                    boxes,
                    self.as_of_after(rng, f["pdf_published_at"]),
                    "T4",
                    {"item": item, "scope": scope, "cum": str(cum_key)},
                )
            )
        return out

    def t6(self, f: dict[str, Any], ev: EvidenceIndex, scope: str) -> list[Candidate]:
        out: list[Candidate] = []
        cid = f["company_id"]
        for item in ("revenue_from_operations", "net_profit_for_period", "total_income"):
            hit = self._locate(ev, cid, scope, item, self.cur)
            if hit is None:
                continue
            rng = self.rng("T6a", f["filing_id"], scope, item)
            slots = Intent(companies=[cid], scope=scope, items=[item], periods=[str(self.cur)], operation="lookup")  # type: ignore[arg-type]
            text = self.render(
                rng,
                T.T1,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "item": self.item_phrase(item),
                    "period": lambda r: T.pick(r, T.period_variants(self.cur)),
                    "unit": "₹ crore",
                },
            )
            if text is None:
                continue
            early = f["pdf_published_at"] - timedelta(hours=rng.uniform(2, 24 * 6))
            out.append(
                Candidate(
                    "T6",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    Gold(answer_type="unavailable", note="filing not published as of as_of"),
                    [],
                    {},
                    early.replace(microsecond=0),
                    "T6a",
                    {"subtype": "not_filed", "item": item, "scope": scope},
                )
            )
        for item in ("revenue_from_operations", "net_profit_for_period", "total_income"):
            cur = self._locate(ev, cid, scope, item, self.cur)
            prev = self._locate(ev, cid, scope, item, self.py)
            if cur is None or prev is None:
                continue
            (cv, cl), (pv, pl) = cur, prev
            if abs(crore(pv)) < self.cfg.tolerances["min_abs_denominator_crore"] or (cv < 0) != (pv < 0) or pv == 0:
                continue
            growth = (cv - pv) / abs(pv) * 100
            if abs(growth) > self.cfg.tolerances["max_abs_growth_percent"]:
                continue
            rng = self.rng("T6b", f["filing_id"], scope, item)
            slots = Intent(
                companies=[cid],
                scope=scope,  # type: ignore[arg-type]
                items=[item],
                periods=[str(self.cur)],
                operation="premise_check",
                premise_claim="decline",
            )
            pool = T.T6B_FALSE if growth > 2.0 else T.T6B_CTRL
            if growth > 2.0:
                gold = Gold(
                    answer_type="premise_false",
                    note=f"actual yoy change {growth:.1f}%",
                    facts=[gfact(cid, scope, item, self.cur, cv, cl, False), gfact(cid, scope, item, self.py, pv, pl, False)],
                )
                pk = self.scope_ok_pages([cl, pl])
            elif growth < -2.0:
                if not self._precise(cl, cv, item) or not self._precise(pl, pv, item):
                    continue
                gold = Gold(
                    answer_type="number",
                    value=round(growth, 1),
                    unit="percent",
                    facts=[gfact(cid, scope, item, self.cur, cv, cl, False), gfact(cid, scope, item, self.py, pv, pl, False)],
                )
                pk = self.scope_ok_pages([cl, pl])
                pool = T.T6B_CTRL
            else:
                continue
            if pk is None:
                continue
            text = self.render(
                rng,
                pool,
                slots,
                {
                    "company": lambda r: T.pick(r, self.company_phrases(cid)),
                    "scope": scope,
                    "item": self.item_phrase(item),
                    "period": lambda r: T.pick(r, T.period_variants(self.cur)),
                },
            )
            if text is None:
                continue
            pages, boxes = pk
            out.append(
                Candidate(
                    "T6",
                    f["split"],
                    f["filing_id"],
                    [cid],
                    text,
                    slots,
                    gold,
                    pages,
                    boxes,
                    self.as_of_after(rng, f["pdf_published_at"]),
                    "T6b",
                    {"subtype": "premise", "control": gold.answer_type == "number", "item": item, "scope": scope},
                )
            )
        return out

    def cross_company(self) -> list[Candidate]:
        out: list[Candidate] = []
        by_split: dict[str, list[dict[str, Any]]] = {}
        for f in self.filings:
            by_split.setdefault(f["split"], []).append(f)
        for split, fl in by_split.items():
            for metric in self.cfg.t5_metrics:
                vals: dict[str, tuple[float, list[Located], dict[str, Any], list[float]]] = {}
                for f in fl:
                    cid = f["company_id"]
                    ev = self.ev(f["filing_id"])
                    scopes = self.fb.scopes(cid, self.cur)
                    if "consolidated" not in scopes:
                        continue
                    inputs = {"net_profit_margin": ["net_profit_for_period", "revenue_from_operations"]}.get(metric, [metric])
                    locs: list[Located] = []
                    raw: list[float] = []
                    good = True
                    for it in inputs:
                        h = self._locate(ev, cid, "consolidated", it, self.cur)
                        if h is None or not self._precise(h[1], h[0], it):
                            good = False
                            break
                        raw.append(h[0])
                        locs.append(h[1])
                    if not good:
                        continue
                    if metric == "net_profit_margin":
                        if raw[1] <= 0:
                            continue
                        val = raw[0] / raw[1] * 100
                    else:
                        val = raw[0]
                    vals[cid] = (val, locs, f, raw)
                ids = sorted(vals)
                if len(ids) < 3:
                    continue
                rng = self.rng("T5", split, metric)
                seen: set[tuple[str, ...]] = set()
                for _ in range(min(60, len(ids) * 6)):
                    trio = tuple(sorted(rng.sample(ids, 3)))
                    if trio in seen:
                        continue
                    seen.add(trio)
                    ordered = sorted(trio, key=lambda c: -vals[c][0])
                    gaps_ok = all(
                        abs(vals[a][0] - vals[b][0]) >= 0.03 * max(abs(vals[a][0]), abs(vals[b][0]), 1e-9)
                        for a, b in itertools.pairwise(ordered)
                    )
                    if not gaps_ok:
                        continue
                    metric_item = metric
                    metric_inputs = {"net_profit_margin": ["net_profit_for_period", "revenue_from_operations"]}.get(metric, [metric])
                    slots = Intent(
                        companies=list(trio),
                        scope="consolidated",
                        items=[],
                        periods=[str(self.cur)],
                        operation="rank",
                        metric=metric_item,
                    )
                    label = T.RATIO_LABELS[metric][0] if metric in T.RATIO_LABELS else T.ITEM_VARIANTS[metric][0]
                    names = [T.pick(rng, self.company_phrases(c)) for c in trio]
                    order = list(range(3))
                    rng.shuffle(order)
                    text_names = T.join_companies([names[i] for i in order])
                    text = None
                    for _a in range(MAX_ATTEMPTS):
                        t = T.pick(rng, T.T5).format(companies=text_names, metric=label, period=T.pick(rng, T.period_variants(self.cur)))
                        parsed = parse_question(t, self.cidx)
                        if set(parsed.companies) == set(trio) and (parsed.metric == metric_item or parsed.metric == label):
                            text = t
                            break
                    if text is None:
                        continue
                    pages: list[str] = []
                    boxes: dict[str, list[list[float]]] = {}
                    for c in trio:
                        for loc in vals[c][1]:
                            if loc.page_id not in pages:
                                pages.append(loc.page_id)
                            if loc.box:
                                boxes.setdefault(loc.page_id, []).append(loc.box)
                    latest = max(vals[c][2]["pdf_published_at"] for c in trio)
                    out.append(
                        Candidate(
                            "T5",
                            split,
                            vals[ordered[0]][2]["filing_id"],
                            list(trio),
                            text,
                            slots,
                            Gold(
                                answer_type="ranking",
                                ranking=ordered,
                                inputs={c: round(vals[c][0], 4) for c in trio},
                                facts=[
                                    gfact(c, "consolidated", metric_inputs[0], self.cur, raw_v[0], vals[c][1][0], False)
                                    for c, raw_v in ((c, vals[c][3]) for c in trio)
                                ],
                            ),
                            pages,
                            boxes,
                            self.as_of_after(rng, latest),
                            "T5",
                            {"metric": metric, "filings": [vals[c][2]["filing_id"] for c in trio]},
                        )
                    )
        return out


def _take(avail: dict[str, list[Candidate]], picked: list[Candidate], counts: dict[str, int], tier: str, n: int, cap: int) -> int:
    got = 0
    rest: list[Candidate] = []
    for c in avail[tier]:
        if got < n and counts.get(c.filing_id, 0) < cap:
            picked.append(c)
            counts[c.filing_id] = counts.get(c.filing_id, 0) + 1
            got += 1
        else:
            rest.append(c)
    avail[tier] = rest
    return got


def select_questions(
    cands: list[Candidate],
    sizes: dict[str, int],
    tier_mix: dict[str, float],
    seed: int,
    max_per_filing: dict[str, int] | None = None,
) -> list[Candidate]:
    max_per_filing = max_per_filing or {"train": 5, "dev": 6, "test": 6}
    chosen: list[Candidate] = []
    for split, size in sizes.items():
        pool = [c for c in cands if c.split == split]
        quotas = {t: round(size * share) for t, share in tier_mix.items()}
        rng = random.Random(int(stable_hash([seed, split], 8), 16))
        avail = {t: [c for c in pool if c.tier == t] for t in tier_mix}
        for t in avail:
            rng.shuffle(avail[t])
        counts: dict[str, int] = {}
        picked: list[Candidate] = []
        cap = max_per_filing.get(split, 6)
        # Proportional filling: always take from the tier furthest behind its quota. The per
        # filing cap can bind (only filings with results pages yield candidates), and filling
        # tier by tier lets whichever tier goes first use up the capacity the others need.
        # Advancing all tiers at the same rate keeps the spec's mix when capacity runs out.
        progress = dict.fromkeys(quotas, 0)
        active = {t for t, q in quotas.items() if q > 0}
        while active:
            tier = min(active, key=lambda t: (progress[t] / quotas[t], t))
            # Prefer the filing with the most capacity left (ties keep the shuffled order), so an
            # abundant tier does not burn the cap of a filing that a scarcer tier needs.
            nxt, room = None, 0
            for i, cand in enumerate(avail[tier]):
                r = cap - counts.get(cand.filing_id, 0)
                if r > room:
                    nxt, room = i, r
            if nxt is None:
                active.discard(tier)
                continue
            c = avail[tier].pop(nxt)
            picked.append(c)
            counts[c.filing_id] = counts.get(c.filing_id, 0) + 1
            progress[tier] += 1
            if progress[tier] >= quotas[tier]:
                active.discard(tier)
        short = sum(quotas.values()) - len(picked)
        guard = 0
        while short > 0 and guard < 5:
            guard += 1
            for tier in sorted(avail, key=lambda t: -len(avail[t])):
                if short <= 0:
                    break
                short -= _take(avail, picked, counts, tier, short, cap)
        chosen.extend(picked)
    return chosen


def write_questions(store: Store, picks: list[Candidate], replace: bool = True) -> int:
    if replace:
        store.x("DELETE FROM questions")
    rows = []
    seen: set[str] = set()
    for c in picks:
        qid = c.qid
        if qid in seen:
            continue
        seen.add(qid)
        rows.append(
            {
                "qid": qid,
                "split": c.split,
                "tier": c.tier,
                "as_of": c.as_of,
                "text": c.text,
                "slots": jdump(c.slots.model_dump()),
                "gold": jdump(c.gold.model_dump()),
                "gold_pages": c.gold_pages,
                "human_checked": False,
                "template_id": c.template_id,
                "gold_boxes": jdump(c.gold_boxes),
                "meta": jdump({**c.meta, "filing_id": c.filing_id, "company_ids": c.company_ids, "paraphrase": "rules"}),
            }
        )
    store.insert_rows("questions", rows)
    return len(rows)
