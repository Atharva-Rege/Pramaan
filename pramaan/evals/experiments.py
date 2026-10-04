from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from pramaan.agent.factory import Runtime, build_agent
from pramaan.agent.normalize import from_parsed_json, select_value, to_answer_units
from pramaan.agent.tools import clear_table_cache
from pramaan.attribution.replay import attribute_records, counterfactual_replay, stage_distribution
from pramaan.core.config import gym_config, line_items, retrieval_config
from pramaan.core.paths import get_paths
from pramaan.core.periods import PeriodKey
from pramaan.core.splits import purpose
from pramaan.data.store import Store, jload
from pramaan.evals.harness import (
    EvalRecord,
    agent_errors,
    error_summary,
    evaluate,
    load_questions,
    run_episodes,
    run_meta,
    summarize,
    write_result,
)
from pramaan.evals.stats import bootstrap_mean_ci, mcnemar, paired_bootstrap_diff, percentile
from pramaan.gym.build import evidence_coverage
from pramaan.gym.evidence import EvidenceIndex
from pramaan.gym.facts import FactBook
from pramaan.retrieval.cascade import Retriever
from pramaan.retrieval.metrics import evaluate_ranking
from pramaan.retrieval.metrics import summarize as summarize_ranking
from pramaan.retrieval.queries import question_query
from pramaan.verify.calibration import Calibrator

ELIGIBLE = (
    "revenue_from_operations",
    "other_income",
    "total_income",
    "total_expenses",
    "finance_costs",
    "employee_benefits_expense",
    "depreciation_and_amortisation",
    "profit_before_tax",
    "total_tax_expense",
    "net_profit_for_period",
    "total_comprehensive_income",
    "basic_eps",
    "diluted_eps",
)


def e1_gym_validity(store: Store, echo: Any = print) -> dict[str, Any]:
    cfg = gym_config()
    cur = PeriodKey(cfg.profile.target_period_end, 3)
    li = line_items()
    fb = FactBook(store)
    counts = store.q("SELECT split, tier, count(*) AS n FROM questions WHERE split <> 'redteam' GROUP BY split, tier ORDER BY split, tier")
    agree = Counter()
    conflicts: list[dict[str, Any]] = []
    by_item: dict[str, Counter[str]] = {}
    kind_of: dict[str, str] = {}
    for f in store.filings("role = 'target' AND pdf_path IS NOT NULL"):
        fid, cid = f["filing_id"], f["company_id"]
        pages = store.q("SELECT page_id, scope, unit, tbl, ocr_layer, text_source FROM pages WHERE filing_id = ? AND is_results_page", [fid])
        if not pages:
            kind_of[fid] = "no_results_page"
            continue
        kind_of[fid] = "ocr_layer" if any(p["ocr_layer"] or p["text_source"] != "text_layer" for p in pages) else "digital"
        ev = EvidenceIndex(store, fid)
        for scope in fb.scopes(cid, cur):
            tables = [from_parsed_json(p["page_id"], jload(p["tbl"]), p["scope"]) for p in pages if p["tbl"] and p["scope"] in (scope, "both", None)]
            for item in ELIGIBLE:
                v = fb.value(cid, scope, item, cur)
                if v is None or abs(v) < 1e-9:
                    continue
                gold = v if li.kind(item) == "per_share" else v / 1e7
                ev.locate(scope, item, v, cur, cur)
                status = "no_row"
                for nt in tables:
                    sel = select_value(nt, item, cur, scope)
                    if sel is None:
                        continue
                    got = to_answer_units(sel, nt.unit)
                    if got is None:
                        status = "no_unit"
                        continue
                    tol = 0.0051 if li.kind(item) == "per_share" else max(0.011, 0.0005 * abs(gold))
                    status = "agree" if abs(got - gold) <= tol else "disagree"
                    if status == "agree":
                        break
                agree[status] += 1
                by_item.setdefault(item, Counter())[status] += 1
        conflicts.extend(ev.conflicts)
    rev = []
    for f in store.filings("role = 'target' AND pdf_path IS NOT NULL"):
        r = store.one(
            "SELECT sum(x.value_inr) AS v FROM xbrl_facts x WHERE x.filing_id = ? AND x.line_item = 'revenue_from_operations' AND x.is_latest AND x.duration_months = 3 AND x.scope = 'consolidated'",
            [f["filing_id"]],
        ) or store.one(
            "SELECT sum(x.value_inr) AS v FROM xbrl_facts x WHERE x.filing_id = ? AND x.line_item = 'revenue_from_operations' AND x.is_latest AND x.duration_months = 3",
            [f["filing_id"]],
        )
        if r and r["v"]:
            rev.append(r["v"] / 1e7)
    q3 = list(np.quantile(rev, [1 / 3, 2 / 3])) if rev else []
    human = store.q("SELECT question_ok, gold_ok FROM human_checks WHERE reviewer <> '__sample__'")
    n_h = len(human)
    n_total = sum(agree.values())
    payload = {
        "meta": run_meta("e1", "all", "none"),
        "question_counts": counts,
        "filing_kinds": dict(Counter(kind_of.values())),
        "evidence_coverage": evidence_coverage(store),
        "pdf_vs_xbrl_agreement": {
            "facts_checked": n_total,
            "agree": agree["agree"],
            "disagree": agree["disagree"],
            "no_row": agree["no_row"],
            "no_unit": agree["no_unit"],
            "agree_rate_when_read": (agree["agree"] / max(1, agree["agree"] + agree["disagree"])),
            "by_item": {k: dict(v) for k, v in sorted(by_item.items())},
        },
        "structural_conflicts": {"count": len(conflicts), "examples": conflicts[:25]},
        "company_size_revenue_crore_terciles": [float(x) for x in q3],
        "human_validation": {
            "reviewed": n_h,
            "question_error_rate": (sum(1 for h in human if h["question_ok"] is False) / n_h) if n_h else None,
            "gold_error_rate": (sum(1 for h in human if h["gold_ok"] is False) / n_h) if n_h else None,
            "status": "complete" if n_h >= cfg.human_check_sample else "pending: use the Review UI (`pramaan serve`, /review)",
        },
    }
    write_result("E1", "gym_validity", payload)
    ag = payload["pdf_vs_xbrl_agreement"]
    cov = payload["evidence_coverage"]["by_page_kind"]
    echo(
        f"E1 written: PDF vs XBRL agreement {ag['agree_rate_when_read']:.1%} when the row was read "
        f"({ag['agree']} agree, {ag['disagree']} disagree; {ag['no_row']} no row and {ag['no_unit']} no unit "
        f"of {n_total} facts). Gold evidence on digital pages {cov['digital']['rate']:.1%} "
        f"(spec P1 bar 85%), OCR layer pages {cov['ocr_layer']['rate']:.1%}."
    )
    return payload


def retrieval_queries(store: Store, split: str, subset: int | None, seed: int = 5) -> list[tuple[Any, str]]:
    names = {c["company_id"]: c["name"] for c in store.q("SELECT company_id, name FROM companies")}
    with purpose("split_stats" if split == "test" else "dev_eval"):
        qs = load_questions(store, split, None, seed, tiers=["T1", "T2", "T3", "T4"])
    qs = [q for q in qs if q.gold_pages and q.gold.answer_type == "number"]
    if subset and subset < len(qs):
        rng = np.random.default_rng(seed)
        qs = [qs[i] for i in sorted(rng.choice(len(qs), subset, replace=False))]
    return [(q, question_query(q, names.get((q.meta.get("company_ids") or [""])[0], ""))) for q in qs]


def e2_retrieval(rt: Runtime, split: str = "dev", subset: int | None = 300, finetuned_path: str | None = None, echo: Any = print) -> dict[str, Any]:
    store = rt.store
    pairs = retrieval_queries(store, split, subset)
    combos: list[tuple[str, str, str | None]] = [
        ("R1 BM25 only", "bm25", None),
        ("R1 dense only", "dense", None),
        ("R1+R2 BM25+dense RRF", "bm25+dense", None),
        ("R1+R2 with row docs", "bm25+dense+rows", None),
        ("R1-R3 + off-the-shelf cross-encoder", "rrf+ce", None),
    ]
    if finetuned_path:
        combos.append(("R1-R3 + fine-tuned cross-encoder", "rrf+ce", finetuned_path))
    out: dict[str, Any] = {}
    for mode in ("scoped", "open"):
        for scope_filter in (False, True):
            key = f"{mode}{'+scope' if scope_filter else ''}"
            rows = []
            for label, stages, ft in combos:
                from pramaan.retrieval.rerank import Reranker

                retriever = Retriever(rt.index, Reranker(model=ft) if ft else rt.retriever.reranker)
                per, lat = [], {"bm25": [], "dense": [], "fusion": [], "ce": [], "total": []}
                for q, text in pairs:
                    cid = (q.meta.get("company_ids") or [None])[0]
                    t0 = time.perf_counter()
                    r = retriever.search(
                        text,
                        company_id=cid if mode == "scoped" else None,
                        scope=q.slots.scope if scope_filter else None,
                        as_of=q.as_of,
                        k=10,
                        stages=stages,
                    )
                    lat["total"].append((time.perf_counter() - t0) * 1000)
                    for k_ in ("bm25", "dense", "fusion", "ce"):
                        lat[k_].append(r.timings_ms.get(k_, 0.0))
                    per.append(evaluate_ranking(r.page_ids(), set(q.gold_pages)))
                m = summarize_ranking(per)
                rows.append(
                    {
                        "stages": label,
                        **{k: round(v, 4) for k, v in m.items()},
                        "latency_ms": {k: round(float(np.mean(v)), 2) for k, v in lat.items()},
                        "n": len(per),
                    }
                )
                echo(f"E2 {key:14s} {label:42s} R@5 {m['recall@5']:.3f} nDCG@10 {m['ndcg@10']:.3f}")
            out[key] = rows
    payload = {"meta": run_meta("e2", split, "none", {"n_queries": len(pairs), "finetuned": finetuned_path}), "results": out}
    write_result("E2", f"retrieval_{split}", payload)
    return payload


def e8_season(store: Store, per_question: list[dict[str, Any]] | None = None, echo: Any = print) -> dict[str, Any]:
    rows = store.q(
        "SELECT f.filing_id, f.company_id, f.pdf_published_at, f.xbrl_published_at, f.revised, f.board_meeting_date, c.in_nifty200 "
        "FROM filings f JOIN companies c USING (company_id) WHERE f.role = 'target' AND f.pdf_published_at IS NOT NULL AND f.xbrl_published_at IS NOT NULL"
    )
    lags = np.asarray([(r["xbrl_published_at"] - r["pdf_published_at"]).total_seconds() / 60.0 for r in rows])
    all_lags = lags.tolist()

    def stats(arr: np.ndarray) -> dict[str, Any]:
        if len(arr) == 0:
            return {}
        return {
            "n": len(arr),
            "median_minutes": float(np.median(arr)),
            "p25_minutes": float(np.percentile(arr, 25)),
            "p75_minutes": float(np.percentile(arr, 75)),
            "p95_minutes": float(np.percentile(arr, 95)),
            "pdf_first_share": float((arr > 0).mean()),
            "xbrl_first_share": float((arr < 0).mean()),
            "within_30min_share": float((np.abs(arr) <= 30).mean()),
            "over_3h_share": float((arr > 180).mean()),
        }

    nifty = np.asarray([lag for lag, r in zip(all_lags, rows, strict=True) if r["in_nifty200"]])
    other = np.asarray([lag for lag, r in zip(all_lags, rows, strict=True) if not r["in_nifty200"]])
    hist_edges = [-1e9, -60, -5, 0, 5, 15, 30, 60, 180, 720, 1e9]
    hist = np.histogram(lags, bins=hist_edges)[0].tolist() if len(lags) else []
    revised = [r["filing_id"] for r in rows if r["revised"]]
    conflicts = store.q("SELECT count(*) AS n FROM adjudications WHERE verdict IN ('xbrl_correct', 'agent_correct', 'both_wrong')")
    adj = store.q("SELECT verdict, count(*) AS n FROM adjudications GROUP BY verdict")
    payload: dict[str, Any] = {
        "meta": run_meta("e8", "all", "none"),
        "lag": {"all": stats(lags), "nifty200": stats(nifty), "others": stats(other), "histogram_edges_minutes": hist_edges, "histogram": hist},
        "revisions": {"filings_with_revised_xbrl": len(revised), "share": len(revised) / max(1, len(rows))},
        "adjudications": {r["verdict"]: r["n"] for r in adj},
        "adjudicated_total": conflicts[0]["n"] if conflicts else 0,
    }
    if per_question:
        qrows = {r["qid"]: r for r in per_question}
        fil = {f["filing_id"]: f for f in store.q("SELECT filing_id, scanned_pages FROM filings")}
        size = {}
        for f in store.filings("role = 'target'"):
            v = store.one(
                "SELECT max(value_inr) AS v FROM xbrl_facts WHERE filing_id = ? AND line_item = 'revenue_from_operations' AND is_latest AND duration_months = 3",
                [f["filing_id"]],
            )
            size[f["filing_id"]] = (v["v"] or 0) / 1e7 if v else 0
        vals = sorted(size.values())
        if vals:
            t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
            by_size: dict[str, list[float]] = {"small": [], "mid": [], "large": []}
            by_kind: dict[str, list[float]] = {"digital": [], "scanned_or_ocr_layer": []}
            for r in qrows.values():
                fid = r["filing"]
                s = size.get(fid, 0)
                by_size["small" if s <= t1 else ("mid" if s <= t2 else "large")].append(1.0 if r["correct"] else 0.0)
                scanned = (fil.get(fid, {}).get("scanned_pages") or 0) > 0
                by_kind["scanned_or_ocr_layer" if scanned else "digital"].append(1.0 if r["correct"] else 0.0)
            payload["accuracy_by_size"] = {k: {"n": len(v), "accuracy": float(np.mean(v)) if v else None} for k, v in by_size.items()}
            payload["accuracy_by_scan"] = {k: {"n": len(v), "accuracy": float(np.mean(v)) if v else None} for k, v in by_kind.items()}
    write_result("E8", "season", payload)
    echo(f"E8 written: {len(rows)} filings with both timestamps")
    return payload


def e3_compare(
    rt: Runtime,
    configs: list[str],
    split: str,
    subset: int | None,
    workers: int,
    baselines: dict[str, Any] | None = None,
    llm_mode: str = "api",
    echo: Any = print,
    instructions: Any = None,
    memory: Any = None,
    tag: str = "",
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    corpus_resolver = lambda n: (rt.corpus.resolve_company(n, limit=1) or [{}])[0].get("company_id")  # noqa: E731
    recs_by_name: dict[str, list[EvalRecord]] = {}
    why = "final_eval" if split == "test" else "dev_eval"
    for name in configs:
        if baselines and name in baselines:
            agent = baselines[name]
        else:
            agent = build_agent(rt, name, instructions=instructions, memory=memory)
        clear_table_cache()  # each config pays for its own table reads (see harness.evaluate)
        with purpose(why):
            qs = load_questions(rt.store, split, subset, seed=7)
            run_id = f"{name}{tag}-{split}-{int(time.time())}"
            t0 = time.time()
            recs = run_episodes(agent, qs, rt.store, run_id, workers, corpus_resolver, persist=True)
        if agent_errors(recs):
            echo(f"E3 {name}: {error_summary(recs)}; result NOT written")
            continue
        summary = summarize(recs)
        meta = run_meta(name, split, llm_mode, {"subset": subset, "wall_seconds": round(time.time() - t0, 1), "tag": tag})
        from pramaan.evals.harness import per_question_rows, register_run

        register_run(rt.store, run_id, name, split, summary["cost_total_usd"], meta)
        payload = {"meta": meta, "run_id": run_id, "summary": summary, "per_question": per_question_rows(recs)}
        write_result("E3", f"{name}{tag}_{split}", payload)
        results[name] = payload
        recs_by_name[name] = recs
        echo(f"E3 {name:26s} acc {summary['accuracy']:.3f} ver {summary['verified_accuracy']:.3f} $/q {summary['cost_per_question_usd']:.4f} p50 {summary['latency_ms_p50']:.0f}ms")
    comparisons: dict[str, Any] = {}
    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra = {r.q.qid: r for r in recs_by_name[a]}
            rb = {r.q.qid: r for r in recs_by_name[b]}
            common = sorted(set(ra) & set(rb))
            ca = [1.0 if ra[q].g.verified_correct else 0.0 for q in common]
            cb = [1.0 if rb[q].g.verified_correct else 0.0 for q in common]
            cl = [ra[q].filing for q in common]
            comparisons[f"{a}_vs_{b}"] = {
                "n": len(common),
                "paired_bootstrap_verified": paired_bootstrap_diff(ca, cb, cl),
                "mcnemar_verified": mcnemar([x > 0 for x in ca], [x > 0 for x in cb]),
            }
    summary_payload = {
        "meta": run_meta("e3", split, llm_mode, {"configs": names, "subset": subset, "tag": tag}),
        "table": {n: {k: v for k, v in results[n]["summary"].items() if k not in ("risk_coverage", "by_tier")} | {"by_tier": results[n]["summary"]["by_tier"]} for n in names},
        "pareto": [
            {"config": n, "cost_per_question_usd": results[n]["summary"]["cost_per_question_usd"], "verified_accuracy": results[n]["summary"]["verified_accuracy"], "accuracy": results[n]["summary"]["accuracy"]}
            for n in names
        ],
        "comparisons": comparisons,
    }
    write_result("E3", f"summary{tag}_{split}", summary_payload)
    return summary_payload


def e4_calibration(rt: Runtime, run_id: str, config: str, split: str = "dev", target_acc: float = 0.95, echo: Any = print) -> dict[str, Any]:
    store = rt.store
    rows = store.q("SELECT e.qid, e.grade, e.diagnostics, e.answer FROM episodes e WHERE e.run_id = ?", [run_id])
    feats, ok = [], []
    for r in rows:
        d = jload(r["diagnostics"], {})
        g = jload(r["grade"], {})
        f = d.get("features")
        if f is None:
            continue
        feats.append(f)
        ok.append(bool(g.get("correct")))
    if len(feats) < 20:
        return {"error": "need at least 20 graded episodes with features"}
    n = len(feats)
    idx = np.arange(n)
    rng = np.random.default_rng(3)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    oof = np.zeros(n)
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
        cal = Calibrator.fit([feats[i] for i in train_idx], [ok[i] for i in train_idx])
        for i in test_idx:
            oof[i] = cal.predict_proba(feats[i])
    from pramaan.evals.stats import expected_calibration_error, risk_coverage

    rc = risk_coverage(oof, ok)
    final = Calibrator.fit(feats, ok)
    thr = final.choose_threshold(feats, ok, target_acc)
    final.save(get_paths().artifacts / "calibrator.json")
    base_rc = risk_coverage([f.get("llm_conf", 0.5) for f in feats], ok)
    payload = {
        "meta": run_meta(config, split, "api", {"run_id": run_id}),
        "n": n,
        "base_accuracy": float(np.mean(ok)),
        "aurc_calibrated_cv": rc["aurc"],
        "aurc_llm_confidence": base_rc["aurc"],
        "risk_coverage_llm": {"coverage": base_rc["coverage"], "risk": base_rc["risk"]},
        "selective_accuracy_at_90_coverage": rc["selective_acc_at_90"],
        "ece_cv": expected_calibration_error(oof, ok),
        "threshold": thr,
        "risk_coverage": {"coverage": rc["coverage"], "risk": rc["risk"]},
    }
    write_result("E4", f"calibration_{config}_{split}", payload)
    echo(f"E4: AURC {rc['aurc']:.4f}, selective acc@90% {rc['selective_acc_at_90']:.3f}")
    return payload


def e6_attribution(rt: Runtime, config: str, run_id_before: str, run_id_after: str | None, split: str = "dev", max_replay: int = 200, echo: Any = print, llm_mode: str = "api") -> dict[str, Any]:
    from pramaan.core.schemas import Episode, Grade
    from pramaan.gym.grade import question_from_row

    def load(run_id: str) -> list[EvalRecord]:
        recs = []
        for r in rt.store.q("SELECT e.*, q.* FROM episodes e JOIN questions q USING (qid) WHERE e.run_id = ?", [run_id]):
            q = question_from_row(r)
            d = jload(r["diagnostics"], {})
            ep = Episode(qid=q.qid, config=config, cost_usd=r["cost_usd"], latency_ms=r["latency_ms"], abstained=d.get("abstained", False), escalated=d.get("escalated", False), retries=d.get("retries", 0), tier_used=d.get("tier_used", ""))
            from pramaan.core.schemas import Answer, Intent, Plan, ValueResult, VerifyReport

            ep.answer = Answer.model_validate(jload(r["answer"])) if r["answer"] and r["answer"] != "null" else None
            ep.intent = Intent.model_validate(d["intent"]) if d.get("intent") else None
            ep.plan = Plan.model_validate(d["plan"]) if d.get("plan") else None
            ep.value_results = [ValueResult.model_validate(v) for v in d.get("values", [])]
            ep.search_log = d.get("search", [])
            ep.draft_by_tier = {k: Answer.model_validate(v) for k, v in d.get("drafts", {}).items()}
            vr = jload(r["verify"], None)
            ep.verify = VerifyReport.model_validate(vr) if vr else None
            recs.append(EvalRecord(q, ep, Grade.model_validate(jload(r["grade"]))))
        return recs

    before = load(run_id_before)
    attrib_b = attribute_records(rt.store, before, rt.corpus)
    payload: dict[str, Any] = {"meta": run_meta(config, split, llm_mode), "before": stage_distribution(attrib_b)}
    agent = build_agent(rt, config)
    with purpose("dev_eval" if split != "test" else "final_eval"):
        payload["counterfactual"] = counterfactual_replay(rt.store, agent, before, max_replay)
    if run_id_after:
        after = load(run_id_after)
        payload["after"] = stage_distribution(attribute_records(rt.store, after, rt.corpus))
    write_result("E6", f"attribution_{config}_{split}", payload)
    echo(f"E6: {payload['before']['n_failed']} failures attributed")
    return payload


def timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


__all__ = ["bootstrap_mean_ci", "e1_gym_validity", "e2_retrieval", "e3_compare", "e4_calibration", "e6_attribution", "e8_season", "evaluate", "json", "percentile", "retrieval_config"]
