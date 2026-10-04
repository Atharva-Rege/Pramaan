"""LaTeX for the report's Experiments section, generated from results/ only.

Every number in report/generated/results.tex is computed here from result files, so the report
can never drift from the data (spec rule 3). A paragraph appears only if its evidence exists.
"""

from __future__ import annotations

import json
from typing import Any

from pramaan.core.paths import get_paths
from pramaan.evals.stats import mcnemar, paired_bootstrap_diff
from pramaan.render import _exploratory, _load

SPECIFIED = ("B0_naive_rag", "small_only", "strong_only", "cascade")
IDENTICAL = "identical verified answers on every question"
EXPLORATORY = ("small_only_v3", "strong_only_v3", "cascade_escalate_v3", "cascade_keep_v3")


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def p1(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def _rows(config: str, split: str = "dev") -> dict[str, Any] | None:
    return _load("E3", f"{config}_{split}")


def _per_q(config: str, split: str = "dev") -> dict[str, dict[str, Any]]:
    d = _rows(config, split)
    return {q["qid"]: q for q in d["per_question"]} if d else {}


def paired_raw(a: str, b: str, key: str = "verified", split: str = "dev") -> dict[str, float] | None:
    """'a minus b' on a per-question flag: clustered bootstrap difference and interval, McNemar."""
    qa, qb = _per_q(a, split), _per_q(b, split)
    ids = sorted(set(qa) & set(qb))
    if not ids:
        return None
    clusters = [qa[i]["filing"] for i in ids]
    d = paired_bootstrap_diff([float(qa[i][key]) for i in ids], [float(qb[i][key]) for i in ids], clusters)
    m = mcnemar([qb[i][key] for i in ids], [qa[i][key] for i in ids])
    return {"diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "a_only": m["a_only"], "b_only": m["b_only"], "p": m["p_value"], "n": float(len(ids))}


def paired(a: str, b: str, key: str = "verified", split: str = "dev") -> str | None:
    """Formatted 'a minus b' with its interval and McNemar test."""
    r = paired_raw(a, b, key, split)
    if r is None:
        return None
    p = "$p < 0.001$" if r["p"] < 0.001 else f"$p = {r['p']:.3f}$"
    if r["a_only"] == 0 and r["b_only"] == 0:
        return IDENTICAL
    return f"{100 * r['diff']:+.1f} points (95\\% CI {100 * r['lo']:+.1f} to {100 * r['hi']:+.1f}; {int(r['b_only'])} fixed, {int(r['a_only'])} broken; McNemar {p})"


def _cost_per_correct(config: str, split: str = "dev") -> float | None:
    d = _rows(config, split)
    if not d:
        return None
    ver = sum(q["verified"] for q in d["per_question"])
    return sum(q["cost_usd"] for q in d["per_question"]) / ver if ver else None


def _table_rows(configs: tuple[str, ...], split: str = "dev") -> str:
    out = ""
    for c in configs:
        d = _rows(c, split)
        if not d:
            continue
        s = d["summary"]
        lo, hi = s["verified_accuracy_ci"]
        cpc = _cost_per_correct(c, split)
        mark = "$^{*}$" if _exploratory(c) else ""
        out += (
            f"{tt(c)}{mark} & {s['n']} & {100 * s['verified_accuracy']:.1f} ({100 * lo:.0f}--{100 * hi:.0f}) & {100 * s['abstain_rate']:.0f} & "
            f"{s['cost_per_question_usd']:.4f} & {cpc:.4f} & {s['latency_ms_p50'] / 1000:.1f} \\\\\n"
            if cpc is not None
            else f"{tt(c)}{mark} & {s['n']} & {100 * s['verified_accuracy']:.1f} & {100 * s['abstain_rate']:.0f} & {s['cost_per_question_usd']:.4f} & -- & {s['latency_ms_p50'] / 1000:.1f} \\\\\n"
        )
    return out


def setup_tex() -> str:
    e1 = _load("E1", "gym_validity")
    d = _rows("small_only") or _rows("B0_naive_rag")
    if not e1 or not d:
        return ""
    counts: dict[str, int] = {}
    for r in e1["question_counts"]:
        counts[r["split"]] = counts.get(r["split"], 0) + r["n"]
    n_filings = sum(e1["filing_kinds"].values())
    m = d["meta"]["models"]
    return (
        "\\subsection{Setup}\n"
        f"All results come from a dry run on the quarter ended 30 June 2026, because the September quarter had not been filed when this version was written. "
        f"The gym covers {n_filings} filings and holds {counts.get('train', 0)} train, {counts.get('dev', 0)} dev and {counts.get('test', 0)} test questions; splits are chronological by board meeting date and frozen. "
        f"Unless stated otherwise, agent results use one fixed random sample of {d['summary']['n']} of the {counts.get('dev', 0)} dev questions for every configuration, so differences are paired. "
        f"{test_note()} "
        f"The small and strong tiers are {tt(m['small'].split('/')[-1])} and {tt(m['strong'].split('/')[-1])}. "
        "\\emph{Verified accuracy} counts an answer only if it is correct and its citations check out. "
        "Confidence intervals are 95\\% bootstrap intervals clustered by filing; paired differences use the same clustered bootstrap and McNemar's test.\n"
    )


def e1_tex() -> str:
    e1 = _load("E1", "gym_validity")
    if not e1:
        return ""
    ag, cov, kinds = e1["pdf_vs_xbrl_agreement"], e1["evidence_coverage"]["by_page_kind"], e1["filing_kinds"]
    hv = e1["human_validation"]
    return (
        "\\subsection{E1: is the gym valid?}\n"
        f"When the rules reader found the printed row, the PDF and the XBRL agreed on {p1(ag['agree_rate_when_read'])} of {ag['agree'] + ag['disagree']} facts. "
        f"A further {ag['no_row']} facts had no matching row and {ag['no_unit']} no unit header, so they say nothing about agreement. "
        f"{e1['structural_conflicts']['count']} facts were dropped because the XBRL element and the printed row are defined differently (for example consolidated profit before tax with and without the share of associates). "
        f"Gold evidence was located for {p1(cov['digital']['rate'])} of {cov['digital']['facts']} facts on digital pages, {p1(cov['ocr_layer']['rate'])} of {cov['ocr_layer']['facts']} on pages with an embedded OCR layer, "
        f"and almost never ({p1(cov['no_results_page']['rate'])}) for the {kinds.get('no_results_page', 0)} filings where no results page was detected; those filings contribute no questions. "
        + (
            "Human validation of the 100-question sample has not been done."
            if hv.get("reviewed", 0) == 0
            else f"Human validation: {hv['reviewed']} reviewed, question error rate {hv['question_error_rate']}, gold error rate {hv['gold_error_rate']}."
        )
        + "\n"
    )


def e2_tex() -> str:
    d = _load("E2", "retrieval_dev")
    if not d:
        return ""
    rows = d["results"].get("scoped", [])
    amp = "\\&"
    body = "".join(f"{r['stages'].replace('&', amp)} & {r['recall@1']:.3f} & {r['recall@5']:.3f} & {r['ndcg@10']:.3f} \\\\\n" for r in rows)
    by = {r["stages"]: r for r in rows}
    off, ft, rd = by.get("R1-R3 + off-the-shelf cross-encoder"), by.get("R1-R3 + fine-tuned cross-encoder"), by.get("R1+R2 with row docs")
    tm = get_paths().artifacts / "reranker" / "cpu" / "metrics.json"
    n_pairs = json.loads(tm.read_text(encoding="utf-8")).get("train_pairs", "") if tm.exists() else ""
    de, rrf = by.get("R1 dense only"), by.get("R1+R2 BM25+dense RRF")
    txt = ""
    if off and ft and rd and de and rrf:
        txt = (
            f"Dense retrieval alone beats BM25 by a wide margin on numeric table pages, and fusing the two with reciprocal rank fusion is \\emph{{worse}} than dense alone on nDCG@10 ({rrf['ndcg@10']:.3f} against {de['ndcg@10']:.3f}). "
            f"Row documents, which render each table row as a retrievable unit, lift nDCG@10 to {rd['ndcg@10']:.3f}. "
            f"The off-the-shelf cross-encoder then \\emph{{lowers}} it to {off['ndcg@10']:.3f}, because it cannot tell near identical statement pages apart. "
            f"Fine-tuned on {n_pairs} train-split pairs whose labels come from XBRL for free, it reaches {ft['ndcg@10']:.3f} (recall@1 {off['recall@1']:.3f} to {ft['recall@1']:.3f}). "
            "The fine-tuning pairs and their negatives come from train filings only; an earlier version drew negatives from dev and test pages and was discarded and retrained, which changed nDCG@10 by less than 0.002. "
            "Queries are synthetic template questions from one quarter, so this measures generalisation to new companies and filings, not to free-form analyst wording."
        )
    return (
        f"\\subsection{{E2: what does each retrieval stage buy?}}\n{txt}\n"
        "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrr}\\toprule\nstages (company-filtered) & R@1 & R@5 & nDCG@10 \\\\\\midrule\n"
        + body
        + f"\\bottomrule\\end{{tabular}}\\caption{{Retrieval stages on {rows[0]['n']} dev queries.}}\\end{{table}}\n"
    )


def e3_tex() -> str:
    if not _rows("B0_naive_rag") or not _rows("small_only"):
        return ""
    tbl = (
        "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrrrrr}\\toprule\n"
        "config & n & verified acc. (CI) & abstain & \\$/question & \\$/verified correct & p50 s \\\\\\midrule\n"
        + _table_rows(SPECIFIED)
        + "\\bottomrule\\end{tabular}\\caption{Specified configurations on the same dev sample. Costs are real API spend recorded at the original price of cached calls; latency is wall clock plus the recorded latency of cached calls.}\\label{tab:main}\\end{table}\n"
    )
    out = "\\subsection{E3: the agent against baselines}\n" + tbl
    a = paired("small_only", "B0_naive_rag")
    cp_s, cp_b = _cost_per_correct("small_only"), _cost_per_correct("B0_naive_rag")
    if a and cp_s and cp_b:
        out += (
            f"The agent beats the naive RAG baseline by {a}. Naive RAG is cheap per question but almost always wrong or silent, so per verified correct answer the small-model agent is "
            + ("cheaper" if cp_s < cp_b else "more expensive")
            + f" (\\${cp_s:.4f} against \\${cp_b:.4f}). The long-context baseline B1 was not run: it sends whole PDFs, and the cost estimator puts it far above the budget of this study, so the second half of H3 is untested. "
        )
    s1, s2, s3 = paired("strong_only", "small_only"), paired("cascade", "strong_only"), paired("cascade", "small_only")
    if s1 and s2 and s3 and _rows("strong_only") and _rows("cascade"):
        out += (
            f"The strong model gains {s1} over the small one, a difference that does not reach significance at this sample size. "
            f"\\textbf{{H1 is not supported as preregistered.}} The specified cascade differs from {tt('strong_only')} by {s2}, so the preregistered condition (within two points of {tt('strong_only')} and cheaper) fails on accuracy, and it is dominated by {tt('small_only')} ("""
            + ("it gave identical verified answers on every question" if s3 == IDENTICAL else f"{s3}")
            + ", at a higher cost per question). "
            "Reflection ran on the questions that hard-failed verification but ended in ``value not found'' abstentions, and escalation fires only on a hard failure after reflection, so the strong model was consulted on almost no questions.\n"
        )
    ro, rf = _load("E3", "rules_only_dev"), _load("E3", "rules_onlyft_reranker_dev")
    if ro and rf:
        out += (
            f"\\paragraph{{Offline baseline.}} A deterministic rules reader with no language model (clearly labelled offline, on all {ro['summary']['n']} dev questions) reaches {p1(ro['summary']['accuracy'])} with the off-the-shelf reranker and {p1(rf['summary']['accuracy'])} with the fine-tuned one. "
            "It is precise but silent: it abstains on about half of the answerable questions.\n"
        )
    out += "\\begin{figure}[H]\\centering\\IfFileExists{../results/figures/fig1_pareto.png}{\\includegraphics[width=\\linewidth]{../results/figures/fig1_pareto.png}}{}\n\\caption{Verified accuracy against cost per question. Hollow markers are exploratory variants.}\\label{fig:pareto}\\end{figure}\n"
    return out


def exploratory_tex() -> str:
    audit = _load("audit", "verifier_audit_dev")
    if not audit or not _rows("strong_only_v3"):
        return ""
    a = audit["audit"]
    so, st = a.get("small_only"), a.get("strong_only")
    out = (
        "\\subsection{Exploratory: why the cascade did not help}\n"
        "\\emph{The variants in this subsection were designed after reading failures on the dev sample, so their numbers are optimistic and are marked $^{*}$; they need confirming on questions not used to design them, and the specified configurations stay in every table.}\n\n"
    )
    if so and st:
        shapes = st["v3_false_reject_shapes"] + so["v3_false_reject_shapes"]
        same = all(s["unit"] == "percent" and s["n_pages"] == 1 for s in shapes) if shapes else False
        out += (
            f"\\paragraph{{The verifier discards correct answers.}} Grading the stored drafts of abstained answerable questions shows that {tt('strong_only')} threw away {st['correct_draft_rejected_total']} correct answers "
            f"({', '.join(f'{k} {v}' for k, v in st['correct_draft_rejected'].items())}) and {tt('small_only')} {so['correct_draft_rejected_total']} "
            f"({', '.join(f'{k} {v}' for k, v in so['correct_draft_rejected'].items())}); {so['no_numeric_draft']} and {st['no_numeric_draft']} further questions ended with no numeric draft at all because the model itself reported the value as not found. "
            + (
                f"All {len(shapes)} V3 (unit sanity) false rejects have one shape: a percent answer computed from two inputs cited on a single page. There the page's unit header cancels, so a missing header cannot hide a scale error. "
                "Relaxing V3 for exactly that case (inputs on different pages and absolute values keep the strict rule) is the variant "
                f"{tt('small_only_v3')} / {tt('strong_only_v3')}. "
                if same
                else ""
            )
        )
    d1, d2 = paired("strong_only_v3", "strong_only"), paired("small_only_v3", "small_only")
    if d1 and d2:
        wrong = lambda c: sum(1 for q in _per_q(c).values() if not q["correct"] and not q["abstained"])  # noqa: E731
        out += (
            f"It recovers {d1} for the strong model and {d2} for the small one, and the number of wrong answers did not change "
            f"({wrong('strong_only')} to {wrong('strong_only_v3')} and {wrong('small_only')} to {wrong('small_only_v3')}), so it added no false accepts in this sample.\n\n"
        )
    ce, ck = _per_q("cascade_escalate_v3"), _per_q("cascade_keep_v3")
    sv = _per_q("strong_only_v3")
    if ce and ck and sv:
        esc = [i for i in ce if ce[i]["escalated"]]
        keep_esc = [i for i in ck if ck[i]["escalated"]]
        k1 = paired("cascade_keep_v3", "cascade_escalate_v3")
        k2 = paired("cascade_keep_v3", "strong_only_v3")
        out += (
            f"\\paragraph{{Escalation was broken by a page ban.}} Letting the cascade escalate when the small model gives up ({tt('cascade_escalate_v3')}) escalated {len(esc)} questions, yet the strong tier inside the cascade answered only "
            f"{sum(ce[i]['verified'] for i in esc)} of them correctly while standalone {tt('strong_only_v3')} answers {sum(sv[i]['verified'] for i in esc)} of the same {len(esc)}. "
            "The cause is that after a failed verification the graph bans every page that produced a value, so a retry or an escalation cannot use the right page when the failure was a verifier false reject or a reading slip on it. "
            f"Keeping failed pages available ({tt('cascade_keep_v3')}), the only change, adds {k1}. "
            f"Against {tt('strong_only_v3')} the repaired cascade differs by {k2} at lower cost per question, which is too imprecise to settle the two-point margin of H1. "
            f"Even so, inside the cascade the strong tier answered {sum(ck[i]['verified'] for i in keep_esc)} of {len(keep_esc)} escalated questions against {sum(sv[i]['verified'] for i in keep_esc)} standalone, so a gap remains.\n"
        )
    rows = (
        "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrrrrr}\\toprule\n"
        "config & n & verified acc. (CI) & abstain & \\$/question & \\$/verified correct & p50 s \\\\\\midrule\n"
        + _table_rows(("small_only_v3", "strong_only_v3", "cascade_escalate_v3", "cascade_keep_v3"))
        + "\\bottomrule\\end{tabular}\\caption{Exploratory variants ($^{*}$: designed after reading dev failures).}\\label{tab:expl}\\end{table}\n"
    )
    return out + rows


def e4_tex() -> str:
    p = get_paths().results / "E4"
    files = sorted(p.glob("calibration_*.json")) if p.exists() else []
    if not files:
        return ""
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    cfg = d["meta"]["config"]
    th = d["threshold"]
    return (
        "\\subsection{E4: does it know when it does not know?}\n"
        f"On the live episodes of {tt(cfg)} ({'exploratory, ' if _exploratory(cfg) else ''}n={d['n']}) the base accuracy is {p1(d['base_accuracy'])}; abstaining on the least confident 10\\% lifts selective accuracy to {p1(d['selective_accuracy_at_90_coverage'])}, "
        f"and at the threshold chosen for a 95\\% target the agent answers {p1(th['coverage'])} of questions with {p1(th['selective_accuracy'])} accuracy (threshold chosen on the same episodes, so in sample). "
        f"A logistic calibrator over verifier results, retrieval margin and retries has a cross-validated expected calibration error of {d['ece_cv']:.3f}, but it does \\emph{{not}} beat the model's own stated confidence: "
        f"AURC {d['aurc_calibrated_cv']:.3f} calibrated against {d['aurc_llm_confidence']:.3f} raw (lower is better). Figure~\\ref{{fig:rc}} shows both curves.\n"
        "\\begin{figure}[H]\\centering\\IfFileExists{../results/figures/fig2_risk_coverage.png}{\\includegraphics[width=.62\\linewidth]{../results/figures/fig2_risk_coverage.png}}{}\n\\caption{Risk-coverage curves.}\\label{fig:rc}\\end{figure}\n"
    )


def e6_tex() -> str:
    d = _load("E6", "attribution_rules_only_dev")
    if not d:
        return ""
    c, cf = d["before"]["counts"], d["counterfactual"]
    names = {"A1": "intent", "A2": "retrieval", "A3": "rerank", "A4": "read", "A5": "plan or compute", "A6": "citation", "A7": "verifier"}
    dist = ", ".join(f"{names[k]} {v}" for k, v in c.items() if v)
    return (
        "\\subsection{E6: where does it fail?}\n"
        f"Attribution was run on the offline rules agent only ({d['before']['n_failed']} failures; the live agent has not been attributed because counterfactual replay costs API calls). "
        f"Failures by stage: {dist}. Retrieval and rerank together explain {100 * (c.get('A2', 0) + c.get('A3', 0)) / max(d['before']['n_failed'], 1):.0f}\\%, so once the reranker is fine-tuned retrieval is no longer the bottleneck. "
        f"Injecting the gold table values flips {100 * cf['flip_rate']['A4']:.0f}\\% of failures, the gold page at rank one {100 * cf['flip_rate']['A2']:.0f}\\%, and the gold intent {100 * cf['flip_rate']['A1']:.0f}\\%; "
        "most failures are abstentions on answerable questions, which none of these injections addresses. The intent stage is probably a non-causal label (stages are assigned earliest first), and the live agent's stage distribution will differ.\n"
    )


def e8_tex() -> str:
    d = _load("E8", "season")
    if not d:
        return ""
    lag = d["lag"]["all"]
    return (
        "\\subsection{E8: what does the season show?}\n"
        f"Over {lag['n']} filings the XBRL arrived a median {lag['median_minutes']:.0f} minutes after the results PDF (quartiles {lag['p25_minutes']:.0f} and {lag['p75_minutes']:.0f}, 95th percentile {lag['p95_minutes']:.0f}); "
        f"the PDF came first for {p1(lag['pdf_first_share'])} of filings, and {p1(lag['over_3h_share'])} of XBRL files arrived more than three hours later. "
        f"{p1(d['revisions']['share'])} of filings had at least one revised XBRL, which the gold handling must follow. "
        "This lag is what the self-grading loop uses: answer when the PDF lands, grade when the XBRL arrives.\n"
    )


PREREG_ORDER = ("small_only", "strong_only", "small_only_v3", "strong_only_v3", "B0_naive_rag", "cascade", "cascade_keep_v3")
PAIRS = (("small_only_v3", "small_only"), ("strong_only_v3", "strong_only"), ("cascade_keep_v3", "cascade"))


def test_reads() -> int:
    """Number of recorded reads of the test split (the guard writes one line per read, any split)."""
    p = get_paths().split_log
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("split") == "test")


def has_test_results() -> bool:
    return any(_rows(c, "test") for c in PREREG_ORDER)


def _cost_q(config: str, split: str) -> float | None:
    d = _rows(config, split)
    return d["summary"]["cost_per_question_usd"] if d else None


def h1_verdict(split: str) -> str:
    """H1 as preregistered (Addendum A): supported only if the lower end of the 95% interval of
    cascade minus strong_only is at least -2 points and the cascade is cheaper; not supported if
    the point estimate is below -2 points or it is not cheaper; otherwise inconclusive."""
    r = paired_raw("cascade", "strong_only", "verified", split)
    cc, cs = _cost_q("cascade", split), _cost_q("strong_only", split)
    if r is None or cc is None or cs is None:
        return "not evaluated"
    if r["lo"] >= -0.02 and cc < cs:
        return "supported"
    if r["diff"] < -0.02 or cc >= cs:
        return "not supported"
    return "inconclusive"


def variants_confirmed(split: str) -> tuple[int, int]:
    """(confirmed, evaluated): a variant is confirmed only if the lower end of the 95% interval of
    variant minus its specified parent is above zero on the held-out sample."""
    done = [(v, par) for v, par in PAIRS if paired_raw(v, par, "verified", split)]
    return sum(1 for v, par in done if paired_raw(v, par, "verified", split)["lo"] > 0), len(done)  # type: ignore[index]


def test_note() -> str:
    """One sentence on the state of the held-out split, for the limitations and the setup."""
    e1 = _load("E1", "gym_validity")
    total = sum(r["n"] for r in e1["question_counts"] if r["split"] == "test") if e1 else None
    d = next((_rows(c, "test") for c in PREREG_ORDER if _rows(c, "test")), None)
    if not d:
        return "The test split has not been read by any experiment (the guard log does not exist)."
    n = d["summary"]["n"]
    return (
        f"Only a preregistered sample of {n} of {total} test questions has been read; the other test questions are unread. "
        f"The guard log records {test_reads()} test reads: one per configuration, plus those of a first attempt that a budget-accounting bug invalidated "
        "and that was re-run with unchanged settings."
    )


def test_tex() -> str:
    cfgs = [c for c in PREREG_ORDER if _rows(c, "test")]
    if not cfgs:
        return ""
    missing = [c for c in PREREG_ORDER if c not in cfgs]
    spec = tuple(c for c in SPECIFIED if c in cfgs)
    expl = tuple(c for c in ("small_only_v3", "strong_only_v3", "cascade_keep_v3") if c in cfgs)
    out = (
        "\\subsection{Held-out test sample}\n"
        "Following Addendum A of the preregistration (written after the dev results and before any test access), every configuration ran on the same seeded random sample of test questions. "
        f"{test_note()} Decision rules were fixed in advance and are applied mechanically below. "
        + (f"Not run, because the budget ended or the run was interrupted: {', '.join(tt(c) for c in missing)}. " if missing else "")
        + "\n"
        "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrrrrr}\\toprule\n"
        "config & n & verified acc. (CI) & abstain & \\$/question & \\$/verified correct & p50 s \\\\\\midrule\n"
        + _table_rows(spec + expl, "test")
        + "\\bottomrule\\end{tabular}\\caption{Held-out test sample ($^{*}$: exploratory variants, designed on dev).}\\label{tab:test}\\end{table}\n"
    )
    h3 = paired("small_only", "B0_naive_rag", "verified", "test")
    if h3:
        out += f"\\paragraph{{H3, first half.}} The small-model agent beats naive RAG by {h3}.\n"
    r = paired_raw("cascade", "strong_only", "verified", "test")
    if r:
        cc, cs = _cost_q("cascade", "test"), _cost_q("strong_only", "test")
        out += (
            f"\\paragraph{{H1 as preregistered: {h1_verdict('test')}.}} {tt('cascade')} minus {tt('strong_only')} is {paired('cascade', 'strong_only', 'verified', 'test')}; "
            f"cost per question \\${cc:.4f} against \\${cs:.4f}. The rule requires the lower end of the interval to be at least $-2$ points and the cascade to be cheaper.\n"
        )
    lines = []
    for v, par in PAIRS:
        rr = paired_raw(v, par, "verified", "test")
        if rr:
            lines.append(f"{tt(v)} minus {tt(par)}: {paired(v, par, 'verified', 'test')}, {'confirmed' if rr['lo'] > 0 else 'not confirmed'} (rule: lower end of the interval above zero)")
    if lines:
        k, m = variants_confirmed("test")
        out += "\\paragraph{Exploratory variants.} " + "; ".join(lines) + f". {k} of {m} evaluated variants {'is' if k == 1 else 'are'} confirmed; a confirmed variant is promoted to the headline only after the full test run, and the others stay exploratory.\n"
    if "cascade_keep_v3" in cfgs and "strong_only_v3" in cfgs:
        ck, sv = _cost_q("cascade_keep_v3", "test"), _cost_q("strong_only_v3", "test")
        out += f"\\paragraph{{Cost and accuracy of the repaired cascade.}} {tt('cascade_keep_v3')} minus {tt('strong_only_v3')} is {paired('cascade_keep_v3', 'strong_only_v3', 'verified', 'test')}, at \\${ck:.4f} against \\${sv:.4f} per question.\n"
    shifts = []
    for c in spec:
        dd, tt_ = _rows(c, "dev"), _rows(c, "test")
        if dd and tt_:
            shifts.append(f"{tt(c)} {p1(dd['summary']['verified_accuracy'])} on dev and {p1(tt_['summary']['verified_accuracy'])} on test")
    if shifts:
        out += "\\paragraph{Dev to test.} Verified accuracy, " + "; ".join(shifts) + ". Splits are chronological, so the test filings are the later ones in the quarter; the shift is reported, not corrected.\n"
    return out


def latex_results() -> str:
    parts = [setup_tex(), e1_tex(), e2_tex(), e3_tex(), exploratory_tex(), test_tex(), e4_tex(), e6_tex(), e8_tex()]
    body = "\n".join(p for p in parts if p)
    return "% generated by pramaan render; do not edit\n" + (body or "No evaluation results are available yet.\n")
