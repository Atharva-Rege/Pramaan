from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pramaan.core.paths import REPO_ROOT, get_paths

BLUE, ORANGE, GREEN, GRAY, PURPLE, RED = "#2563eb", "#ea580c", "#16a34a", "#6b7280", "#7c3aed", "#dc2626"
MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PENDING = "pending (run the experiment, see Reproduce)"


def _load(exp: str, name: str) -> dict[str, Any] | None:
    p = get_paths().results / exp / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _all(exp: str, suffix: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted((get_paths().results / exp).glob(f"*_{suffix}.json")) if (get_paths().results / exp).exists() else []:
        if p.stem.startswith("summary"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "summary" not in d:  # other result files (audits) can share the folder; they are not config runs
            continue
        tag = d["meta"].get("tag")
        # A tagged run (for example a different reranker) is its own row, not a replacement.
        out[f"{d['meta']['config']} [{tag}]" if tag else d["meta"]["config"]] = d
    return out


def pct(x: float | None, d: int = 1) -> str:
    return PENDING if x is None else f"{100 * x:.{d}f}%"


def usd(x: float | None, d: int = 4) -> str:
    return "n/a" if x is None else f"${x:.{d}f}"


def best_pramaan(rows: dict[str, dict[str, Any]], exploratory: bool = False) -> str | None:
    """Best agent row by verified accuracy. By default only the specified configs count:
    no baselines, ablations, offline rules, tagged runs (smoke, reranker variants) or
    exploratory variants, so a post hoc variant can never become the headline."""
    cands = {
        k: v
        for k, v in rows.items()
        if not k.startswith(("B", "ablation")) and k != "rules_only" and " [" not in k and _exploratory(k) == exploratory
    }
    if not cands:
        return None
    return max(cands, key=lambda k: cands[k]["summary"]["verified_accuracy"])


def _exploratory(name: str) -> bool:
    """True for agent variants that switch on a post hoc flag (designed after reading dev failures)."""
    from pramaan.core.config import agent_config

    try:
        v = agent_config().variant(name.split(" [")[0])
    except Exception:
        return False
    return bool(v.relax_ratio_units or v.escalate_on_abstain or not v.exclude_failed_pages)


def leaderboard_md(split: str) -> str:
    rows = _all("E3", split)
    if not rows:
        return PENDING
    lines = ["| config | n | accuracy | verified accuracy | abstain | $/question | $/verified correct | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for name, d in sorted(rows.items(), key=lambda kv: -kv[1]["summary"]["verified_accuracy"]):
        s = d["summary"]
        lo, hi = s["verified_accuracy_ci"]
        lines.append(
            f"| {name}{' (exploratory)' if _exploratory(name) else ''} | {s['n']} | {pct(s['accuracy'])} | {pct(s['verified_accuracy'])} ({100 * lo:.1f} to {100 * hi:.1f}) | {pct(s['abstain_rate'])} | {usd(s['cost_per_question_usd'])} | "
            f"{usd(s['cost_per_verified_correct_usd'])} | {s['latency_ms_p50']:.0f} | {s['latency_ms_p95']:.0f} |"
        )
    if any(_exploratory(n) for n in rows):
        note = "(exploratory) marks variants whose flags were chosen after reading dev failures"
        if split == "test":
            note += "; one counts as confirmed only if the 95% interval of its paired gain over its specified parent is above zero (see the report). The preregistered configs are the rows without the mark."
        else:
            note += ": their dev numbers are optimistic and need confirming on unseen questions. The preregistered configs are the rows without the mark."
        lines += ["", note]
    sizes = sorted({d["summary"]["n"] for d in rows.values()})
    if len(sizes) > 1:
        lines += ["", f"Rows differ in question count (n = {', '.join(str(n) for n in sizes)}): they are different question sets, so compare only rows with the same n."]
    return "\n".join(lines)


def fig_pareto(out: Path) -> bool:
    """Accuracy versus cost for the rows that share the main question set. Markers are numbered
    and keyed beside the plot, so nothing depends on label placement. Filled markers are specified
    configs, hollow ones exploratory variants, orange the baseline; the dashed line is the
    non-dominated frontier. Tagged runs and offline rules (other question sets) are left out."""
    from collections import Counter

    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows = _all("E3", "dev") or _all("E3", "test")
    rows = {k: v for k, v in rows.items() if " [" not in k and k != "rules_only"}
    if not rows:
        return False
    n_main = Counter(v["summary"]["n"] for v in rows.values()).most_common(1)[0][0]
    pts = sorted(
        ((k, max(v["summary"]["cost_per_question_usd"], 1e-5), 100 * v["summary"]["verified_accuracy"]) for k, v in rows.items() if v["summary"]["n"] == n_main),
        key=lambda t: t[1],
    )
    fig, (ax, key) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    xs = [x for _, x, _ in pts]
    ax.set_xscale("log")
    ax.set_xlim(min(xs) * 0.6, max(xs) * 1.6)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.003, 0.01, 0.03])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    front = sorted((x, y) for _, x, y in pts if not any(x2 <= x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in pts))
    ax.plot([x for x, _ in front], [y for _, y in front], ls="--", lw=1, color=GRAY, zorder=2)
    key.axis("off")
    for i, (name, x, y) in enumerate(pts, 1):
        base, expl = name.startswith("B"), _exploratory(name)
        c = ORANGE if base else BLUE
        ax.scatter(x, y, s=210, facecolors="white" if expl else c, edgecolors=c, linewidths=1.8, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8, color=c if expl else "white", zorder=4, fontweight="bold")
        key.text(0.0, 1.0 - 0.085 * i, f"{i}  {name}{' *' if expl else ''}", fontsize=8.5, transform=key.transAxes, color="#111827")
        key.text(0.0, 1.0 - 0.085 * i - 0.04, f"     {y:.0f}%  at  ${x:.4f} per question", fontsize=7.5, transform=key.transAxes, color=GRAY)
    ax.set_xlabel("cost per question (USD, log scale)")
    ax.set_ylabel("verified accuracy (%)")
    ax.set_title(f"Accuracy versus cost ({n_main} dev questions)")
    key.text(0.0, 0.02, "* exploratory variant (hollow marker), designed\nafter reading dev failures; dashed line = frontier", fontsize=7.5, transform=key.transAxes, color=GRAY)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_risk_coverage(out: Path) -> bool:
    p = get_paths().results / "E4"
    files = sorted(p.glob("calibration_*.json")) if p.exists() else []
    if not files:
        return False
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(d["risk_coverage"]["coverage"], [100 * r for r in d["risk_coverage"]["risk"]], color=BLUE, lw=2, label=f"calibrated, 5-fold CV (AURC {d['aurc_calibrated_cv']:.3f})")
    if d.get("risk_coverage_llm"):
        rl = d["risk_coverage_llm"]
        ax.plot(rl["coverage"], [100 * r for r in rl["risk"]], color=GRAY, lw=2, label=f"model's own confidence (AURC {d['aurc_llm_confidence']:.3f})")
    ax.axhline(100 * (1 - d["base_accuracy"]), color=RED, ls=":", lw=1, label="error rate with no abstention")
    ax.set_xlabel("coverage")
    ax.set_ylabel("error rate among answered (%)")
    ax.set_title(f"Risk coverage, {d['meta']['config']} (n={d['n']})")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_learning(out: Path) -> bool:
    p = get_paths().results / "E5"
    files = sorted(p.glob("online_*.json")) if p.exists() else []
    if not files:
        return False
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    colors = {"none": GRAY, "unvalidated": ORANGE, "validated": BLUE}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm, v in d["arms"].items():
        ax.plot(range(1, len(v["curve"]) + 1), [100 * c for c in v["curve"]], color=colors.get(arm, GREEN), lw=2, label=f"{arm} memory")
    ax.set_xlabel("questions answered in publication order")
    ax.set_ylabel("cumulative accuracy (%)")
    ax.set_title("Learning during the season")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_attribution(out: Path) -> bool:
    p = get_paths().results / "E6"
    files = sorted(p.glob("attribution_*.json")) if p.exists() else []
    if not files:
        return False
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    stages = list(d["before"]["counts"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    xs = range(len(stages))
    ax.bar([x - 0.2 for x in xs], [d["before"]["share"][s] * 100 for s in stages], 0.4, color=ORANGE, label="before")
    if "after" in d:
        ax.bar([x + 0.2 for x in xs], [d["after"]["share"][s] * 100 for s in stages], 0.4, color=BLUE, label="after")
    ax.set_xticks(list(xs), stages)
    ax.set_ylabel("share of failures (%)")
    ax.set_title("Failure stage")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fr = d.get("counterfactual", {}).get("flip_rate", {})
    ax2 = axes[1]
    ax2.bar(list(fr), [100 * (v or 0) for v in fr.values()], color=GREEN)
    ax2.set_ylabel("fixed by gold injection (%)")
    ax2.set_title("Counterfactual replay")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_lag(out: Path) -> bool:
    d = _load("E8", "season")
    if not d or not d["lag"]["histogram"]:
        return False
    edges = d["lag"]["histogram_edges_minutes"]
    labels = []
    for a, b in itertools.pairwise(edges):
        labels.append(f"<{b:g}" if a < -1e8 else (f">{a:g}" if b > 1e8 else f"{a:g} to {b:g}"))
    fig, ax = plt.subplots(figsize=(7.6, 4))
    ax.bar(labels, d["lag"]["histogram"], color=BLUE)
    ax.set_xlabel("minutes from PDF to XBRL")
    ax.set_ylabel("filings")
    ax.set_title("How long the answer key lags the PDF")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def headline() -> dict[str, Any]:
    for split in ("test", "dev"):
        rows = _all("E3", split)
        if not rows:
            continue
        b0, b1 = rows.get("B0_naive_rag"), rows.get("B1_long_context")
        best = best_pramaan(rows)
        h: dict[str, Any] = {"split": split, "best": best}
        if best:
            s = rows[best]["summary"]
            h.update({"pramaan_verified": s["verified_accuracy"], "pramaan_cost_per_correct": s["cost_per_verified_correct_usd"], "n": s["n"]})
        xbest = best_pramaan(rows, exploratory=True)
        if xbest:
            xs = rows[xbest]["summary"]
            h["exploratory_best"] = {"config": xbest, "verified": xs["verified_accuracy"], "cost_per_correct": xs["cost_per_verified_correct_usd"], "n": xs["n"]}
        if b0:
            h["b0_verified"] = b0["summary"]["verified_accuracy"]
            h["b0_cost_per_correct"] = b0["summary"]["cost_per_verified_correct_usd"]
        if b1:
            h["b1_verified"] = b1["summary"]["verified_accuracy"]
            h["b1_cost_per_correct"] = b1["summary"]["cost_per_verified_correct_usd"]
        if h.get("pramaan_cost_per_correct") and h.get("b1_cost_per_correct"):
            h["cost_ratio_vs_b1"] = h["b1_cost_per_correct"] / h["pramaan_cost_per_correct"]
        return h
    return {}


def _e2_ndcg() -> tuple[float, float] | None:
    d = _load("E2", "retrieval_dev")
    if not d:
        return None
    rows = {r["stages"]: r for r in d["results"].get("scoped", [])}
    off, ft = rows.get("R1-R3 + off-the-shelf cross-encoder"), rows.get("R1-R3 + fine-tuned cross-encoder")
    return (off["ndcg@10"], ft["ndcg@10"]) if off and ft else None


def _has(exp: str, prefix: str) -> bool:
    p = get_paths().results / exp
    return p.exists() and any(p.glob(f"{prefix}*.json"))


def resume_bullets(h: dict[str, Any]) -> str:
    """Resume bullets built only from results/. A claim appears only when its evidence exists:
    no self-improvement line without E5, no long-context comparison without B1, no framework
    names for things the code does not use (the orchestrator is plain Python, DECISIONS #7)."""
    e1 = _load("E1", "gym_validity")
    n_q = sum(r["n"] for r in e1["question_counts"]) if e1 else None
    if not h or "pramaan_verified" not in h or n_q is None:
        return "Resume bullets are filled from results/ after the evaluation runs.\n"
    tech = "Python, DSPy, MCP, FastAPI, React, DuckDB" + (", GEPA" if _has("E5", "gepa") else "")
    out = [f"**Pramaan** | {tech}"]
    out.append(
        f"- Built a research agent over Indian listed-company filings that answers analyst questions with page-level citations and checks each answer (quote on the page, recomputation, unit and accounting identities) before showing it; graded automatically against exchange XBRL filings on an auto-generated {n_q}-question benchmark that can be regenerated every quarter from new filings."
    )
    best, b0 = h["pramaan_verified"], h.get("b0_verified")
    where = f"{h['n']} {h['split']} questions, one quarter"
    line = f"- Reached {100 * best:.0f}% verified accuracy" + (f" against {100 * b0:.0f}% for a naive RAG baseline" if b0 is not None else "") + f" on {where} ({h['best']})"
    cpc, b0c = h.get("pramaan_cost_per_correct"), h.get("b0_cost_per_correct")
    if cpc and b0c and cpc < b0c:
        line += f", at ${cpc:.3f} per verified-correct answer against ${b0c:.3f}"
    if h.get("cost_ratio_vs_b1") and h["cost_ratio_vs_b1"] > 1:
        line += f" and {h['cost_ratio_vs_b1']:.1f}x lower cost per correct answer than long-context prompting"
    out.append(line + ".")
    nd = _e2_ndcg()
    if nd:
        out.append(
            f"- Fine-tuned a cross-encoder reranker on relevance labels that come free from XBRL (no manual annotation), raising page-retrieval nDCG@10 from {nd[0]:.2f} to {nd[1]:.2f} over the off-the-shelf reranker."
        )
    missing = []
    if not _has("E5", "gepa") and not _has("E5", "online"):
        missing.append("self-improvement (E5: GEPA, validated memory)")
    if "b1_verified" not in h:
        missing.append("comparison with long-context prompting (B1)")
    if not _has("E7", "redteam"):
        missing.append("prompt-injection robustness (E7)")
    n_test = sum(r["n"] for r in e1["question_counts"] if r["split"] == "test") if e1 else 0
    if h["split"] != "test":
        missing.append("results on the test split")
    elif h["n"] < n_test:
        missing.append(f"the full test split (only {h['n']} of {n_test} test questions have been read)")
    note = f"\n\n<!-- Numbers come from results/ ({h['split']} split, one dry-run quarter); every bullet above is backed by a result file. -->\n"
    if missing:
        note += "\n**Not measured yet, so not claimed above:** " + "; ".join(missing) + ".\n"
    x = h.get("exploratory_best")
    if x:
        from pramaan.render_report import PAIRS, paired_raw

        status = ""
        parent = dict(PAIRS).get(x["config"])
        r = paired_raw(x["config"], parent, "verified", h["split"]) if parent and h["split"] == "test" else None
        if r:
            status = f" Against `{parent}` on the same test questions it is {100 * r['diff']:+.0f} points (95% CI {100 * r['lo']:+.0f} to {100 * r['hi']:+.0f}), " + (
                "confirmed by the preregistered rule, but promoted only after the full test run." if r["lo"] > 0 else "not confirmed by the preregistered rule (lower end of the interval above zero)."
            )
        note += (
            f"\n**Exploratory, do not use until confirmed on unseen questions:** `{x['config']}` reached {100 * x['verified']:.0f}% verified accuracy on {x['n']} {h['split']} questions. "
            f"That variant was designed after reading dev failures.{status}\n"
        )
    return "\n".join(out) + note


def linkedin_post(h: dict[str, Any]) -> str:
    """A draft post from the same measured facts. Links are placeholders for the owner to fill."""
    if not h or "pramaan_verified" not in h:
        return "The post is drafted from results/ after the evaluation runs.\n"
    b0 = h.get("b0_verified")
    nd = _e2_ndcg()
    lines = [
        "I built Pramaan: a research agent for Indian company filings that shows its proof.",
        "",
        "Every answer comes with the exact page and the printed number it was read from, is checked before it is shown, and is graded automatically when the exchange publishes the matching XBRL a little later. That makes the benchmark free, refreshable every quarter, and impossible to find in a model's training data.",
        "",
        f"What I measured so far ({h['n']} {h['split']} questions, one quarter, honest about the small sample):",
        f"- {100 * h['pramaan_verified']:.0f}% verified accuracy" + (f" against {100 * b0:.0f}% for naive RAG" if b0 is not None else ""),
    ]
    if nd:
        lines.append(f"- a cross-encoder fine-tuned on free XBRL labels lifts page retrieval nDCG@10 from {nd[0]:.2f} to {nd[1]:.2f}")
    lines += [
    ]
    e8 = _load("E8", "season")
    if e8:
        lag = e8["lag"]["all"]
        lines.append(f"- the exchange's XBRL arrives a median {lag['median_minutes']:.0f} minutes after the results PDF (the PDF came first in {100 * lag['pdf_first_share']:.0f}% of {lag['n']} filings), which is the gap the self-grading loop uses")
    lines += [
        "",
        "Limits I want to be upfront about: one quarter, one reporting format, scanned PDFs bound the accuracy, and "
        + (f"the numbers come from a {h.get('n')}-question sample of the held-out test split; the rest of it is still unread." if h.get("split") == "test" else "results are on a dev sample until I run the held-out test."),
        "",
        "Repo: [link]   Demo: [link]",
        "",
        "#LLM #RAG #AIagents #Finance #India",
    ]
    return "\n".join(lines) + "\n"


def latex_numbers(h: dict[str, Any]) -> str:
    def m(name: str, val: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"

    out = "% generated by pramaan render; do not edit\n"
    out += m("PramaanVerified", pct(h.get("pramaan_verified")).replace("%", "\\%"))
    out += m("BzeroVerified", pct(h.get("b0_verified")).replace("%", "\\%"))
    out += m("BoneVerified", pct(h.get("b1_verified")).replace("%", "\\%"))
    out += m("CostRatioVsLongContext", f"{h['cost_ratio_vs_b1']:.1f}" if h.get("cost_ratio_vs_b1") else "n/a")
    out += m("PramaanConfig", "\\texttt{" + str(h.get("best", "n/a")).replace("_", "\\_") + "}")
    out += m("NSample", str(h.get("n", "n/a")))
    out += m("EvalSplit", str(h.get("split", "n/a")))
    from pramaan.render_report import (
        PAIRS,
        h1_verdict,
        has_test_results,
        paired_raw,
        test_note,
        tt,
        variants_confirmed,
    )

    split = h.get("split", "dev")
    out += m("HOneVerdict", h1_verdict(split))
    out += m("TestStatusNote", test_note())
    if has_test_results():
        k, n = variants_confirmed("test")
        names = [v for v, par in PAIRS if (r := paired_raw(v, par, "verified", "test")) and r["lo"] > 0]
        note = f"{k} of {n} {'is' if k == 1 else 'are'} confirmed on the held-out test sample under the preregistered rule"
        out += m("VariantNote", note + (": " + ", ".join(tt(v) for v in names) if names else ""))
    else:
        out += m("VariantNote", "they still need confirming on held-out questions")
    return out


def latex_tables() -> str:
    out = "% generated by pramaan render; do not edit\n"
    rows = _all("E3", "test") or _all("E3", "dev")
    if rows:
        out += "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrrrr}\\toprule\nconfig & n & verified acc. & abstain & \\$/question & p50 ms \\\\\\midrule\n"
        for name, d in sorted(rows.items(), key=lambda kv: -kv[1]["summary"]["verified_accuracy"]):
            s = d["summary"]
            out += f"{name.replace('_', ' ')} & {s['n']} & {100 * s['verified_accuracy']:.1f}\\% & {100 * s['abstain_rate']:.1f}\\% & {s['cost_per_question_usd']:.4f} & {s['latency_ms_p50']:.0f} \\\\\n"
        out += "\\bottomrule\\end{tabular}\\caption{Agent versus baselines.}\\end{table}\n"
    e2 = _load("E2", "retrieval_dev")
    if e2:
        key = "scoped+scope" if "scoped+scope" in e2["results"] else next(iter(e2["results"]))
        out += "\\begin{table}[H]\\centering\\small\\begin{tabular}{lrrr}\\toprule\nstages & R@1 & R@5 & nDCG@10 \\\\\\midrule\n"
        for r in e2["results"][key]:
            out += f"{r['stages']} & {r['recall@1']:.3f} & {r['recall@5']:.3f} & {r['ndcg@10']:.3f} \\\\\n"
        out += f"\\bottomrule\\end{{tabular}}\\caption{{Retrieval stages ({key}).}}\\end{{table}}\n"
    return out


def render_readme(blocks: dict[str, str]) -> bool:
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if MARK_START not in text:
        return False
    body = "\n\n".join(f"### {title}\n\n{content}" for title, content in blocks.items())
    new = re.sub(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), MARK_START + "\n\n" + body + "\n\n" + MARK_END, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    return True


def render_all(echo: Callable[[str], None] = print) -> dict[str, Any]:
    paths = get_paths()
    fig_dir = paths.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = {
        "fig1_pareto": fig_pareto(fig_dir / "fig1_pareto.png"),
        "fig2_risk_coverage": fig_risk_coverage(fig_dir / "fig2_risk_coverage.png"),
        "fig3_learning": fig_learning(fig_dir / "fig3_learning.png"),
        "fig4_attribution": fig_attribution(fig_dir / "fig4_attribution.png"),
        "fig_lag": fig_lag(fig_dir / "fig_lag.png"),
    }
    h = headline()
    blocks = {
        "Held-out test sample": leaderboard_md("test"),
    }
    site = paths.results / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "leaderboard.json").write_text(json.dumps({s: [{k: v for k, v in d["summary"].items() if k not in ("risk_coverage",)} | {"config": c} for c, d in _all("E3", s).items()] for s in ("dev", "test")}, indent=1, default=str), encoding="utf-8")
    (site / "headline.json").write_text(json.dumps(h, indent=1, default=str), encoding="utf-8")
    (paths.results / "resume_bullets.md").write_text(resume_bullets(h), encoding="utf-8")
    (paths.results / "linkedin_post.md").write_text(linkedin_post(h), encoding="utf-8")
    gen = REPO_ROOT / "report" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "numbers.tex").write_text(latex_numbers(h), encoding="utf-8")
    (gen / "tables.tex").write_text(latex_tables(), encoding="utf-8")
    from pramaan.render_report import latex_results

    (gen / "results.tex").write_text(latex_results(), encoding="utf-8")
    results_md = "# Results\n\n" + "\n\n".join(f"## {t}\n\n{c}" for t, c in blocks.items()) + "\n"
    (REPO_ROOT / "docs" / "RESULTS.md").write_text(results_md, encoding="utf-8")
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "RESULTS.md").write_text(results_md, encoding="utf-8")
    readme_ok = render_readme(blocks)
    echo(f"figures: {[k for k, v in made.items() if v]}; headline: {h}; readme updated: {readme_ok}")
    return {"figures": made, "headline": h, "readme_updated": readme_ok}
