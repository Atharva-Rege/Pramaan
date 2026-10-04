"""Audit of abstentions on answerable questions: how often was the draft correct and thrown away?

Reads stored episodes (no API calls) and writes results/audit/verifier_audit_dev.json. For every
abstained answerable non-ranking question it grades the tier's draft against gold and sorts it
into: correct draft rejected by a check (verifier false reject), wrong draft rightly refused, or
no numeric draft (the model itself found nothing). For V3 false rejects it records the answer
unit and the number of cited pages, the shape the opt-in relax_ratio_units rule is built on.
"""

import argparse
import json
from collections import Counter

from pramaan.core.schemas import Answer
from pramaan.data.store import Store, jload
from pramaan.evals.harness import run_meta, write_result
from pramaan.gym.grade import grade, question_from_row

p = argparse.ArgumentParser()
p.add_argument("--configs", default="small_only,strong_only")
p.add_argument("--split", default="dev")
a = p.parse_args()

store = Store(read_only=True)
out: dict[str, object] = {}
for cfg in a.configs.split(","):
    with open(f"results/E3/{cfg}_{a.split}.json", encoding="utf-8") as fh:
        rid = json.load(fh)["run_id"]
    rows = store.q("SELECT e.qid, e.verify, e.diagnostics, q.* FROM episodes e JOIN questions q USING (qid) WHERE e.run_id = ?", [rid])
    n = len(rows)
    rejected: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    no_draft = n_abs = 0
    v3_shapes: list[dict[str, object]] = []
    for r in rows:
        dg = jload(r["diagnostics"], {})
        if not dg.get("abstained") or r["tier"] in ("T5", "T6"):
            continue
        n_abs += 1
        vr = jload(r["verify"], None) or {}
        hard = sorted({c["id"] for c in vr.get("checks", []) if c.get("status") == "hard_fail"})
        label = ",".join(hard) or "soft_fail"
        draft = next(iter((dg.get("drafts") or {}).values()), None)
        if not draft or draft.get("answer_type") != "number":
            no_draft += 1
            continue
        ok = grade(store, question_from_row(r), Answer.model_validate(draft), None).correct
        (rejected if ok else refused)[label] += 1
        if ok and "V3" in hard:
            pages = {c.get("page_id") for c in draft.get("citations", [])}
            v3_shapes.append({"tier": r["tier"], "unit": draft.get("unit"), "n_inputs": len(draft.get("inputs") or []), "n_pages": len(pages)})
    out[cfg] = {
        "n_questions": n,
        "abstained_answerable_nonranking": n_abs,
        "correct_draft_rejected": dict(rejected),
        "correct_draft_rejected_total": sum(rejected.values()),
        "wrong_draft_refused": dict(refused),
        "no_numeric_draft": no_draft,
        "v3_false_reject_shapes": v3_shapes,
    }
write_result("audit", f"verifier_audit_{a.split}", {"meta": run_meta("verifier_audit", a.split, "api"), "audit": out})
for cfg, r in out.items():
    print(cfg, {k: v for k, v in r.items() if k != "v3_false_reject_shapes"}, "| V3 shapes:", Counter((s["unit"], s["n_inputs"], s["n_pages"]) for s in r["v3_false_reject_shapes"]))  # type: ignore[attr-defined]
