import argparse
import json
import sys
from pathlib import Path


def decide(baseline: dict, current: dict, max_new_failures: int = 2, max_drop_points: float = 3.0) -> dict:
    base_pass = {q["qid"] for q in baseline["per_question"] if q["verified"]}
    now = {q["qid"]: q for q in current["per_question"]}
    newly_failing = sorted(q for q in base_pass if q in now and not now[q]["verified"])
    b = baseline["summary"]["verified_accuracy"] * 100
    c = current["summary"]["verified_accuracy"] * 100
    reasons = []
    if len(newly_failing) > max_new_failures:
        reasons.append(f"{len(newly_failing)} previously passing questions now fail (limit {max_new_failures})")
    if b - c > max_drop_points:
        reasons.append(f"verified accuracy dropped {b - c:.1f} points (limit {max_drop_points})")
    return {"blocked": bool(reasons), "reasons": reasons, "newly_failing": newly_failing, "baseline_acc": b, "current_acc": c}


def comment(result: dict, diagnostics: dict[str, str]) -> str:
    head = "### Eval gate: BLOCKED" if result["blocked"] else "### Eval gate: passed"
    lines = [head, "", f"verified accuracy {result['baseline_acc']:.1f}% -> {result['current_acc']:.1f}%"]
    lines += [f"- {r}" for r in result["reasons"]]
    if result["newly_failing"]:
        lines += ["", "Newly failing questions:"]
        lines += [f"- `{q}`: {diagnostics.get(q, 'no diagnostic recorded')}" for q in result["newly_failing"]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="ci/baseline_dev60.json")
    ap.add_argument("--current", required=True)
    ap.add_argument("--comment-out", default="ci/gate_comment.md")
    a = ap.parse_args()
    base = json.loads(Path(a.baseline).read_text())
    cur = json.loads(Path(a.current).read_text())
    res = decide(base, cur)
    diag = {}
    p = Path(a.current).with_suffix(".diagnostics.json")
    if p.exists():
        diag = json.loads(p.read_text())
    Path(a.comment_out).write_text(comment(res, diag))
    print(json.dumps(res, indent=1))
    sys.exit(1 if res["blocked"] else 0)
