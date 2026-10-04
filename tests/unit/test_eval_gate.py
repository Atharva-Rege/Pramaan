import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_gate", Path(__file__).resolve().parents[2] / "scripts" / "eval_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def run(passing, n=60):
    return {"summary": {"verified_accuracy": sum(passing) / n}, "per_question": [{"qid": f"q{i}", "verified": bool(passing[i])} for i in range(n)]}


def test_gate_blocks_on_regressions():
    base = run([1] * 50 + [0] * 10)
    cur = run([1] * 46 + [0] * 14)
    r = gate.decide(base, cur)
    assert r["blocked"] and len(r["newly_failing"]) == 4


def test_gate_allows_small_noise():
    base = run([1] * 50 + [0] * 10)
    cur = run([1] * 49 + [0] * 11)
    assert not gate.decide(base, cur)["blocked"]


def test_gate_blocks_on_accuracy_drop_even_with_few_flips():
    base = {"summary": {"verified_accuracy": 0.90}, "per_question": [{"qid": "a", "verified": True}]}
    cur = {"summary": {"verified_accuracy": 0.80}, "per_question": [{"qid": "a", "verified": True}]}
    assert gate.decide(base, cur)["blocked"]


def test_comment_lists_diagnostics():
    res = {"blocked": True, "reasons": ["x"], "newly_failing": ["q1"], "baseline_acc": 90.0, "current_acc": 80.0}
    assert "wrong unit" in gate.comment(res, {"q1": "wrong unit"})
