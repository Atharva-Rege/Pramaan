import numpy as np

from pramaan.evals.stats import (
    bootstrap_mean_ci,
    expected_calibration_error,
    mcnemar,
    paired_bootstrap_diff,
    risk_coverage,
)
from pramaan.retrieval.metrics import evaluate_ranking, ndcg_at_k, reciprocal_rank


def test_bootstrap_ci_contains_mean():
    vals = [1.0] * 70 + [0.0] * 30
    clusters = [f"f{i // 2}" for i in range(100)]
    m, lo, hi = bootstrap_mean_ci(vals, clusters, n_boot=500)
    assert lo <= m <= hi and abs(m - 0.7) < 1e-9


def test_paired_diff_detects_gap():
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 50 + [0.0] * 50
    r = paired_bootstrap_diff(a, b, n_boot=500)
    assert r["diff"] > 0.25 and r["lo"] > 0


def test_mcnemar_symmetry():
    r = mcnemar([True] * 30 + [False] * 10, [False] * 30 + [True] * 10)
    assert r["a_only"] == 30 and r["b_only"] == 10 and r["p_value"] < 0.01


def test_risk_coverage_perfect_ranking():
    conf = list(np.linspace(1, 0, 10))
    ok = [True] * 8 + [False] * 2
    rc = risk_coverage(conf, ok)
    assert rc["selective_acc_at_90"] > 0.85 and rc["aurc"] < 0.1


def test_ece_perfectly_calibrated():
    assert expected_calibration_error([1.0] * 5, [True] * 5) == 0.0


def test_ranking_metrics():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert ndcg_at_k(["a", "b"], {"a"}) == 1.0
    m = evaluate_ranking(["x", "p"], {"p"})
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
