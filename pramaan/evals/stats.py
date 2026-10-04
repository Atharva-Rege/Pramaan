from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest


def _cluster_index(clusters: Sequence[str]) -> tuple[list[str], list[np.ndarray]]:
    keys: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        keys.setdefault(c, []).append(i)
    names = list(keys)
    return names, [np.asarray(keys[k]) for k in names]


def bootstrap_mean_ci(
    values: Sequence[float],
    clusters: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx_sets = [np.array([i]) for i in range(len(v))]
    else:
        _, idx_sets = _cluster_index(clusters)
    k = len(idx_sets)
    sums = np.asarray([v[i].sum() for i in idx_sets])
    cnts = np.asarray([len(i) for i in idx_sets], dtype=float)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        boots[b] = sums[pick].sum() / max(cnts[pick].sum(), 1.0)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 11,
) -> dict[str, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    d = x - y
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx_sets = [np.array([i]) for i in range(len(d))]
    else:
        _, idx_sets = _cluster_index(clusters)
    k = len(idx_sets)
    sums = np.asarray([d[i].sum() for i in idx_sets])
    cnts = np.asarray([len(i) for i in idx_sets], dtype=float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        boots[i] = sums[pick].sum() / max(cnts[pick].sum(), 1.0)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "p_two_sided": min(1.0, p)}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float]:
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01 = int(((~a) & b).sum())
    n10 = int((a & (~b)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only": float(n10), "b_only": float(n01), "p_value": p}


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> dict[str, object]:
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(conf)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan"), "selective_acc_at_90": float("nan")}
    order = np.argsort(-conf, kind="stable")
    ok_sorted = ok[order]
    cum_err = np.cumsum(~ok_sorted)
    cov = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    k90 = max(1, int(np.ceil(0.9 * n)))
    sel_acc90 = float(ok_sorted[:k90].mean())
    return {
        "coverage": [float(x) for x in cov[:: max(1, n // 100)]],
        "risk": [float(x) for x in risk[:: max(1, n // 100)]],
        "aurc": aurc,
        "selective_acc_at_90": sel_acc90,
    }


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(ece)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")
