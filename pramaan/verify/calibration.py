from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = [
    "n_soft",
    "v1_pass",
    "v2_pass",
    "v3_pass",
    "v4_pass",
    "v4_tested",
    "v5_pass",
    "margin",
    "rank",
    "strong",
    "escalated",
    "retries",
    "abstain",
    "llm_conf",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Calibrator:
    def __init__(self, mean: list[float], scale: list[float], coef: list[float], intercept: float, threshold: float = 0.5) -> None:
        self.mean = np.asarray(mean)
        self.scale = np.asarray(scale)
        self.coef = np.asarray(coef)
        self.intercept = float(intercept)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, feats: list[dict[str, float]], correct: list[bool], l2: float = 1.0, iters: int = 50) -> Calibrator:
        x = np.asarray([[f.get(k, 0.0) for k in FEATURES] for f in feats], dtype=float)
        y = np.asarray(correct, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        z = (x - mean) / scale
        w = np.zeros(z.shape[1])
        b = float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
        for _ in range(iters):
            p = _sigmoid(z @ w + b)
            g_w = z.T @ (p - y) / len(y) + l2 * w / len(y)
            g_b = float((p - y).mean())
            s = p * (1 - p) + 1e-6
            h = (z.T * s) @ z / len(y) + l2 * np.eye(z.shape[1]) / len(y)
            try:
                step = np.linalg.solve(h, g_w)
            except np.linalg.LinAlgError:
                step = g_w
            w -= step
            b -= g_b / float(s.mean())
        return cls(mean.tolist(), scale.tolist(), w.tolist(), b)

    def predict_proba(self, feats: dict[str, float]) -> float:
        x = np.asarray([feats.get(k, 0.0) for k in FEATURES], dtype=float)
        z = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return float(_sigmoid(np.asarray(z)))

    def choose_threshold(self, feats: list[dict[str, float]], correct: list[bool], target_selective_acc: float = 0.95) -> dict[str, float]:
        conf = np.asarray([self.predict_proba(f) for f in feats])
        ok = np.asarray(correct, dtype=bool)
        order = np.argsort(-conf)
        best = float(conf.max()) if len(conf) else 1.0
        cov_at = 0.0
        acc_at = float(ok.mean()) if len(ok) else 0.0
        run = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
        for k in range(len(order), 0, -1):
            if run[k - 1] >= target_selective_acc:
                best = float(conf[order[k - 1]])
                cov_at = k / len(order)
                acc_at = float(run[k - 1])
                break
        self.threshold = best
        return {"threshold": best, "coverage": cov_at, "selective_accuracy": acc_at}

    def to_json(self) -> dict[str, Any]:
        return {"features": FEATURES, "mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept, "threshold": self.threshold}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("threshold", 0.5))
