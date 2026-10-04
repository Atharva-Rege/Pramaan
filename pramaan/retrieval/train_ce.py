from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from pramaan.retrieval.metrics import evaluate_ranking, summarize


def _device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_pairs(model: Any, tok: Any, pairs: list[tuple[str, str]], batch_size: int, max_length: int, device: str) -> list[float]:
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation="only_second", max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.extend(logits.view(-1).float().cpu().tolist())
    return out


def evaluate_groups(model: Any, tok: Any, groups: pd.DataFrame, batch_size: int, max_length: int, device: str) -> dict[str, float]:
    per = []
    for _, g in groups.iterrows():
        docs = list(g["positive"]) + list(g["negatives"])
        if not docs or not len(g["positive"]):
            continue
        scores = score_pairs(model, tok, [(g["query"], d) for d in docs], batch_size, max_length, device)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        ids = [str(i) for i in order]
        gold = {str(i) for i in range(len(g["positive"]))}
        per.append(evaluate_ranking(ids, gold))
    return summarize(per)


def train(
    pairs_path: str | Path,
    groups_path: str | Path,
    base_model: str,
    out_dir: str | Path,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 384,
    device: str | None = None,
    seed: int = 7,
    patience: int = 1,
    max_train_pairs: int | None = None,
    echo: Any = print,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = _device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pairs_path)
    groups = pd.read_parquet(groups_path)
    train_df = pairs[pairs["split"] == "train"]
    hold_groups = groups[groups["split"] == "holdout"]
    if max_train_pairs and len(train_df) > max_train_pairs:
        train_df = train_df.sample(max_train_pairs, random_state=seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1).to(dev)
    base_metrics = evaluate_groups(model, tok, hold_groups, batch_size * 2, max_length, dev) if len(hold_groups) else {}
    echo(f"base model holdout metrics: {base_metrics}")
    rows = list(zip(train_df["query"], train_df["text"], train_df["label"].astype(float), strict=True))
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(total * warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
    pos_weight = torch.tensor([max(1.0, (len(rows) - sum(r[2] for r in rows)) / max(1.0, sum(r[2] for r in rows)) / 2)]).to(dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    t0 = time.time()
    bad = 0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(rows)
        run_loss = 0.0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            enc = tok([c[0] for c in chunk], [c[1] for c in chunk], padding=True, truncation="only_second", max_length=max_length, return_tensors="pt").to(dev)
            labels = torch.tensor([c[2] for c in chunk], dtype=torch.float32, device=dev)
            logits = model(**enc).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run_loss += float(loss)
            if (i // batch_size) % 50 == 0:
                echo(f"epoch {epoch + 1} step {i // batch_size}/{steps_per_epoch} loss {float(loss):.4f}")
        metrics = evaluate_groups(model, tok, hold_groups, batch_size * 2, max_length, dev) if len(hold_groups) else {"ndcg@10": 0.0}
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, steps_per_epoch), **metrics})
        echo(f"epoch {epoch + 1}: {metrics}")
        if metrics.get("ndcg@10", 0.0) > best:
            best, best_epoch, bad = metrics.get("ndcg@10", 0.0), epoch + 1, 0
            if (out / "model").exists():
                shutil.rmtree(out / "model")
            model.save_pretrained(out / "model")
            tok.save_pretrained(out / "model")
        else:
            bad += 1
            if bad > patience:
                break
    final = {
        "base_model": base_model,
        "device": dev,
        "train_pairs": len(rows),
        "holdout_groups": len(hold_groups),
        "base_holdout": base_metrics,
        "best_epoch": best_epoch,
        "best_holdout": next((h for h in history if h["epoch"] == best_epoch), {}),
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "model_path": str(out / "model"),
    }
    (out / "metrics.json").write_text(json.dumps(final, indent=1), encoding="utf-8")
    return final
