import argparse
import json

from pramaan.core.config import retrieval_config
from pramaan.core.paths import get_paths
from pramaan.retrieval.train_ce import train

p = argparse.ArgumentParser()
p.add_argument("--pairs", default=None)
p.add_argument("--groups", default=None)
p.add_argument("--base", default=None)
p.add_argument("--out", default=None)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--max-pairs", type=int, default=None)
a = p.parse_args()
cfg = retrieval_config().training
paths = get_paths().ensure()
res = train(
    a.pairs or paths.exports / "reranker_pairs.parquet",
    a.groups or paths.exports / "reranker_groups.parquet",
    a.base or cfg["base_model"],
    a.out or paths.artifacts / "reranker" / "cpu",
    epochs=a.epochs or int(cfg["epochs"]),
    batch_size=int(cfg["batch_size"]),
    lr=float(cfg["lr"]),
    max_length=int(cfg["max_length"]),
    device="cpu",
    max_train_pairs=a.max_pairs,
)
print(json.dumps({k: res[k] for k in ("base_holdout", "best_holdout", "model_path", "seconds")}, indent=1))
print("set retrieval.yaml r3.finetuned_path to:", res["model_path"])
