import argparse
import json

from pramaan.core.config import budget_config
from pramaan.data.store import Store
from pramaan.evals.cost import estimate

p = argparse.ArgumentParser()
p.add_argument("--config", action="append", required=True)
p.add_argument("--split", default="dev")
p.add_argument("--subset", type=int, default=None)
a = p.parse_args()
store = Store(read_only=True)
rows = [estimate(c, store, a.split, a.subset) for c in a.config]
total = sum(r["usd"] for r in rows)
print(json.dumps(rows, indent=1))
print(f"total estimated cost: ${total:.2f} (confirm threshold ${budget_config().confirm_above_usd:.2f}, api cap ${budget_config().api_usd_cap:.2f})")
