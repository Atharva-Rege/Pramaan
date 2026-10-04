"""Freeze the 60 dev questions the CI eval gate runs.

The spec caps a gate run at $0.50 and page reads dominate cost, so the gate is built from whole
filings: every question on a filing shares that filing's page reads, which makes the summed cost
of a filing's questions its true cost. Filings are taken cheapest per question first from the
evaluated dev sample (results/E3/small_only_dev.json), so the cost comes from a real run.
Cross-company ranking questions (T5) are left out because each one reads three filings.
scripts/gate_run.py then measures the cold-cache cost before the baseline is committed.
"""

import json
from pathlib import Path

SRC = Path("results/E3/small_only_dev.json")
TOTAL = 60

rows = [r for r in json.loads(SRC.read_text(encoding="utf-8"))["per_question"] if r["tier"] != "T5"]
by_filing: dict[str, list[dict]] = {}
for r in rows:
    by_filing.setdefault(r["filing"], []).append(r)
order = sorted(by_filing.items(), key=lambda kv: (sum(x["cost_usd"] for x in kv[1]) / len(kv[1]), kv[0]))
chosen: list[dict] = []
cost = 0.0
for _, qs in order:
    if len(chosen) >= TOTAL:
        break
    chosen += qs
    cost += sum(x["cost_usd"] for x in qs)
chosen = chosen[:TOTAL]
Path("ci").mkdir(exist_ok=True)
Path("ci/gate_qids.json").write_text(json.dumps(sorted(r["qid"] for r in chosen), indent=1), encoding="utf-8")
tiers = {t: sum(1 for r in chosen if r["tier"] == t) for t in ("T1", "T2", "T3", "T4", "T6")}
print(f"froze {len(chosen)} dev questions from {len({r['filing'] for r in chosen})} filings {tiers}; estimated cold cost ${cost:.3f}")
print("next: python scripts/gate_run.py --qids ci/gate_qids.json --out ci/baseline_dev60.json --cache <fresh path> --cap 0.50")
