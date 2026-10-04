from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pramaan.core.config import gym_config
from pramaan.core.paths import REPO_ROOT, get_paths
from pramaan.core.splits import load_splits, purpose
from pramaan.data.collector import all_meta_paths, load_meta
from pramaan.data.store import Store, jload


def release_gym(out: Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(read_only=True)
    cfg = gym_config()
    n_q = 0
    with purpose("release"):
        for split in ("train", "dev", "test"):
            rows = store.q("SELECT * FROM questions WHERE split = ? ORDER BY as_of, qid", [split])
            with open(out / f"questions_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    meta = jload(r["meta"], {})
                    f.write(
                        json.dumps(
                            {
                                "qid": r["qid"],
                                "split": r["split"],
                                "tier": r["tier"],
                                "as_of": r["as_of"].isoformat() if r["as_of"] else None,
                                "text": r["text"],
                                "slots": jload(r["slots"]),
                                "gold": jload(r["gold"]),
                                "gold_pages": list(r["gold_pages"] or []),
                                "gold_boxes": jload(r["gold_boxes"], {}),
                                "filing_id": meta.get("filing_id"),
                                "template_id": r["template_id"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_q += 1
    paths = get_paths()
    n_f = 0
    with open(out / "filings.jsonl", "w", encoding="utf-8") as f:
        for mp in all_meta_paths(paths):
            m = load_meta(mp)
            if m is None:
                continue
            f.write(
                json.dumps(
                    {
                        "filing_id": m.filing_id,
                        "exchange": m.exchange,
                        "symbol": m.symbol,
                        "isin": m.isin,
                        "company": m.company,
                        "period_end": m.period_end,
                        "role": m.role,
                        "board_meeting_date": m.board_meeting_date,
                        "pdf_url": (m.pdf or {}).get("url"),
                        "pdf_sha256": (m.pdf or {}).get("sha256"),
                        "pdf_published_at": m.pdf_published_at,
                        "xbrl": [{"scope": x.scope, "url": x.url, "sha256": x.sha256, "created_at": x.created_at} for x in m.xbrl],
                        "xbrl_published_at": m.xbrl_published_at,
                    }
                )
                + "\n"
            )
            n_f += 1
    ev = store.q("SELECT filing_id, scope, line_item, period, page_id, box, unit_inr, col_index, row_label, cell_text FROM gold_evidence")
    with open(out / "gold_evidence.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")
    gen = out / "generator"
    gen.mkdir(exist_ok=True)
    for p in (REPO_ROOT / "pramaan" / "gym").glob("*.py"):
        shutil.copy2(p, gen / p.name)
    for name in ("gym.yaml", "line_items.yaml"):
        shutil.copy2(REPO_ROOT / "configs" / name, gen / name)
    splits = load_splits()
    card = f"""# Pramaan gym

Question set for Indian company results filings, graded against exchange XBRL.

- profile: {cfg.profile.name}
- target period end: {cfg.profile.target_period_end}
- splits hash: {splits.get('hash')}
- questions: {n_q}
- filings listed: {n_f}

Raw exchange PDFs and XBRL are not redistributed. `filings.jsonl` lists the original exchange URLs and SHA-256 hashes; fetch them with
`python -m pramaan.gym.fetch filings.jsonl --out raw/` (one request every two seconds per host). Gold answers come from the XBRL
Integrated Filing; `gold_pages` and `gold_boxes` point to the page and cell where the value is printed. Licence: Apache-2.0 for code and annotations.
Source documents belong to their filers and the exchanges; personal and educational use only.
"""
    (out / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return {"questions": n_q, "filings": n_f, "evidence_rows": len(ev), "path": str(out)}
