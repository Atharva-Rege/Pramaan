from __future__ import annotations

import json
import sys
from pathlib import Path

from pramaan.core.ids import sha256_bytes
from pramaan.data.collector import build_client


def fetch_release(filings_jsonl: Path, out_dir: Path, roles: tuple[str, ...] = ("target",)) -> int:
    client = build_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in filings_jsonl.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["role"] not in roles or not rec.get("pdf_url"):
            continue
        target = out_dir / f"{rec['filing_id']}.pdf"
        if target.exists():
            continue
        res = client.get(rec["pdf_url"], accept_statuses=(200, 404))
        if res.status == 200:
            target.write_bytes(res.content)
            if rec.get("pdf_sha256") and sha256_bytes(res.content) != rec["pdf_sha256"]:
                print(f"hash mismatch for {rec['filing_id']} (the document may have been replaced)")
            n += 1
    return n


if __name__ == "__main__":
    a = sys.argv
    print(fetch_release(Path(a[1]), Path(a[a.index("--out") + 1]) if "--out" in a else Path("raw")))
