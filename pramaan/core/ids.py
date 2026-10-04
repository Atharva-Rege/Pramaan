from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_hash(obj: Any, n: int = 16) -> str:
    return sha256_text(canonical_json(obj))[:n]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def company_id(isin: str | None, symbol: str) -> str:
    return (isin or symbol).strip().upper()


def filing_id(exchange: str, symbol: str, period_end: str, seq: str) -> str:
    return f"{exchange.lower()}-{slug(symbol)}-{period_end}-{slug(str(seq))}"


def page_id(fid: str, page_no: int) -> str:
    return f"{fid}#p{page_no}"


def split_page_id(pid: str) -> tuple[str, int]:
    fid, _, n = pid.rpartition("#p")
    return fid, int(n)


def question_id(split_hint: str, tier: str, parts: Mapping[str, Any]) -> str:
    return f"{tier.lower()}-{stable_hash([split_hint, tier, dict(parts)], 12)}"
