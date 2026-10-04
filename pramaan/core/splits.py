from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from typing import Any

from pramaan.core.config import GymProfile
from pramaan.core.errors import SplitAccessError
from pramaan.core.ids import stable_hash
from pramaan.core.paths import get_paths
from pramaan.core.timeutil import iso, utcnow

OPTIMIZING = frozenset(
    {
        "gepa",
        "memory",
        "reranker_export",
        "reranker_train",
        "prompt_edit",
        "calibration",
        "diagnose",
        "dev_eval",
        "redteam",
        "gym_build",
        "ablation",
    }
)
TEST_ALLOWED = frozenset({"final_eval", "online_season", "render", "release", "human_review", "split_stats"})

_purpose: ContextVar[str] = ContextVar("pramaan_purpose", default="unspecified")
_lock = threading.Lock()


@contextmanager
def purpose(name: str) -> Iterator[None]:
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> str:
    return _purpose.get()


def assign_split(board_meeting: date | None, profile: GymProfile) -> str | None:
    if board_meeting is None:
        return None
    if board_meeting <= profile.train_until:
        return "train"
    if profile.dev_from <= board_meeting <= profile.dev_until:
        return "dev"
    if board_meeting >= profile.test_from:
        return "test"
    return None


def log_split_access(split: str, what: str, allowed: bool) -> None:
    frame = inspect.stack()[2] if len(inspect.stack()) > 2 else None
    caller = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}" if frame else ""
    rec = {
        "ts": iso(utcnow()),
        "split": split,
        "what": what,
        "purpose": current_purpose(),
        "allowed": allowed,
        "caller": caller,
    }
    path = get_paths().split_log
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def check_split_access(split: str, what: str) -> None:
    if split != "test":
        return
    allowed = current_purpose() in TEST_ALLOWED
    log_split_access(split, what, allowed)
    if not allowed:
        raise SplitAccessError(
            f"test split access denied for purpose '{current_purpose()}' ({what}); "
            f"wrap final evaluation code in pramaan.core.splits.purpose('final_eval')"
        )


def freeze_splits(
    assignments: dict[str, str], profile: GymProfile, force: bool = False
) -> dict[str, Any]:
    path = get_paths().splits_path
    if path.exists() and not force:
        raise SplitAccessError(f"{path} already frozen; use --force to overwrite")
    body = {
        "profile": profile.name,
        "windows": {
            "train_until": profile.train_until.isoformat(),
            "dev_from": profile.dev_from.isoformat(),
            "dev_until": profile.dev_until.isoformat(),
            "test_from": profile.test_from.isoformat(),
        },
        "assignments": dict(sorted(assignments.items())),
    }
    body["hash"] = stable_hash(body["assignments"], 16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


def load_splits() -> dict[str, Any]:
    path = get_paths().splits_path
    if not path.exists():
        return {"profile": None, "assignments": {}, "hash": "unfrozen"}
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def splits_hash() -> str:
    return str(load_splits().get("hash", "unfrozen"))


def first_test_access() -> str | None:
    path = get_paths().split_log
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == "test":
                return str(rec["ts"])
    return None
