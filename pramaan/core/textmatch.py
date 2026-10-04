from __future__ import annotations

from rapidfuzz import fuzz


def best_alias(label: str, index: list[tuple[str, str]], threshold: float = 88.0) -> str | None:
    if not label:
        return None
    best_item: str | None = None
    best_score = 0.0
    for alias, item in index:
        if label == alias:
            return item
        if len(alias) < 4:
            continue
        score = fuzz.ratio(label, alias)
        if score > best_score:
            best_score = score
            best_item = item
    if best_score >= threshold:
        return best_item
    return None


def edit_distance_leq1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = sum(1 for x, y in zip(a, b, strict=True) if x != y)
        return diffs <= 1
    if la > lb:
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True
