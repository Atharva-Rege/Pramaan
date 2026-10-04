from __future__ import annotations

from dataclasses import dataclass, field

from pramaan.core.schemas import Intent, ValueResult


@dataclass
class Injection:
    intent: Intent | None = None
    gold_pages: list[str] = field(default_factory=list)
    gold_values: dict[tuple[str, str, str], ValueResult] = field(default_factory=dict)
