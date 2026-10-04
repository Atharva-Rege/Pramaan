from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    model: str
    system: str | None
    messages: list[dict[str, Any]]
    max_tokens: int
    effort: str | None = None
    thinking: dict[str, Any] | None = None
    temperature: float | None = None
    cache_prefix: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    request_id: str | None = None


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool
    stop_reason: str | None = None
    tag: str = ""
    provenance: str = "api"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
