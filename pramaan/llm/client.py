from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pramaan.core.config import budget_config, models_config
from pramaan.core.ids import stable_hash
from pramaan.llm.backends import AnthropicBackend, Backend, LiteLLMBackend
from pramaan.llm.cache import BudgetGuard, LLMStore
from pramaan.llm.types import LLMRequest, LLMResult

_scopes: contextvars.ContextVar[tuple[CallScope, ...]] = contextvars.ContextVar("pramaan_llm_scopes", default=())
CACHE_VERSION = "v1"


@dataclass
class CallScope:
    results: list[LLMResult] = field(default_factory=list)
    run_id: str = ""  # spend inside this scope is booked to this run (per-run cap), unless a call names one

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def latency_ms(self) -> int:
        return sum(r.latency_ms for r in self.results)

    @property
    def calls(self) -> int:
        return len(self.results)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.results if r.cache_hit)

    @property
    def tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)


@contextmanager
def call_scope(run_id: str = "") -> Iterator[CallScope]:
    scope = CallScope(run_id=run_id)
    token = _scopes.set((*_scopes.get(), scope))
    try:
        yield scope
    finally:
        _scopes.reset(token)


def compute_cost(model: str, tin: int, tout: int, cread: int, cwrite: int, batch: bool = False) -> float:
    p = models_config().price(model)
    cost = (tin * p.input + tout * p.output + cread * p.cache_read + cwrite * p.cache_write) / 1e6
    return cost * (models_config().batch_discount if batch else 1.0)


class LLMClient:
    def __init__(
        self,
        store: LLMStore | None = None,
        backends: dict[str, Backend] | None = None,
        run_id: str = "adhoc",
        enforce_budget: bool = True,
        daily_cap: float | None = None,
    ) -> None:
        self.store = store or LLMStore()
        self._backends: dict[str, Backend] = dict(backends or {})
        self._lock = threading.Lock()
        self.run_id = run_id
        b = budget_config()
        self.guard = BudgetGuard(self.store, b.api_usd_cap, b.per_run_usd_cap, daily_cap) if enforce_budget else None
        self.models = models_config()

    def backend_for(self, model: str) -> Backend:
        prefix = model.split("/", 1)[0] if "/" in model else "anthropic"
        with self._lock:
            if prefix not in self._backends:
                if prefix == "anthropic":
                    self._backends[prefix] = AnthropicBackend()
                else:
                    self._backends[prefix] = LiteLLMBackend()
            return self._backends[prefix]

    def set_backend(self, prefix: str, backend: Backend) -> None:
        with self._lock:
            self._backends[prefix] = backend

    def role_model(self, role: str) -> str:
        return self.models.role(role).model

    def complete(
        self,
        role: str | None = None,
        *,
        model: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        tag: str = "",
        cache_prefix: bool = False,
        use_cache: bool = True,
        temperature: float | None = None,
        extra_key: str = "",
        run_id: str | None = None,
    ) -> LLMResult:
        effort = thinking = None
        if role is not None:
            rc = self.models.role(role)
            model = model or rc.model
            max_tokens = max_tokens or rc.max_tokens
            effort, thinking = rc.effort, rc.thinking
        if model is None:
            raise ValueError("either role or model is required")
        max_tokens = max_tokens or 4096
        req = LLMRequest(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            effort=effort,
            thinking=thinking,
            temperature=temperature,
            cache_prefix=cache_prefix,
        )
        key = stable_hash(
            {
                "v": CACHE_VERSION,
                "model": model,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "effort": effort,
                "thinking": thinking,
                "temperature": temperature,
                "extra": extra_key,
            },
            32,
        )
        rid = run_id or next((s.run_id for s in reversed(_scopes.get()) if s.run_id), "") or self.run_id
        if use_cache:
            hit = self.store.get(key)
            if hit is not None:
                res = LLMResult(
                    text=hit["text"],
                    model=hit["model"],
                    input_tokens=hit["input_tokens"],
                    output_tokens=hit["output_tokens"],
                    cache_read_tokens=hit.get("cache_read_tokens", 0),
                    cache_write_tokens=hit.get("cache_write_tokens", 0),
                    cost_usd=hit["cost_usd"],
                    latency_ms=hit["latency_ms"],
                    cache_hit=True,
                    stop_reason=hit.get("stop_reason"),
                    tag=tag,
                    provenance=hit.get("provenance", "api"),
                )
                self._account(res, rid, model, tag)
                return res
        if self.guard is not None:
            self.guard.check(rid)
        t0 = time.perf_counter()
        raw = self.backend_for(model).complete(req)
        latency = int((time.perf_counter() - t0) * 1000)
        cost = compute_cost(model, raw.input_tokens, raw.output_tokens, raw.cache_read_tokens, raw.cache_write_tokens)
        prefix = model.split("/", 1)[0] if "/" in model else "anthropic"
        provenance = "offline" if prefix == "offline" else "api"
        if prefix == "offline":
            cost = 0.0
        res = LLMResult(
            text=raw.text,
            model=raw.model,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cache_read_tokens=raw.cache_read_tokens,
            cache_write_tokens=raw.cache_write_tokens,
            cost_usd=cost,
            latency_ms=latency,
            cache_hit=False,
            stop_reason=raw.stop_reason,
            tag=tag,
            provenance=provenance,
        )
        if use_cache and raw.stop_reason != "refusal":
            self.store.put(
                key,
                model,
                tag,
                {
                    "text": res.text,
                    "model": res.model,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "cache_read_tokens": res.cache_read_tokens,
                    "cache_write_tokens": res.cache_write_tokens,
                    "cost_usd": res.cost_usd,
                    "latency_ms": res.latency_ms,
                    "stop_reason": res.stop_reason,
                    "provenance": provenance,
                },
            )
        self._account(res, rid, model, tag)
        return res

    def _account(self, res: LLMResult, run_id: str, model: str, tag: str) -> None:
        self.store.record_spend(run_id, model, tag, 0.0 if res.cache_hit else res.cost_usd, res.input_tokens, res.output_tokens, res.cache_hit)
        for s in _scopes.get():
            s.results.append(res)


_default: LLMClient | None = None
_default_lock = threading.Lock()


def get_client() -> LLMClient:
    global _default
    with _default_lock:
        if _default is None:
            _default = LLMClient()
        return _default


def set_client(client: LLMClient | None) -> None:
    global _default
    with _default_lock:
        _default = client
