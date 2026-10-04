from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from pramaan.core.config import models_config
from pramaan.llm.types import LLMRequest, RawResponse

log = logging.getLogger("pramaan.llm")


class Backend(Protocol):
    def complete(self, req: LLMRequest) -> RawResponse: ...


def bare_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


class AnthropicBackend:
    def __init__(self, api_key: str | None = None, max_retries: int = 4, timeout: float = 600.0) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=max_retries, timeout=timeout) if key else anthropic.Anthropic(max_retries=max_retries, timeout=timeout)

    def complete(self, req: LLMRequest) -> RawResponse:
        model = bare_model(req.model)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "messages": req.messages,
        }
        if req.system:
            kwargs["system"] = req.system
        if req.cache_prefix:
            kwargs["cache_control"] = {"type": "ephemeral"}
        if req.effort:
            kwargs["output_config"] = {"effort": req.effort}
        if req.thinking:
            kwargs["thinking"] = req.thinking
        if req.temperature is not None and models_config().allows_sampling(model):
            kwargs["temperature"] = req.temperature
        msg = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        return RawResponse(
            text=text,
            model=msg.model,
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            stop_reason=msg.stop_reason,
            request_id=getattr(msg, "_request_id", None),
        )


class LiteLLMBackend:
    def complete(self, req: LLMRequest) -> RawResponse:
        import litellm

        msgs: list[dict[str, Any]] = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        msgs.extend(req.messages)
        kwargs: dict[str, Any] = {"model": req.model, "messages": msgs, "max_tokens": req.max_tokens, "num_retries": 4}
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        resp = litellm.completion(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return RawResponse(
            text=choice.message.content or "",
            model=getattr(resp, "model", req.model),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=getattr(choice, "finish_reason", None),
        )


class ScriptedBackend:
    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> RawResponse:
        self.calls.append(req)
        text = self.fn(req)
        return RawResponse(
            text=text,
            model=req.model,
            input_tokens=max(1, sum(len(str(m.get("content", ""))) for m in req.messages) // 4),
            output_tokens=max(1, len(text) // 4),
            stop_reason="end_turn",
        )
