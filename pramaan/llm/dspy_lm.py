from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import dspy
from dspy.clients.base_lm import BaseLM

from pramaan.llm.client import LLMClient, get_client


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


class PramaanLM(BaseLM):
    def __init__(
        self,
        role: str | None = None,
        model: str | None = None,
        client: LLMClient | None = None,
        tag: str = "dspy",
        max_tokens: int | None = None,
    ) -> None:
        self.role = role
        self.tag = tag
        self._client = client
        self._model_override = model
        super().__init__(model=f"pramaan/{role or model}", model_type="chat", cache=False, max_tokens=max_tokens)

    @property
    def client(self) -> LLMClient:
        return self._client or get_client()

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        msgs = messages or [{"role": "user", "content": prompt or ""}]
        system_parts = [_text_of(m["content"]) for m in msgs if m["role"] == "system"]
        chat = [{"role": m["role"], "content": _text_of(m["content"])} for m in msgs if m["role"] != "system"]
        if not chat:
            chat = [{"role": "user", "content": ""}]
        max_tokens = kwargs.get("max_tokens") or self.kwargs.get("max_tokens")
        res = self.client.complete(
            self.role,
            model=self._model_override,
            system="\n\n".join(system_parts) or None,
            messages=chat,
            max_tokens=max_tokens,
            tag=self.tag,
        )
        message = SimpleNamespace(content=res.text, tool_calls=None, reasoning_content=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0, text=res.text)
        usage = {
            "prompt_tokens": res.input_tokens + res.cache_read_tokens + res.cache_write_tokens,
            "completion_tokens": res.output_tokens,
            "total_tokens": res.total_tokens,
        }
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=res.model,
            _hidden_params={"response_cost": res.cost_usd},
            cache_hit=res.cache_hit,
        )


def make_lm(role: str, client: LLMClient | None = None, tag: str = "dspy") -> PramaanLM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PramaanLM(role=role, client=client, tag=tag)


def configure_dspy(role: str = "small", client: LLMClient | None = None) -> PramaanLM:
    lm = make_lm(role, client)
    dspy.configure(lm=lm)
    return lm
