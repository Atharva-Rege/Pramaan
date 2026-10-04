from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger("pramaan.http")

RETRY_STATUS = {403, 408, 425, 429, 500, 502, 503, 504}


class Blocked(RuntimeError):
    pass


@dataclass
class HttpResult:
    status: int
    content: bytes
    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        import json

        return json.loads(self.content.decode("utf-8", errors="replace"))


class PoliteClient:
    def __init__(
        self,
        header_sets: dict[str, dict[str, str]],
        min_interval: dict[str, float],
        timeout: float = 60.0,
        max_attempts: int = 5,
        backoff_base: float = 3.0,
        backoff_cap: float = 90.0,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.header_sets = header_sets
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._client = httpx.Client(
            http2=False,
            follow_redirects=True,
            timeout=timeout,
            transport=transport,
        )
        self.requests_made = 0

    def _host_key(self, url: str) -> str:
        host = urlparse(url).hostname or ""
        for key in self.header_sets:
            if key in host:
                return key
        return "default"

    def _pace(self, key: str) -> None:
        interval = self.min_interval.get(key, self.min_interval.get("default", 2.0))
        with self._lock:
            now = self._clock()
            last = self._last.get(key)
            if last is not None:
                wait = interval - (now - last)
                if wait > 0:
                    self._sleep(wait)
            self._last[key] = self._clock()

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        accept_statuses: tuple[int, ...] = (200,),
    ) -> HttpResult:
        key = self._host_key(url)
        headers = self.header_sets.get(key, {})
        last_err: str = ""
        for attempt in range(self.max_attempts):
            self._pace(key)
            try:
                resp = self._client.get(url, params=params, headers=headers)
                self.requests_made += 1
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code in accept_statuses:
                    return HttpResult(resp.status_code, resp.content, str(resp.url), dict(resp.headers))
                if resp.status_code not in RETRY_STATUS:
                    return HttpResult(resp.status_code, resp.content, str(resp.url), dict(resp.headers))
                last_err = f"HTTP {resp.status_code}"
            delay = min(self.backoff_cap, self.backoff_base * (2**attempt))
            delay *= 1.0 + random.uniform(0.0, 0.25)
            log.warning("retry %d for %s after %s (sleep %.1fs)", attempt + 1, url, last_err, delay)
            self._sleep(delay)
        if "403" in last_err:
            raise Blocked(f"blocked by {key}: {url}")
        raise RuntimeError(f"giving up on {url}: {last_err}")

    def close(self) -> None:
        self._client.close()
