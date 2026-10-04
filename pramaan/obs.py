from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pramaan.core.paths import get_paths

_lock = threading.Lock()
_tracer: Any = None
_init_done = False


def _init() -> None:
    global _tracer, _init_done
    if _init_done:
        return
    _init_done = True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "pramaan"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("pramaan")
    except Exception:
        _tracer = None


def new_trace_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    _init()
    record: dict[str, Any] = {"name": name, **{k: v for k, v in attrs.items() if v is not None}}
    t0 = time.perf_counter()
    cm = _tracer.start_as_current_span(name) if _tracer is not None else None
    otel_span = cm.__enter__() if cm is not None else None
    try:
        if otel_span is not None:
            for k, v in record.items():
                if isinstance(v, (str, int, float, bool)):
                    otel_span.set_attribute(k, v)
        yield record
    finally:
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if otel_span is not None:
            for k, v in record.items():
                if isinstance(v, (str, int, float, bool)):
                    otel_span.set_attribute(k, v)
        if cm is not None:
            cm.__exit__(None, None, None)
        sink = os.environ.get("PRAMAAN_TRACE_FILE")
        if sink:
            path = get_paths().logs / sink
            path.parent.mkdir(parents=True, exist_ok=True)
            with _lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
