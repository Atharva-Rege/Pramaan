SHELL := /bin/bash
PY ?= python
CONFIG ?= cascade
SPLIT ?= dev
E ?= E3
WORKERS ?= 4

.PHONY: setup collect ingest index gym eval attrib exp render serve mcp snapshot check itest demo web gpu-bundle lint types test clean-cache

setup:
	$(PY) -m pip install -e ".[dev,serve,ml,ocr,graph,obs]"
	cd web && npm ci
	pramaan doctor

collect:
	pramaan collect

ingest:
	pramaan ingest --workers 8

index:
	pramaan index

gym:
	pramaan gym build

eval:
	pramaan eval --config $(CONFIG) --split $(SPLIT) --workers $(WORKERS)

attrib:
	pramaan attrib --run-id $(RUN)

exp:
	pramaan exp $(E)

render:
	$(PY) scripts/render_results.py

serve:
	pramaan serve

mcp:
	pramaan mcp

snapshot:
	pramaan snapshot

lint:
	ruff check .

types:
	mypy pramaan/core

test:
	pytest tests/unit tests/property tests/golden

check: lint types test

itest:
	pytest tests/integration

demo:
	PRAMAAN_DEMO_ONLY=1 pramaan serve --demo-only

web:
	cd web && npm ci && npm run build

gpu-bundle:
	$(PY) scripts/make_gpu_bundle.py

clean-cache:
	rm -rf cache/llm.sqlite*
