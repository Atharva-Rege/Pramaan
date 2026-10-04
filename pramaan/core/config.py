from __future__ import annotations

import os
import re
from datetime import date
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pramaan.core.errors import ConfigError
from pramaan.core.ids import stable_hash
from pramaan.core.paths import get_paths

CONFIG_NAMES = (
    "models",
    "budget",
    "line_items",
    "gym",
    "retrieval",
    "agent",
    "sources",
)


@lru_cache(maxsize=64)
def _load_yaml_cached(path: str, mtime: float) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def load_yaml(name: str) -> dict[str, Any]:
    path = get_paths().config_dir / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"missing config {path}")
    return _load_yaml_cached(str(path), path.stat().st_mtime)


def config_hash(names: tuple[str, ...] = CONFIG_NAMES) -> str:
    return stable_hash({n: load_yaml(n) for n in names if (get_paths().config_dir / f"{n}.yaml").exists()}, 12)


class RoleConfig(BaseModel):
    model: str
    max_tokens: int = 4096
    effort: str | None = None
    thinking: dict[str, Any] | None = None


class PriceConfig(BaseModel):
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class ModelsConfig(BaseModel):
    roles: dict[str, RoleConfig]
    embedding: dict[str, Any] = Field(default_factory=dict)
    prices_usd_per_mtok: dict[str, PriceConfig]
    no_sampling_params: list[str] = Field(default_factory=list)
    batch_discount: float = 0.5

    def role(self, name: str) -> RoleConfig:
        base = self.roles[name]
        override = os.environ.get(f"PRAMAAN_MODEL_{name.upper()}")
        if override:
            return base.model_copy(update={"model": override})
        return base

    def price(self, model: str) -> PriceConfig:
        bare = model.split("/", 1)[-1]
        for key in sorted(self.prices_usd_per_mtok, key=len, reverse=True):
            if key != "default" and bare.startswith(key):
                return self.prices_usd_per_mtok[key]
        return self.prices_usd_per_mtok["default"]

    def allows_sampling(self, model: str) -> bool:
        bare = model.split("/", 1)[-1]
        return not any(bare.startswith(m) for m in self.no_sampling_params)


class PerQuestionBudget(BaseModel):
    max_tool_calls: int = 12
    max_input_tokens_per_call: int = 60000
    max_output_tokens_per_call: int = 4000
    max_escalations: int = 1
    max_reflections: int = 2
    max_usd: float = 0.6


class BudgetConfig(BaseModel):
    api_usd_cap: float = 100.0
    per_run_usd_cap: float = 30.0
    confirm_above_usd: float = 2.0
    demo_daily_usd_cap: float = 5.0
    per_question: PerQuestionBudget = Field(default_factory=PerQuestionBudget)


class AgentVariant(BaseModel):
    tiers: list[str]
    reflect: bool = False
    verify: bool = True
    reader: str | None = None
    calculator: bool = True
    row_docs: bool = True
    # Exploratory, off by default so the specified configs stay as preregistered.
    relax_ratio_units: bool = False  # V3: a percent answer from one page does not need a unit header
    escalate_on_abstain: bool = False  # escalate when a tier answers "unavailable" with a soft fail
    exclude_failed_pages: bool = True  # False keeps pages from a failed attempt available to retries


class AgentConfig(BaseModel):
    orchestrator: str = "python"
    reader: str = "llm"
    reader_pages_per_get: int = 3
    search_k: int = 8
    rerank_cutoff: int = 5
    cascade: dict[str, Any] = Field(default_factory=dict)
    configs: dict[str, AgentVariant]
    abstention: dict[str, Any] = Field(default_factory=dict)

    def variant(self, name: str) -> AgentVariant:
        if name not in self.configs:
            raise ConfigError(f"unknown agent config {name}; have {sorted(self.configs)}")
        return self.configs[name]


class RetrievalConfig(BaseModel):
    r1: dict[str, Any]
    r2: dict[str, Any]
    r3: dict[str, Any]
    r4: dict[str, Any]
    units: dict[str, Any] = Field(default_factory=dict)
    dense: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=4)
def _models(key: str) -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml("models"))


def models_config() -> ModelsConfig:
    p = get_paths().config_dir / "models.yaml"
    return _models(f"{p}:{p.stat().st_mtime}")


def budget_config() -> BudgetConfig:
    return BudgetConfig.model_validate(load_yaml("budget"))


def agent_config() -> AgentConfig:
    return AgentConfig.model_validate(load_yaml("agent"))


def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig.model_validate(load_yaml("retrieval"))


class GymProfile(BaseModel):
    name: str
    target_period_end: date
    prior_year_period_end: date
    train_until: date
    dev_from: date
    dev_until: date
    test_from: date
    sizes: dict[str, int]


class GymConfig(BaseModel):
    profile: GymProfile
    seed: int
    tier_mix: dict[str, float]
    t1_items: list[str]
    t2_items: list[str]
    t5_metrics: list[str]
    ratios: dict[str, dict[str, Any]]
    tolerances: dict[str, float]
    evidence: dict[str, Any]
    human_check_sample: int = 100


def gym_config(profile: str | None = None) -> GymConfig:
    raw = dict(load_yaml("gym"))
    name = profile or os.environ.get("PRAMAAN_GYM_PROFILE") or raw["profile"]
    profiles = raw["profiles"]
    if name not in profiles:
        raise ConfigError(f"unknown gym profile {name}")
    prof = GymProfile(name=name, **profiles[name])
    return GymConfig(
        profile=prof,
        seed=raw["seed"],
        tier_mix=raw["tier_mix"],
        t1_items=raw["t1_items"],
        t2_items=raw["t2_items"],
        t5_metrics=raw["t5_metrics"],
        ratios=raw["ratios"],
        tolerances=raw["tolerances"],
        evidence=raw["evidence"],
        human_check_sample=raw.get("human_check_sample", 100),
    )


class LineItems:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.items: dict[str, dict[str, Any]] = raw["items"]
        self.scopes: dict[str, dict[str, Any]] = raw["scopes"]
        self.meta: dict[str, str] = raw["xbrl_meta"]
        self.identities: list[dict[str, Any]] = raw["identities"]
        self.tol: dict[str, float] = raw["identity_tolerance"]
        self.unit_factor: dict[str, float] = {k: float(v) for k, v in raw["units"].items()}
        self.unit_inr: dict[str, float] = {k: float(v) for k, v in raw["unit_to_inr"].items()}
        self._element_to_item: dict[str, str] = {}
        self._element_rank: dict[str, int] = {}
        for item, spec in self.items.items():
            for rank, el in enumerate(spec["xbrl"]):
                self._element_to_item.setdefault(el, item)
                self._element_rank.setdefault(el, rank)
        self._alias_index: list[tuple[str, str]] = []
        self._alias_exact: dict[str, str] = {}
        for item, spec in self.items.items():
            for a in list(spec.get("aliases", [])) + list(spec.get("names", [])):
                na = normalize_label(a)
                self._alias_index.append((na, item))
                self._alias_exact.setdefault(na, item)
        self._alias_by_len = sorted(self._alias_exact.items(), key=lambda kv: -len(kv[0]))

    @property
    def item_names(self) -> list[str]:
        return list(self.items)

    def kind(self, item: str) -> str:
        return str(self.items[item]["kind"])

    def element_item(self, element: str) -> str | None:
        return self._element_to_item.get(element)

    def element_rank(self, element: str) -> int:
        return self._element_rank.get(element, 99)

    def human_name(self, item: str) -> str:
        names = self.items[item].get("names") or [item.replace("_", " ")]
        return str(names[0])

    def scope_from_xbrl(self, value: str | None) -> str | None:
        if not value:
            return None
        v = value.strip().lower()
        for scope, spec in self.scopes.items():
            if any(v == x.lower() for x in spec["xbrl_values"]):
                return scope
        return None

    def match_label(self, label: str) -> str | None:
        return self.match_label_ex(label)[0]

    def match_label_ex(self, label: str) -> tuple[str | None, bool]:
        norm = normalize_label(label)
        if not norm:
            return None, False
        hit = self._alias_exact.get(norm)
        if hit:
            return hit, True
        trimmed = _TRAILING_NOISE.sub("", norm).strip()
        if trimmed != norm:
            hit = self._alias_exact.get(trimmed)
            if hit:
                return hit, True
        tokens = trimmed.split()
        for alias, item in self._alias_by_len:
            if trimmed.startswith(alias + " "):
                rest = trimmed[len(alias) :].split()
                if len(alias.split()) == 1 and not set(rest) <= _SINGLE_OK:
                    continue
                if len(rest) <= 3 and not (set(rest) & _NEGATORS):
                    return item, False
        from pramaan.core.textmatch import best_alias

        if len(tokens) >= 2:
            return best_alias(trimmed, self._alias_index, threshold=94.0), False
        return None, False


_NEGATORS = frozenset(
    {"attributable", "owners", "non", "non-controlling", "controlling", "discontinued", "per", "share", "margin",
     "segment", "excluding", "attributed", "minority", "earlier", "provision", "reserves", "ratio", "before", "after", "other"}
)
_SINGLE_OK = frozenset({"net", "of", "products", "product", "and", "services", "service", "from", "operations", "operation", "goods", "total", "income", "-", "&", "expense", "expenses", "(net)", "(in", "inr)", "(inr)", "(refer"})
_TRAILING_NOISE = re.compile(
    r"(?:\s+for\s+the\s+(?:period|quarter|year|half year|period/year|quarter/year|period / year))+\s*(?:\(net(?: of tax)?\))?\s*$"
    r"|\s*\(net(?: of tax)?\)\s*$|\s+from continuing operations\s*$|\s*\*+\s*$"
)


def normalize_label(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\(\s*\d+\s*[-+&]\s*\d+\s*\)", " ", t)
    t = re.sub(r"\(\s*\d+\s*\)", " ", t)
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\b(?:quarter|period|year)/(?:quarter|period|year)\b", "period", t)
    t = re.sub(r"\bfor the (?:quarter|year)\b", "for the period", t)
    t = re.sub(r"\([ivxlc]+(?:\s*[-+]\s*[ivxlc]+)*\)", " ", t)
    t = re.sub(r"\(refer[^)]*\)", " ", t)
    t = re.sub(r"\bnote[s]?\s*\d+\b", " ", t)
    t = re.sub(r"[^a-z0-9/()\- ]+", " ", t)
    t = re.sub(r"^\s*(?:[ivxlc]+|\d+)[.)]?\s+", "", t)
    t = re.sub(r"^\s*\([a-z0-9]{1,3}\.?\)\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@lru_cache(maxsize=4)
def _line_items(key: str) -> LineItems:
    return LineItems(load_yaml("line_items"))


def line_items() -> LineItems:
    p = get_paths().config_dir / "line_items.yaml"
    return _line_items(f"{p}:{p.stat().st_mtime}")


def sources_config() -> dict[str, Any]:
    return load_yaml("sources")
