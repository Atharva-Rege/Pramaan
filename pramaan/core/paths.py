from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SYNC_DIRS = ("data/raw", "data/pages", "data/db", "cache", "indexes")


@dataclass(frozen=True)
class Paths:
    home: Path
    work: Path
    config_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        home = Path(os.environ.get("PRAMAAN_HOME") or REPO_ROOT).expanduser().resolve()
        work_env = os.environ.get("PRAMAAN_WORK")
        work = Path(work_env).expanduser().resolve() if work_env else home
        cfg_env = os.environ.get("PRAMAAN_CONFIG_DIR")
        cfg = Path(cfg_env).expanduser().resolve() if cfg_env else REPO_ROOT / "configs"
        return cls(home=home, work=work, config_dir=cfg)

    @classmethod
    def for_root(cls, root: Path) -> Paths:
        root = Path(root).resolve()
        return cls(home=root, work=root, config_dir=REPO_ROOT / "configs")

    @property
    def raw(self) -> Path:
        return self.work / "data" / "raw"

    @property
    def pages(self) -> Path:
        return self.work / "data" / "pages"

    @property
    def db_dir(self) -> Path:
        return self.work / "data" / "db"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "pramaan.duckdb"

    @property
    def llm_cache(self) -> Path:
        return self.work / "cache" / "llm.sqlite"

    @property
    def indexes(self) -> Path:
        return self.work / "indexes"

    @property
    def manifest(self) -> Path:
        return self.work / "data" / "manifest.jsonl"

    @property
    def splits_path(self) -> Path:
        return self.home / "data" / "splits.json"

    @property
    def results(self) -> Path:
        return self.home / "results"

    @property
    def artifacts(self) -> Path:
        return self.home / "artifacts"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def reports(self) -> Path:
        return self.home / "reports"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def exports(self) -> Path:
        return self.home / "exports"

    @property
    def snapshots(self) -> Path:
        return self.home / "snapshots"

    @property
    def split_log(self) -> Path:
        return self.logs / "split_access.jsonl"

    def ensure(self) -> Paths:
        for p in (
            self.raw,
            self.pages,
            self.db_dir,
            self.llm_cache.parent,
            self.indexes,
            self.results,
            self.artifacts / "gepa",
            self.artifacts / "memory",
            self.artifacts / "reranker",
            self.artifacts / "ocr",
            self.logs,
            self.reports,
            self.models,
            self.exports,
            self.snapshots,
            self.splits_path.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


_PATHS: Paths | None = None


def get_paths() -> Paths:
    global _PATHS
    if _PATHS is None:
        _PATHS = Paths.from_env()
    return _PATHS


def set_paths(paths: Paths | None) -> None:
    global _PATHS
    _PATHS = paths
