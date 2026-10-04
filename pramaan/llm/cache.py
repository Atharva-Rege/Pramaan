from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from pramaan.core.errors import BudgetExceeded
from pramaan.core.paths import get_paths


class LLMStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else get_paths().llm_cache
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, tag TEXT, response TEXT, created REAL)"
            )
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, model TEXT, tag TEXT, cost REAL, input_tokens INTEGER, output_tokens INTEGER, cache_hit INTEGER)"
            )
            self._con.execute("CREATE INDEX IF NOT EXISTS idx_spend_run ON spend(run_id)")
            self._con.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._con.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, tag: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO cache (key, model, tag, response, created) VALUES (?, ?, ?, ?, ?)",
                (key, model, tag, json.dumps(response), time.time()),
            )
            self._con.commit()

    def record_spend(self, run_id: str, model: str, tag: str, cost: float, tin: int, tout: int, hit: bool) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO spend (ts, run_id, model, tag, cost, input_tokens, output_tokens, cache_hit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), run_id, model, tag, cost, tin, tout, int(hit)),
            )
            self._con.commit()

    def total_spend(self, run_id: str | None = None) -> float:
        with self._lock:
            if run_id is None:
                row = self._con.execute("SELECT COALESCE(SUM(cost), 0) FROM spend WHERE cache_hit = 0").fetchone()
            else:
                row = self._con.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM spend WHERE run_id = ? AND cache_hit = 0", (run_id,)
                ).fetchone()
        return float(row[0])

    def day_spend(self) -> float:
        since = time.time() - 86400
        with self._lock:
            row = self._con.execute(
                "SELECT COALESCE(SUM(cost), 0) FROM spend WHERE cache_hit = 0 AND ts > ?", (since,)
            ).fetchone()
        return float(row[0])

    def stats(self) -> dict[str, float]:
        with self._lock:
            n = self._con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            calls = self._con.execute("SELECT COUNT(*), COALESCE(SUM(cache_hit), 0) FROM spend").fetchone()
        return {"cached_entries": float(n), "calls": float(calls[0]), "hits": float(calls[1]), "spend_usd": self.total_spend()}

    def close(self) -> None:
        with self._lock:
            self._con.close()


class BudgetGuard:
    def __init__(self, store: LLMStore, total_cap: float, run_cap: float | None = None, daily_cap: float | None = None) -> None:
        self.store = store
        self.total_cap = total_cap
        self.run_cap = run_cap
        self.daily_cap = daily_cap

    def check(self, run_id: str) -> None:
        if self.store.total_spend() >= self.total_cap:
            raise BudgetExceeded(f"api_usd_cap {self.total_cap} reached")
        if self.run_cap is not None and self.store.total_spend(run_id) >= self.run_cap:
            raise BudgetExceeded(f"per_run_usd_cap {self.run_cap} reached for run {run_id}")
        if self.daily_cap is not None and self.store.day_spend() >= self.daily_cap:
            raise BudgetExceeded(f"daily cap {self.daily_cap} reached")
