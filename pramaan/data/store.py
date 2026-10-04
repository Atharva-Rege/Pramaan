from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb

from pramaan.core.paths import get_paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
  company_id TEXT PRIMARY KEY, symbol TEXT, isin TEXT, name TEXT, aliases TEXT[],
  sector TEXT, in_nifty200 BOOLEAN, scrip_code TEXT);
CREATE TABLE IF NOT EXISTS filings (
  filing_id TEXT PRIMARY KEY, company_id TEXT, exchange TEXT, period_end DATE,
  board_meeting_date DATE, pdf_published_at TIMESTAMP, xbrl_published_at TIMESTAMP,
  pdf_path TEXT, xbrl_paths TEXT[], split TEXT, role TEXT, status TEXT, pdf_source TEXT,
  n_pages INTEGER, revised BOOLEAN, scanned_pages INTEGER, meta TEXT);
CREATE TABLE IF NOT EXISTS pages (
  page_id TEXT PRIMARY KEY, filing_id TEXT, page_no INTEGER, text TEXT, words TEXT,
  text_source TEXT, is_results_page BOOLEAN, scope TEXT, png_path TEXT, page_type TEXT,
  qc_score DOUBLE, ocr_layer BOOLEAN, needs_ocr BOOLEAN, unit TEXT, width DOUBLE,
  height DOUBLE, columns TEXT, tbl TEXT);
CREATE TABLE IF NOT EXISTS xbrl_facts (
  filing_id TEXT, scope TEXT, line_item TEXT, period_end DATE, duration_months INTEGER,
  value_inr DOUBLE, decimals INTEGER, raw_element TEXT, xbrl_seq TEXT, revision INTEGER,
  is_latest BOOLEAN);
CREATE TABLE IF NOT EXISTS questions (
  qid TEXT PRIMARY KEY, split TEXT, tier TEXT, as_of TIMESTAMP, text TEXT, slots TEXT,
  gold TEXT, gold_pages TEXT[], human_checked BOOLEAN, template_id TEXT, gold_boxes TEXT,
  meta TEXT);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, config_hash TEXT, git_sha TEXT, split_hash TEXT,
  started_at TIMESTAMP, cost_usd DOUBLE, name TEXT, split TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS episodes (
  run_id TEXT, qid TEXT, trajectory TEXT, answer TEXT, verify TEXT, grade TEXT,
  diagnostics TEXT, cost_usd DOUBLE, latency_ms INTEGER);
CREATE TABLE IF NOT EXISTS memories (
  mem_id TEXT PRIMARY KEY, scope TEXT, key TEXT, lesson TEXT, source_episodes TEXT[],
  validation TEXT, uses INTEGER, wins INTEGER, losses INTEGER, status TEXT,
  created_at TIMESTAMP, embedding BLOB, variant TEXT);
CREATE TABLE IF NOT EXISTS adjudications (
  qid TEXT, verdict TEXT, note TEXT, reviewer TEXT, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS human_checks (
  qid TEXT, question_ok BOOLEAN, gold_ok BOOLEAN, note TEXT, reviewer TEXT, ts TIMESTAMP);
CREATE TABLE IF NOT EXISTS gold_evidence (
  filing_id TEXT, scope TEXT, line_item TEXT, period TEXT, page_id TEXT, box TEXT,
  unit_inr DOUBLE, approx BOOLEAN, ambiguous BOOLEAN, col_index INTEGER, row_label TEXT,
  cell_text TEXT, source_filing TEXT);
CREATE INDEX IF NOT EXISTS idx_evidence_filing ON gold_evidence(filing_id);
CREATE INDEX IF NOT EXISTS idx_pages_filing ON pages(filing_id);
CREATE INDEX IF NOT EXISTS idx_facts_filing ON xbrl_facts(filing_id);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
"""


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def jload(s: str | None, default: Any = None) -> Any:
    if s is None or s == "":
        return default
    return json.loads(s)


class Store:
    def __init__(self, path: str | Path | None = None, read_only: bool = False) -> None:
        p = Path(path) if path else get_paths().db_path
        if not read_only:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self.read_only = read_only
        self.con = duckdb.connect(str(p), read_only=read_only)
        self._lock = threading.RLock()
        if not read_only:
            self.con.execute(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def q(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            cur = self.con.execute(sql, params or [])
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def one(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def x(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self._lock:
            self.con.execute(sql, params or [])

    def many(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self.con.executemany(sql, list(rows))

    def upsert(self, table: str, key_cols: list[str], row: dict[str, Any]) -> None:
        with self._lock:
            where = " AND ".join(f"{k} = ?" for k in key_cols)
            self.con.execute(f"DELETE FROM {table} WHERE {where}", [row[k] for k in key_cols])
            cols = list(row)
            self.con.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [row[c] for c in cols],
            )

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0])
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        with self._lock:
            self.con.executemany(sql, [[r[c] for c in cols] for r in rows])

    def count(self, table: str, where: str = "", params: Sequence[Any] | None = None) -> int:
        row = self.one(f"SELECT count(*) AS n FROM {table} {('WHERE ' + where) if where else ''}", params)
        return int(row["n"]) if row else 0

    def filing(self, filing_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM filings WHERE filing_id = ?", [filing_id])

    def filings(self, where: str = "", params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        return self.q(f"SELECT * FROM filings {('WHERE ' + where) if where else ''} ORDER BY board_meeting_date, filing_id", params)

    def company(self, company_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM companies WHERE company_id = ?", [company_id])

    def pages_of(self, filing_id: str, columns: str = "*") -> list[dict[str, Any]]:
        return self.q(f"SELECT {columns} FROM pages WHERE filing_id = ? ORDER BY page_no", [filing_id])

    def page(self, page_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM pages WHERE page_id = ?", [page_id])

    def facts(self, filing_id: str, scope: str | None = None, latest_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM xbrl_facts WHERE filing_id = ?"
        params: list[Any] = [filing_id]
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if latest_only:
            sql += " AND is_latest"
        return self.q(sql, params)
