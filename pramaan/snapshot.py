from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

from pramaan.core.paths import get_paths


def make_snapshot(name: str = "main", echo: Callable[[str], None] = print) -> Path:
    p = get_paths().ensure()
    out = p.snapshots / f"pramaan_snapshot_{name}.zip"
    roots = [p.db_path, p.indexes / name, p.pages, p.results, p.artifacts / "calibrator.json"]
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root in roots:
            if not root.exists():
                continue
            files = [root] if root.is_file() else [f for f in root.rglob("*") if f.is_file()]
            for f in files:
                base = p.home if str(f).startswith(str(p.home)) else p.work
                try:
                    z.write(f, f.relative_to(base))
                except ValueError:
                    z.write(f, f.name)
                n += 1
    echo(f"snapshot with {n} files: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _mini_db(src_db: Path, dst_db: Path, run_id: str | None, pages: set[str]) -> int:
    """A small DuckDB for a demo host: every filing and company (cheap rows, so the Season page
    has something to list), only the cited pages, and only the demo run with its questions.
    Every other table keeps its schema and no rows."""
    import duckdb

    dst_db.unlink(missing_ok=True)
    con = duckdb.connect(str(dst_db))
    con.execute(f"ATTACH '{src_db}' AS src (READ_ONLY)")
    quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in sorted(pages)) or "''"
    rid = (run_id or "").replace("'", "''")
    where = {
        "companies": "true",
        "filings": "true",
        "pages": f"page_id IN ({quoted})",
        "runs": f"run_id = '{rid}'",
        "episodes": f"run_id = '{rid}'",
        "questions": f"qid IN (SELECT qid FROM src.episodes WHERE run_id = '{rid}')",
    }
    n = 0
    for (table,) in con.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src' AND table_schema = 'main'").fetchall():
        con.execute(f"CREATE TABLE main.{table} AS SELECT * FROM src.main.{table} WHERE {where.get(table, 'false')}")
        n += 1
    con.execute("DETACH src")
    con.close()
    return n


def make_demo_snapshot(echo: Callable[[str], None] = print) -> Path:
    """A small bundle for a public demo host: results/, a mini database, and the page renders cited
    by the cached demo episodes. No index and no other filing pages, so it stays small and does not
    redistribute more of the exchange documents than the demo itself shows."""
    import json

    p = get_paths().ensure()
    out = p.snapshots / "pramaan_snapshot_demo.zip"
    demo = p.results / "demo" / "episodes.json"
    pages: set[str] = set()
    run_id = None
    if demo.exists():
        d = json.loads(demo.read_text(encoding="utf-8"))
        run_id = d.get("run_id")
        for e in d.get("episodes", []):
            for c in (e.get("answer") or {}).get("citations", []) or []:
                if c.get("page_id"):
                    pages.add(c["page_id"])
    files: list[tuple[Path, str]] = [(f, str(f.relative_to(p.home if str(f).startswith(str(p.home)) else p.work))) for f in p.results.rglob("*") if f.is_file()]
    for pid in sorted(pages):
        filing_id, _, n = pid.partition("#p")
        png = p.pages / filing_id / f"p{n}.png"
        if png.exists():
            files.append((png, str(png.relative_to(p.home if str(png).startswith(str(p.home)) else p.work))))
    mini = p.snapshots / "demo.duckdb"
    n_tables = _mini_db(p.db_path, mini, run_id, pages)
    files.append((mini, str(p.db_path.relative_to(p.home if str(p.db_path).startswith(str(p.home)) else p.work))))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f, arc in files:
            z.write(f, arc)
    mini.unlink(missing_ok=True)
    echo(f"demo snapshot with {len(files)} files ({len(pages)} cited pages, mini database with {n_tables} tables): {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
