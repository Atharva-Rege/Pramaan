from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from pramaan.core.config import gym_config
from pramaan.core.ids import page_id as make_page_id
from pramaan.core.paths import Paths, get_paths
from pramaan.core.splits import assign_split, freeze_splits, load_splits
from pramaan.core.timeutil import parse_date_flexible, parse_iso, utc_to_ist
from pramaan.data.collector import FilingMeta, all_meta_paths, load_meta
from pramaan.data.layout import Word, plain_text, words_to_layout_text
from pramaan.data.ocr import available_backends, ocr_image
from pramaan.data.pages import PageInfo, classify_page
from pramaan.data.pdf import PageData, extract_pdf, render_page_image, render_pages_png
from pramaan.data.store import Store, jdump, jload
from pramaan.data.tables import ParsedTable, infer_columns, parse_table
from pramaan.data.textqc import assess_text
from pramaan.data.xbrl import XbrlError, has_core_items, parse_xbrl

log = logging.getLogger("pramaan.ingest")

COMPANY_SUFFIX = re.compile(r"\b(limited|ltd\.?|private|pvt\.?|\(india\)|india)\b", re.I)


def company_aliases(name: str, symbol: str) -> list[str]:
    base = re.sub(r"\s+", " ", COMPANY_SUFFIX.sub("", name)).strip(" .,-")
    out = {name.strip(), base, symbol, base.title()}
    if "&" in base:
        out.add(base.replace("&", "and"))
    if " and " in base.lower():
        out.add(re.sub(r"(?i)\band\b", "&", base))
    words = [w for w in re.split(r"[\s&]+", base) if w and w.lower() not in {"and", "the", "of"}]
    if len(words) >= 2:
        out.add("".join(w[0].upper() for w in words))
    return sorted(a for a in out if a)


def table_to_json(table: ParsedTable) -> dict[str, Any]:
    return {
        "unit": table.unit,
        "unit_text": table.unit_text,
        "title": table.title,
        "columns": [
            {"index": c.index, "x": round(c.x, 1), "period": str(c.period) if c.period else None, "scope": c.scope, "inferred": c.inferred}
            for c in table.columns
        ],
        "rows": [
            {
                "label": r.label,
                "item": r.item,
                "exact": r.exact,
                "section": r.section,
                "values": r.values,
                "boxes": [list(map(lambda v: round(v, 1), b)) if b else None for b in r.cell_boxes],
                "box": [round(v, 1) for v in r.box],
            }
            for r in table.rows
        ],
    }


def _page_state(pd: PageData, prev_scope: str | None, prev_cols: list[Any] | None, period_end: date | None = None) -> tuple[ParsedTable, PageInfo]:
    table = parse_table(pd.words, pd.width)
    info = classify_page(pd.text, table, prev_scope, pd.page_no)
    if info.is_results and period_end is not None:
        infer_columns(table, period_end)
    if info.is_results and prev_cols and not any(c.period for c in table.columns) and len(table.columns) == len(prev_cols):
        for c, p in zip(table.columns, prev_cols, strict=True):
            c.period, c.scope = p.period, p.scope
        if not table.unit:
            pass
    return table, info


def _better(new_qc: Any, old_qc: Any) -> bool:
    if old_qc.n_chars < 40 and new_qc.n_chars >= 40:
        return True
    return bool(new_qc.score > old_qc.score + 0.02 and new_qc.digit_clean_rate >= old_qc.digit_clean_rate - 0.02)


def process_filing_pdf(job: dict[str, Any]) -> dict[str, Any]:
    pdf_path = Path(job["pdf_path"])
    fid: str = job["filing_id"]
    png_dir = Path(job["png_dir"])
    backends: list[str] = job.get("ocr_backends", [])
    ocr_cap: int = job.get("ocr_cap", 12)
    device: str | None = job.get("device")
    pages = extract_pdf(pdf_path)
    pe: date | None = parse_date_flexible(job.get("period_end"))
    states: list[tuple[PageData, ParsedTable, PageInfo]] = []
    prev_scope: str | None = None
    prev_cols: list[Any] | None = None
    for pd in pages:
        table, info = _page_state(pd, prev_scope, prev_cols, pe)
        states.append((pd, table, info))
        if info.is_results:
            prev_scope = info.scope if info.scope not in (None, "both") else prev_scope
            prev_cols = table.columns if any(c.period for c in table.columns) else prev_cols

    ocr_done = 0
    if backends:
        results_found = any(s[2].is_results for s in states)
        order = sorted(range(len(states)), key=lambda i: (not states[i][0].qc.n_chars < 40, i))
        for i in order:
            pd, table, info = states[i]
            wants = False
            if pd.qc.n_chars < 40 and not results_found and pd.image_share >= 0.5:
                wants = True
            elif pd.ocr_layer and info.is_results and pd.needs_ocr:
                wants = True
            if not wants or ocr_done >= ocr_cap:
                continue
            try:
                img = render_page_image(pdf_path, pd.page_no, 200)
                words: list[Word] = []
                src = backends[0]
                for b in backends:
                    words = ocr_image(img, pd.width, pd.height, b, device)
                    src = b
                    if words:
                        break
                ocr_done += 1
            except Exception as exc:
                log.warning("ocr failed %s p%d: %s", fid, pd.page_no, exc)
                continue
            plain = plain_text(words)
            qc = assess_text(plain)
            if words and _better(qc, pd.qc):
                new = PageData(
                    page_no=pd.page_no,
                    width=pd.width,
                    height=pd.height,
                    words=words,
                    text=words_to_layout_text(words, pd.width),
                    plain=plain,
                    qc=qc,
                    image_share=pd.image_share,
                    text_source=src,
                    ocr_layer=False,
                    needs_ocr=not qc.usable,
                )
                states[i] = (new, *_page_state(new, None, None, pe))
        prev_scope, prev_cols = None, None
        rebuilt: list[tuple[PageData, ParsedTable, PageInfo]] = []
        for pd, _, _ in states:
            table, info = _page_state(pd, prev_scope, prev_cols, pe)
            rebuilt.append((pd, table, info))
            if info.is_results:
                prev_scope = info.scope if info.scope not in (None, "both") else prev_scope
                prev_cols = table.columns if any(c.period for c in table.columns) else prev_cols
        states = rebuilt

    result_nos = [pd.page_no for pd, _, info in states if info.is_results]
    pngs: dict[int, Path] = {}
    if result_nos:
        try:
            pngs = render_pages_png(pdf_path, result_nos, png_dir, dpi=130)
        except Exception as exc:
            log.warning("png render failed for %s: %s", fid, exc)

    rows: list[dict[str, Any]] = []
    for pd, table, info in states:
        weak = info.is_results and (
            any(c.period is None or c.inferred for c in table.columns) or pd.qc.score < 0.95 or pd.qc.digit_clean_rate < 0.93
        )
        if weak and pd.text_source == "text_layer":
            pd.needs_ocr = True
        rows.append(
            {
                "page_id": make_page_id(fid, pd.page_no),
                "filing_id": fid,
                "page_no": pd.page_no,
                "text": pd.text,
                "words": jdump([w.to_list() for w in pd.words]) if pd.words else "[]",
                "text_source": pd.text_source,
                "is_results_page": info.is_results,
                "scope": info.scope,
                "png_path": str(pngs[pd.page_no]) if pd.page_no in pngs else None,
                "page_type": info.page_type,
                "qc_score": pd.qc.score,
                "ocr_layer": pd.ocr_layer,
                "needs_ocr": bool(pd.needs_ocr),
                "unit": table.unit,
                "width": pd.width,
                "height": pd.height,
                "columns": jdump([{"period": str(c.period) if c.period else None, "scope": c.scope, "inferred": c.inferred} for c in table.columns]),
                "tbl": jdump(table_to_json(table)) if table.rows else "",
            }
        )
    return {"filing_id": fid, "pages": rows, "ocr_pages": ocr_done}


def _ingest_meta(store: Store, meta: FilingMeta, meta_path: Path, paths: Paths, profile_name: str | None) -> bool:
    cid = (meta.isin or meta.symbol).upper()
    store.upsert(
        "companies",
        ["company_id"],
        {
            "company_id": cid,
            "symbol": meta.symbol,
            "isin": meta.isin,
            "name": meta.company,
            "aliases": company_aliases(meta.company, meta.symbol),
            "sector": meta.industry or None,
            "in_nifty200": bool(meta.in_nifty200),
            "scrip_code": meta.scrip_code,
        },
    )
    facts_rows: list[dict[str, Any]] = []
    xbrl_paths: list[str] = []
    by_scope: dict[str, list[tuple[str, Any]]] = {}
    revised = False
    board = parse_date_flexible(meta.board_meeting_date)
    for xf in sorted(meta.xbrl, key=lambda x: (x.created_at or "", x.seq_id)):
        path = paths.work / xf.path
        if not path.exists():
            continue
        try:
            parsed = parse_xbrl(path.read_bytes())
        except XbrlError as exc:
            log.warning("xbrl parse failed %s: %s", path, exc)
            continue
        scope = xf.scope or parsed.scope or "standalone"
        xbrl_paths.append(xf.path)
        by_scope.setdefault(scope, []).append((xf.seq_id, parsed))
        if xf.type_sub.lower() == "revision":
            revised = True
        if board is None and parsed.meta.get("board_meeting_date"):
            board = parsed.meta["board_meeting_date"]
    store.x("DELETE FROM xbrl_facts WHERE filing_id = ?", [meta.filing_id])
    for scope, plist in by_scope.items():
        core_idx = [i for i, (_, p) in enumerate(plist) if has_core_items(p)]
        latest_idx = core_idx[-1] if core_idx else len(plist) - 1
        for i, (seq, parsed) in enumerate(plist):
            for f in parsed.facts:
                facts_rows.append(
                    {
                        "filing_id": meta.filing_id,
                        "scope": scope,
                        "line_item": f.line_item,
                        "period_end": f.period_end,
                        "duration_months": f.duration_months,
                        "value_inr": f.value,
                        "decimals": f.decimals,
                        "raw_element": f.raw_element,
                        "xbrl_seq": seq,
                        "revision": i,
                        "is_latest": i == latest_idx,
                    }
                )
    store.insert_rows("xbrl_facts", facts_rows)
    pdf_pub = parse_iso(meta.pdf_published_at)
    if board is None and pdf_pub:
        board = utc_to_ist(pdf_pub).date()
    pdf_rel = (meta.pdf or {}).get("path")
    n_pages = store.count("pages", "filing_id = ?", [meta.filing_id])
    existing = store.filing(meta.filing_id)
    store.upsert(
        "filings",
        ["filing_id"],
        {
            "filing_id": meta.filing_id,
            "company_id": cid,
            "exchange": meta.exchange,
            "period_end": parse_date_flexible(meta.period_end),
            "board_meeting_date": board,
            "pdf_published_at": pdf_pub,
            "xbrl_published_at": parse_iso(meta.xbrl_published_at),
            "pdf_path": pdf_rel,
            "xbrl_paths": xbrl_paths,
            "split": existing["split"] if existing else None,
            "role": meta.role,
            "status": meta.status,
            "pdf_source": meta.pdf_source,
            "n_pages": n_pages,
            "revised": revised,
            "scanned_pages": existing["scanned_pages"] if existing else 0,
            "meta": jdump({"pdf_text": meta.pdf_text, "meta_path": str(meta_path), "pdf_url": (meta.pdf or {}).get("url"), "pdf_source": meta.pdf_source}),
        },
    )
    return bool(pdf_rel)


def assign_all_splits(store: Store, profile_name: str | None = None, freeze: bool = True) -> dict[str, int]:
    gcfg = gym_config(profile_name)
    prof = gcfg.profile
    assignments: dict[str, str] = {}
    for f in store.filings("role = 'target'"):
        bm: date | None = f["board_meeting_date"]
        s = assign_split(bm, prof)
        if s and f["pdf_path"]:
            assignments[f["filing_id"]] = s
    for fid, s in assignments.items():
        store.x("UPDATE filings SET split = ? WHERE filing_id = ?", [s, fid])
    store.x(
        "UPDATE filings SET split = NULL WHERE role <> 'target' OR filing_id NOT IN (" + ",".join("?" for _ in assignments) + ")",
        list(assignments),
    ) if assignments else None
    if freeze and not get_paths().splits_path.exists():
        freeze_splits(assignments, prof)
    counts: dict[str, int] = {}
    for s in assignments.values():
        counts[s] = counts.get(s, 0) + 1
    return counts


def ingest(
    workers: int = 8,
    limit: int | None = None,
    force: bool = False,
    ocr: str = "none",
    profile: str | None = None,
    roles: list[str] | None = None,
    echo: Any = print,
) -> dict[str, Any]:
    paths = get_paths().ensure()
    store = Store()
    metas = [(p, load_meta(p)) for p in all_meta_paths(paths)]
    metas = [(p, m) for p, m in metas if m is not None and (not roles or m.role in roles)]
    jobs: list[dict[str, Any]] = []
    n_xbrl_only = 0
    backends: list[str] = []
    if ocr == "auto":
        backends = available_backends()
    elif ocr in ("tesseract", "doctr"):
        backends = [ocr]
    echo(f"ingest: {len(metas)} filings, ocr backends={backends or 'none'}")
    for mp, meta in metas:
        assert meta is not None
        has_pdf = _ingest_meta(store, meta, mp, paths, profile)
        if not has_pdf:
            n_xbrl_only += 1
            continue
        done = store.count("pages", "filing_id = ?", [meta.filing_id]) > 0
        if done and not force:
            continue
        jobs.append(
            {
                "filing_id": meta.filing_id,
                "pdf_path": str(paths.work / (meta.pdf or {})["path"]),
                "png_dir": str(paths.pages / meta.filing_id),
                "ocr_backends": backends,
                "ocr_cap": 12,
                "device": os.environ.get("PRAMAAN_OCR_DEVICE"),
                "period_end": meta.period_end,
            }
        )
        if limit and len(jobs) >= limit:
            break
    echo(f"ingest: {len(jobs)} PDFs to process, {n_xbrl_only} xbrl-only filings")
    done_n = 0
    if jobs:
        if force:
            for j in jobs:
                store.x("DELETE FROM pages WHERE filing_id = ?", [j["filing_id"]])
        if workers <= 1:
            iterator = (process_filing_pdf(j) for j in jobs)
            for res in iterator:
                _store_result(store, res)
                done_n += 1
                echo(f"  [{done_n}/{len(jobs)}] {res['filing_id']} pages={len(res['pages'])} ocr={res['ocr_pages']}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_filing_pdf, j): j for j in jobs}
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        log.exception("filing failed %s: %s", futs[fut]["filing_id"], exc)
                        continue
                    _store_result(store, res)
                    done_n += 1
                    if done_n % 10 == 0 or done_n == len(jobs):
                        echo(f"  [{done_n}/{len(jobs)}] {res['filing_id']}")
    for mp, meta in metas:
        assert meta is not None
        n = store.count("pages", "filing_id = ?", [meta.filing_id])
        scanned = store.count("pages", "filing_id = ? AND needs_ocr", [meta.filing_id])
        store.x("UPDATE filings SET n_pages = ?, scanned_pages = ? WHERE filing_id = ?", [n, scanned, meta.filing_id])
    counts = assign_all_splits(store, profile)
    summary = {
        "filings": store.count("filings"),
        "with_pdf": store.count("filings", "pdf_path IS NOT NULL"),
        "pages": store.count("pages"),
        "results_pages": store.count("pages", "is_results_page"),
        "facts": store.count("xbrl_facts"),
        "splits": counts,
        "frozen": bool(load_splits().get("assignments")),
    }
    store.close()
    return summary


def _store_result(store: Store, res: dict[str, Any]) -> None:
    fid = res["filing_id"]
    store.x("DELETE FROM pages WHERE filing_id = ?", [fid])
    store.insert_rows("pages", res["pages"])
    store.x("UPDATE filings SET n_pages = ? WHERE filing_id = ?", [len(res["pages"]), fid])


def parse_quality(table: ParsedTable, info: PageInfo, qc_score: float) -> float:
    periods = sum(1 for c in table.columns if c.period and not c.inferred)
    full_rows = sum(1 for r in table.rows if r.n_values >= max(2, len(table.columns) - 1))
    return 2.0 * periods + 1.5 * info.n_items + 0.05 * full_rows + 3.0 * qc_score + (2.0 if table.unit else 0.0)


def reprocess_filing(store: Store, filing_id: str, replacements: dict[int, list[Word]], source: str = "doctr") -> dict[str, int]:
    f = store.filing(filing_id)
    pe = f["period_end"] if f else None
    rows = store.pages_of(filing_id)
    prev_scope: str | None = None
    prev_cols: list[Any] | None = None
    updated = kept = 0
    for r in rows:
        words = [Word.from_list(w) for w in (jload(r["words"], []) or [])]
        width, height = float(r["width"] or 595), float(r["height"] or 842)
        pd_old = PageData(r["page_no"], width, height, words, r["text"] or "", plain_text(words), assess_text(plain_text(words)), 0.0, r["text_source"], bool(r["ocr_layer"]))
        t_old, i_old = _page_state(pd_old, prev_scope, prev_cols, pe)
        chosen_pd, chosen_t, chosen_i, src = pd_old, t_old, i_old, r["text_source"]
        new_words = replacements.get(r["page_no"])
        if new_words:
            plain = plain_text(new_words)
            qc = assess_text(plain)
            pd_new = PageData(r["page_no"], width, height, new_words, words_to_layout_text(new_words, width), plain, qc, 0.0, source, False)
            t_new, i_new = _page_state(pd_new, prev_scope, prev_cols, pe)
            if parse_quality(t_new, i_new, qc.score) > parse_quality(t_old, i_old, pd_old.qc.score) + 0.5:
                chosen_pd, chosen_t, chosen_i, src = pd_new, t_new, i_new, source
                updated += 1
            else:
                kept += 1
        if chosen_i.is_results:
            prev_scope = chosen_i.scope if chosen_i.scope not in (None, "both") else prev_scope
            prev_cols = chosen_t.columns if any(c.period for c in chosen_t.columns) else prev_cols
        store.x(
            "UPDATE pages SET text = ?, words = ?, text_source = ?, is_results_page = ?, scope = ?, page_type = ?, qc_score = ?, needs_ocr = ?, unit = ?, columns = ?, tbl = ? WHERE page_id = ?",
            [
                chosen_pd.text,
                jdump([w.to_list() for w in chosen_pd.words]),
                src,
                chosen_i.is_results,
                chosen_i.scope,
                chosen_i.page_type,
                chosen_pd.qc.score,
                not chosen_pd.qc.usable,
                chosen_t.unit,
                jdump([{"period": str(c.period) if c.period else None, "scope": c.scope, "inferred": c.inferred} for c in chosen_t.columns]),
                jdump(table_to_json(chosen_t)) if chosen_t.rows else "",
                r["page_id"],
            ],
        )
    return {"updated": updated, "kept_text_layer": kept}


def select_ocr_pages(store: Store, per_filing_cap: int = 40) -> list[dict[str, Any]]:
    """Pages worth sending to OCR: flagged `needs_ocr` and either a results page, nearly blank, or on
    a filing where no results page was found at all (otherwise the pipeline has nothing to read there,
    and those filings contribute no questions). At most `per_filing_cap` pages per filing."""
    paths = get_paths().ensure()
    no_results = {
        r["filing_id"]
        for r in store.q(
            "SELECT f.filing_id FROM filings f WHERE f.pdf_path IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM pages p WHERE p.filing_id = f.filing_id) "
            "AND NOT EXISTS (SELECT 1 FROM pages p WHERE p.filing_id = f.filing_id AND p.is_results_page)"
        )
    }
    rows = store.q(
        "SELECT p.page_id, p.page_no, p.width, p.height, p.is_results_page, p.needs_ocr, length(p.text) AS n, f.pdf_path, f.filing_id "
        "FROM pages p JOIN filings f USING (filing_id) WHERE f.pdf_path IS NOT NULL AND p.needs_ocr ORDER BY f.filing_id, p.page_no"
    )
    by_f: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    for r in rows:
        wanted = r["is_results_page"] or (r["n"] or 0) < 40 or r["filing_id"] in no_results
        if not wanted:
            continue
        by_f[r["filing_id"]] = by_f.get(r["filing_id"], 0) + 1
        if by_f[r["filing_id"]] > per_filing_cap:
            continue
        items.append(
            {
                "page_id": r["page_id"],
                "page_no": r["page_no"],
                "width": r["width"],
                "height": r["height"],
                "filing_id": r["filing_id"],
                "pdf_path": str(paths.work / r["pdf_path"]),
                "reason": "results_page" if r["is_results_page"] else ("blank" if (r["n"] or 0) < 40 else "no_results_page_in_filing"),
            }
        )
    return items


def export_scanned_pages_cmd(out_dir: Path, per_filing_cap: int = 40) -> dict[str, Any]:
    from pramaan.data.ocr import export_scanned_pages

    store = Store(read_only=True)
    items = select_ocr_pages(store, per_filing_cap)
    zp = export_scanned_pages(items, out_dir)
    return {"pages": len(items), "filings": len({i["filing_id"] for i in items}), "zip": str(zp)}


def import_ocr_cmd(zip_path: Path) -> dict[str, Any]:
    from pramaan.data.ocr import read_ocr_zip, words_from_doc

    store = Store()
    docs = read_ocr_zip(zip_path)
    by_filing: dict[str, dict[int, list[Word]]] = {}
    for pid, doc in docs.items():
        fid, no = pid.rsplit("#p", 1)
        by_filing.setdefault(fid, {})[int(no)] = words_from_doc(doc)
    tot = {"filings": 0, "updated": 0, "kept_text_layer": 0}
    for fid, repl in by_filing.items():
        res = reprocess_filing(store, fid, repl)
        tot["filings"] += 1
        tot["updated"] += res["updated"]
        tot["kept_text_layer"] += res["kept_text_layer"]
    store.close()
    return tot
