"""Build pramaan_gpu_inputs.zip: the one file notebooks/03_gpu_colab.ipynb needs.

It holds
  Pramaan.zip                       the code (scripts/make_colab_zip.py)
  ocr_inputs.zip                    manifest.json plus one small PDF per filing holding only the pages worth
                                    OCR; the notebook renders them itself on the Colab machine
  reranker/reranker_pairs.parquet   reranker training pairs (train split only), if exported
  reranker/reranker_groups.parquet
  bundle_manifest.json              what is inside, with SHA-256 hashes

The data home is PRAMAAN_HOME (or --home): the dry-run database for a coverage check, or the season
home (~/.local/share/pramaan-season) once the September data is ingested. The bundle contains exchange
documents, so keep it private (Drive, or a private Hugging Face dataset), never a public link.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--home", default=None, help="data home (default: PRAMAAN_HOME or the repository)")
p.add_argument("--out", default=str(ROOT / "colab_inputs" / "pramaan_gpu_inputs.zip"))
p.add_argument("--per-filing-cap", type=int, default=40)
p.add_argument("--no-reranker", action="store_true")
a = p.parse_args()
if a.home:
    os.environ["PRAMAAN_HOME"] = str(Path(a.home).expanduser())

from pramaan.core.paths import get_paths
from pramaan.data.ingest import select_ocr_pages
from pramaan.data.store import Store

paths = get_paths()
store = Store(read_only=True)
items = select_ocr_pages(store, a.per_filing_cap)
store.close()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


with tempfile.TemporaryDirectory() as tmp:
    tmpd = Path(tmp)
    # 1. OCR inputs: manifest plus, per filing, a small PDF holding only the pages to OCR (whole PDFs are
    #    about 4.5x larger). `page_no` in the manifest is the page's position in that small PDF.
    import pypdfium2 as pdfium

    ocr_zip = tmpd / "ocr_inputs.zip"
    by_filing: dict[str, list[dict]] = {}
    for i in items:
        by_filing.setdefault(i["filing_id"], []).append(i)
    filings = sorted(by_filing)
    manifest: list[dict] = []
    pdf_bytes = 0
    with zipfile.ZipFile(ocr_zip, "w", zipfile.ZIP_STORED) as z:
        for fid in filings:
            src = Path(by_filing[fid][0]["pdf_path"])
            if not src.exists():
                print(f"  skipped {fid}: {src} is missing")
                continue
            pages = sorted(by_filing[fid], key=lambda i: i["page_no"])
            doc, sub = pdfium.PdfDocument(str(src)), pdfium.PdfDocument.new()
            sub.import_pages(doc, [i["page_no"] - 1 for i in pages])
            part = tmpd / f"{fid}.pdf"
            sub.save(str(part))
            sub.close()
            doc.close()
            z.write(part, f"pdfs/{fid}.pdf")
            pdf_bytes += part.stat().st_size
            part.unlink()
            for pos, i in enumerate(pages, start=1):
                manifest.append(
                    {
                        "page_id": i["page_id"],
                        "pdf": f"pdfs/{fid}.pdf",
                        "page_no": pos,
                        "source_page_no": i["page_no"],
                        "width": i["width"],
                        "height": i["height"],
                        "filing_id": fid,
                        "reason": i["reason"],
                    }
                )
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    # 2. the code
    code_zip = tmpd / "Pramaan.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_colab_zip.py"), "--out", str(code_zip)],
        check=True,
        capture_output=True,
    )
    # 3. reranker training pairs, if they were exported from this home
    rr: list[tuple[Path, str]] = []
    if not a.no_reranker:
        for name in ("reranker_pairs.parquet", "reranker_groups.parquet"):
            src = paths.exports / name
            if src.exists():
                rr.append((src, f"reranker/{name}"))
    parts: list[tuple[Path, str]] = [(code_zip, "Pramaan.zip"), (ocr_zip, "ocr_inputs.zip"), *rr]
    info = {
        "built_from": str(paths.home),
        "ocr_pages": len(manifest),
        "ocr_filings": len(filings),
        "ocr_reasons": {
            r: sum(1 for m in manifest if m["reason"] == r)
            for r in sorted({m["reason"] for m in manifest})
        },
        "reranker_files": [n for _, n in rr],
        "sha256": {n: sha(src) for src, n in parts},
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # parts are already compressed
        for src, name in parts:
            z.write(src, name)
        z.writestr("bundle_manifest.json", json.dumps(info, indent=1))
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
print(
    f"  OCR: {len(manifest)} pages on {len(filings)} filings ({pdf_bytes / 1e6:.0f} MB of PDFs); reasons {info['ocr_reasons']}"
)
print(
    f"  reranker files: {info['reranker_files'] or 'none (run `pramaan improve export-reranker-data` first)'}"
)
