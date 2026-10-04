from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from pramaan.data.ocr import OCR_DPI, ocr_image


def ocr_zip(in_zip: Path, out_zip: Path, backend: str = "doctr", device: str | None = None, limit: int | None = None, echo: Any = print) -> dict[str, Any]:
    work = out_zip.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(in_zip) as z:
        z.extractall(work / "in")
    manifest = json.loads((work / "in" / "manifest.json").read_text(encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    done = 0
    for i, item in enumerate(manifest):
        target = out_dir / (item["page_id"].replace("#", "_") + ".json")
        if target.exists():
            done += 1
            continue
        if "pdf" in item:  # a bundle of PDFs: render the page here, so the upload stays small
            from pramaan.data.pdf import render_page_image

            img = render_page_image(str(work / "in" / item["pdf"]), int(item["page_no"]), OCR_DPI).convert("RGB")
        else:
            img = Image.open(work / "in" / item["image"]).convert("RGB")
        words = ocr_image(img, float(item["width"]), float(item["height"]), backend, device)
        target.write_text(json.dumps({"page_id": item["page_id"], "backend": backend, "width": item["width"], "height": item["height"], "words": [w.to_list() for w in words]}), encoding="utf-8")
        done += 1
        if (i + 1) % 20 == 0:
            echo(f"ocr {i + 1}/{len(manifest)} pages, {round(time.time() - t0)}s")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.glob("*.json")):
            z.write(p, p.name)
    return {"pages": done, "seconds": round(time.time() - t0, 1), "zip": str(out_zip), "backend": backend, "device": device or "cpu"}
