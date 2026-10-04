from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2 as pdfium

from pramaan.data.layout import Word, merge_fragments, plain_text, words_to_layout_text
from pramaan.data.textqc import TextQuality, assess_text

log = logging.getLogger("pramaan.pdf")

RENDER_DPI = 150
SCAN_IMAGE_SHARE = 0.8


@dataclass
class PageData:
    page_no: int
    width: float
    height: float
    words: list[Word]
    text: str
    plain: str
    qc: TextQuality
    image_share: float
    text_source: str = "text_layer"
    ocr_layer: bool = False
    needs_ocr: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _image_share(page: Any) -> float:
    try:
        area = 0.0
        pw, ph = float(page.width), float(page.height)
        for im in page.images:
            x0, x1 = max(0.0, im["x0"]), min(pw, im["x1"])
            top, bottom = max(0.0, im["top"]), min(ph, im["bottom"])
            if x1 > x0 and bottom > top:
                area += (x1 - x0) * (bottom - top)
        return min(1.0, area / max(1.0, pw * ph))
    except Exception:
        return 0.0


def extract_page(page: Any, page_no: int) -> PageData:
    raw = page.extract_words(x_tolerance=1.5, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=False)
    words = merge_fragments(
        [Word(w["text"], float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])) for w in raw]
    )
    width, height = float(page.width), float(page.height)
    layout = words_to_layout_text(words, width)
    plain = plain_text(words)
    qc = assess_text(plain)
    share = _image_share(page)
    scan_like = share >= SCAN_IMAGE_SHARE
    ocr_layer = scan_like and qc.n_chars >= 40
    needs_ocr = (not qc.usable) or (ocr_layer and (qc.score < 0.9 or qc.digit_clean_rate < 0.9))
    return PageData(
        page_no=page_no,
        width=width,
        height=height,
        words=words,
        text=layout,
        plain=plain,
        qc=qc,
        image_share=round(share, 3),
        text_source="text_layer",
        ocr_layer=ocr_layer,
        needs_ocr=needs_ocr,
    )


def extract_pdf(path: str | Path, max_pages: int | None = None) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages is not None and i >= max_pages:
                break
            try:
                pages.append(extract_page(page, i + 1))
            except Exception as exc:
                log.warning("page %d of %s failed: %s", i + 1, path, exc)
                pages.append(
                    PageData(
                        page_no=i + 1,
                        width=float(page.width),
                        height=float(page.height),
                        words=[],
                        text="",
                        plain="",
                        qc=assess_text(""),
                        image_share=0.0,
                        needs_ocr=True,
                    )
                )
            page.flush_cache()
    return pages


def extract_many(paths: list[Path], workers: int = 8) -> dict[Path, list[PageData]]:
    if workers <= 1 or len(paths) == 1:
        return {p: extract_pdf(p) for p in paths}
    out: dict[Path, list[PageData]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, pages in zip(paths, ex.map(extract_pdf, paths), strict=True):
            out[p] = pages
    return out


def render_pages_png(pdf_path: str | Path, page_numbers: list[int], out_dir: Path, dpi: int = RENDER_DPI) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    out: dict[int, Path] = {}
    try:
        for n in page_numbers:
            if n < 1 or n > len(doc):
                continue
            target = out_dir / f"p{n}.png"
            if not target.exists():
                bitmap = doc[n - 1].render(scale=dpi / 72.0)
                img = bitmap.to_pil().convert("L") if dpi < 110 else bitmap.to_pil().convert("RGB")
                img.save(target, optimize=True)
            out[n] = target
    finally:
        doc.close()
    return out


def render_page_image(pdf_path: str | Path, page_no: int, dpi: int = RENDER_DPI) -> Any:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return doc[page_no - 1].render(scale=dpi / 72.0).to_pil().convert("RGB")
    finally:
        doc.close()
