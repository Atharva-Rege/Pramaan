from __future__ import annotations

import io
import json
import logging
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from pramaan.data.layout import Word

log = logging.getLogger("pramaan.ocr")

OCR_DPI = 200


def _have_tesseract() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _have_doctr() -> bool:
    try:
        import doctr  # noqa: F401
    except ImportError:
        return False
    return True


def available_backends() -> list[str]:
    out: list[str] = []
    if _have_tesseract():
        out.append("tesseract")
    if _have_doctr():
        out.append("doctr")
    return out


@lru_cache(maxsize=1)
def _doctr_predictor(device: str | None = None) -> Any:
    from doctr.models import ocr_predictor

    model = ocr_predictor(
        det_arch="db_resnet50" if device == "cuda" else "db_mobilenet_v3_large",
        reco_arch="crnn_vgg16_bn" if device == "cuda" else "crnn_mobilenet_v3_small",
        pretrained=True,
    )
    if device == "cuda":
        model = model.cuda()
    return model


def ocr_tesseract(img: Image.Image, page_w: float, page_h: float, lang: str = "eng") -> list[Word]:
    import pytesseract

    data = pytesseract.image_to_data(img, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT)
    sx, sy = page_w / img.width, page_h / img.height
    words: list[Word] = []
    for i, text in enumerate(data["text"]):
        t = (text or "").strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(t, x * sx, y * sy, (x + w) * sx, (y + h) * sy))
    return words


def ocr_doctr(img: Image.Image, page_w: float, page_h: float, device: str | None = None) -> list[Word]:
    import numpy as np

    model = _doctr_predictor(device)
    result = model([np.asarray(img.convert("RGB"))])
    words: list[Word] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for w in line.words:
                    (x0, y0), (x1, y1) = w.geometry
                    words.append(Word(w.value, x0 * page_w, y0 * page_h, x1 * page_w, y1 * page_h))
    return words


def ocr_image(img: Image.Image, page_w: float, page_h: float, backend: str, device: str | None = None) -> list[Word]:
    if backend == "tesseract":
        return ocr_tesseract(img, page_w, page_h)
    if backend == "doctr":
        return ocr_doctr(img, page_w, page_h, device)
    raise ValueError(f"unknown OCR backend {backend}")


def export_scanned_pages(
    items: list[dict[str, Any]],
    out_dir: Path,
    zip_path: Path | None = None,
) -> Path:
    from pramaan.data.pdf import render_page_image

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for it in items:
        png = out_dir / f"{it['page_id'].replace('#', '_')}.png"
        if not png.exists():
            render_page_image(it["pdf_path"], it["page_no"], OCR_DPI).save(png, optimize=True)
        manifest.append(
            {
                "page_id": it["page_id"],
                "image": png.name,
                "width": it["width"],
                "height": it["height"],
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    zp = zip_path or out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            z.write(p, p.name)
    return zp


def read_ocr_zip(zip_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".json") and name != "manifest.json":
                doc = json.loads(z.read(name).decode("utf-8"))
                out[doc["page_id"]] = doc
    return out


def words_from_doc(doc: dict[str, Any]) -> list[Word]:
    return [Word.from_list(v) for v in doc["words"]]


def image_bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
