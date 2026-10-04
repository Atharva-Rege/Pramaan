import json
import zipfile

from reportlab.pdfgen import canvas

from pramaan.data import ocr_run
from pramaan.data.layout import Word


def _pdf(path, pages):
    c = canvas.Canvas(str(path))
    for n in range(pages):
        c.drawString(72, 700, f"page {n + 1}")
        c.showPage()
    c.save()


def test_ocr_zip_renders_pdf_pages_and_writes_one_json_per_page(tmp_path, monkeypatch):
    pdf = tmp_path / "f.pdf"
    _pdf(pdf, 3)
    manifest = [
        {"page_id": "f#p2", "pdf": "pdfs/f.pdf", "page_no": 2, "width": 595.0, "height": 842.0},
        {"page_id": "f#p3", "pdf": "pdfs/f.pdf", "page_no": 3, "width": 595.0, "height": 842.0},
    ]
    inp = tmp_path / "in.zip"
    with zipfile.ZipFile(inp, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.write(pdf, "pdfs/f.pdf")
    seen = []

    def fake_ocr(img, w, h, backend, device):
        seen.append((img.size, backend, device))
        return [Word("ok", 1.0, 2.0, 3.0, 4.0)]

    monkeypatch.setattr(ocr_run, "ocr_image", fake_ocr)
    out = tmp_path / "out.zip"
    res = ocr_run.ocr_zip(inp, out, "doctr", "cuda")
    assert res["pages"] == 2 and res["device"] == "cuda"
    assert all(size[0] > 1000 for size, _, _ in seen)  # rendered at OCR resolution, not the 72 dpi page size
    with zipfile.ZipFile(out) as z:
        names = sorted(z.namelist())
        assert names == ["f_p2.json", "f_p3.json"]
        doc = json.loads(z.read("f_p2.json"))
        assert doc["page_id"] == "f#p2" and doc["words"] == [["ok", 1.0, 2.0, 3.0, 4.0]]
