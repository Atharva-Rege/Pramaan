import argparse
import json
from pathlib import Path

from pramaan.data.ocr import available_backends
from pramaan.data.ocr_run import ocr_zip

p = argparse.ArgumentParser(description="CPU fallback for notebook 02: OCR a scanned-pages zip with Tesseract or docTR")
p.add_argument("zip")
p.add_argument("--out", default="ocr_results.zip")
p.add_argument("--backend", default=None)
p.add_argument("--limit", type=int, default=None)
a = p.parse_args()
backend = a.backend or (available_backends() or ["doctr"])[0]
print(json.dumps(ocr_zip(Path(a.zip), Path(a.out), backend, None, a.limit), indent=1))
print("import with: pramaan ingest --import-ocr", a.out)
