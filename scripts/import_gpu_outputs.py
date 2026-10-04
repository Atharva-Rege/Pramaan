"""Bring the results of notebooks/03_gpu_colab.ipynb back into the project.

    python scripts/import_gpu_outputs.py ~/Downloads/pramaan_gpu_outputs.zip [--home DIR]

* ocr_results.zip is loaded with `pramaan ingest --import-ocr` (pages get text_source = doctr and are
  re-parsed; the text layer is kept wherever it parses better). This changes the database.
* reranker_model.zip is unpacked to <home>/artifacts/reranker/gpu/model.
"""

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("outputs")
p.add_argument("--home", default=None, help="data home (default: PRAMAAN_HOME or the repository)")
p.add_argument("--skip-ocr", action="store_true")
a = p.parse_args()
if a.home:
    os.environ["PRAMAAN_HOME"] = str(Path(a.home).expanduser())

from pramaan.core.paths import get_paths

paths = get_paths()
with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(a.outputs) as z:
    z.extractall(tmp)
    tmpd = Path(tmp)
    report = json.loads((tmpd / "run_report.json").read_text()) if (tmpd / "run_report.json").exists() else {}
    print("run report:", json.dumps({k: report.get(k) for k in ("gpu", "ocr", "reranker")}, default=str)[:600])
    if (tmpd / "ocr_results.zip").exists() and not a.skip_ocr:
        from pramaan.data.ingest import import_ocr_cmd

        print("OCR import:", import_ocr_cmd(tmpd / "ocr_results.zip"))
        has_results = (paths.results / "E3").exists() and any((paths.results / "E3").glob("*_dev.json"))
        print("next: pramaan index && pramaan gym build && pramaan exp E1")
        if has_results:
            print(
                "WARNING: this home already has evaluation results. Rebuilding the gym after OCR changes which pages count as results\n"
                "pages and so which questions exist, so new numbers will not be comparable with the old ones. Do that on a copy, or\n"
                "on the season home."
            )
    if (tmpd / "reranker_model.zip").exists():
        dest = paths.artifacts / "reranker" / "gpu" / "model"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmpd / "reranker_model.zip") as mz:
            mz.extractall(dest)
        print(f"reranker model unpacked to {dest}\nuse it with:  PRAMAAN_RERANKER_PATH={dest} pramaan exp E2 --finetuned {dest}")
