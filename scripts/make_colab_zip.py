import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pramaan", "configs", "scripts", "notebooks", "tests", "ci", "web/src", "web/public", "web/index.html", "web/package.json", "web/package-lock.json", "web/tsconfig.json", "web/vite.config.ts"]
FILES = ["pyproject.toml", "README.md", "LICENSE", "Makefile", "requirements-gpu-notebook.txt", ".env.example", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "dist"}

p = argparse.ArgumentParser()
p.add_argument("--out", default="Pramaan.zip")
p.add_argument("--seed", default=None, help="optional pramaan_data_seed.zip to embed under seed/")
p.add_argument("--with-web-dist", action="store_true")
a = p.parse_args()

out = Path(a.out)
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    def add(path: Path) -> None:
        global n
        if any(part in SKIP_PARTS for part in path.parts):
            return
        z.write(path, Path("Pramaan") / path.relative_to(ROOT))
        n += 1

    for item in INCLUDE:
        base = ROOT / item
        if base.is_file():
            add(base)
        elif base.is_dir():
            for f in sorted(base.rglob("*")):
                if f.is_file() and f.suffix not in {".pyc", ".duckdb", ".sqlite"}:
                    add(f)
    for f in FILES:
        if (ROOT / f).exists():
            add(ROOT / f)
    if a.with_web_dist and (ROOT / "web" / "dist").exists():
        for f in (ROOT / "web" / "dist").rglob("*"):
            if f.is_file():
                z.write(f, Path("Pramaan") / f.relative_to(ROOT))
                n += 1
    if a.seed and Path(a.seed).exists():
        z.write(a.seed, Path("Pramaan") / "seed" / "pramaan_data_seed.zip")
        n += 1
print(f"{out}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
