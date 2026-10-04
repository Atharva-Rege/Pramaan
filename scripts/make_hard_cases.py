"""Compose docs/img/hard_cases.png: three real excerpts that show why filings are hard.

Crops come from page renders the pipeline already produced (data/pages), located through the
word boxes of the stored text layer. The excerpts are small and shown for illustration with the
filing named under each; the originals stay on the exchange sites.
"""

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pramaan.core.paths import REPO_ROOT
from pramaan.data.store import Store, jload

st = Store(read_only=True)


def words_of(row: dict) -> list[list]:
    return jload(row["words"], [])


def crop(row: dict, word: list, above: float = 70, below: float = 150) -> Image.Image:
    """Crop a band around the word and draw a red box on the word itself."""
    im = Image.open(row["png_path"]).convert("RGB")
    s = im.width / float(row["width"])
    top = max(0, int((float(word[2]) - above) * s))
    band = im.crop((0, top, im.width, min(im.height, int((float(word[2]) + below) * s))))
    x0, y0, x1, y1 = (float(word[1]) * s, float(word[2]) * s - top, float(word[3]) * s, float(word[4]) * s - top)
    ImageDraw.Draw(band).rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), outline="#dc2626", width=3)
    return band


def find(pattern: str, where: str, scope: str | None = None) -> tuple[dict, list] | None:
    sql = f"SELECT p.page_id, p.png_path, p.words, p.width, p.scope, p.text_source, f.company_id FROM pages p JOIN filings f USING (filing_id) WHERE p.is_results_page AND p.png_path IS NOT NULL AND {where} ORDER BY p.page_id"
    rx = re.compile(pattern, re.I)
    for r in st.q(sql):
        for w in words_of(r):
            if rx.fullmatch(str(w[0])):
                return r, w
    return None


cases = []
# 1. the same kind of table printed in lakhs (a missed unit header is a 100x error)
hit = find(r"lakhs?", "p.unit = 'lakh'")
if hit:
    cases.append(("Lakhs, not crores: the unit is stated once in the header, and a missed header is a 100x error", *hit))
# 2. a noisy OCR layer: months and digits are corrupted ("Jl-03-2026", "34.0S")
hit = find(r"J[l1]-\d{2}-\d{4}", "TRUE") or find(r"\d+\.\d[SO]", "TRUE")
if hit:
    cases.append((f"Scanned page, noisy OCR layer: the image is legible but the text layer reads {hit[1][0]!r} at the red box", *hit))
# 3. standalone and consolidated side by side in one table
for r in st.q("SELECT p.page_id, p.png_path, p.words, p.width, p.scope, p.text_source, f.company_id FROM pages p JOIN filings f USING (filing_id) WHERE p.is_results_page AND p.png_path IS NOT NULL ORDER BY p.page_id"):
    ws = words_of(r)
    sa = [w for w in ws if str(w[0]).lower() == "standalone" and w[2] < 200]
    co = [w for w in ws if str(w[0]).lower() == "consolidated" and w[2] < 200]
    if sa and co and abs(sa[0][2] - co[0][2]) < 8 and abs(sa[0][1] - co[0][1]) > 60:
        cases.append(("Standalone and consolidated columns side by side in one table", r, sa[0]))
        break

def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


font, small = load_font(19), load_font(15)
panels = []
for title, r, w in cases:
    im = crop(r, w)
    cap_h = 56
    p = Image.new("RGB", (im.width, im.height + cap_h), "white")
    p.paste(im, (0, cap_h))
    d = ImageDraw.Draw(p)
    d.text((8, 6), title, fill="black", font=font)
    d.text((8, 33), f"{r['page_id']}  (embedded text layer; red box marks the located word)", fill="#555555", font=small)
    panels.append(p)
if not panels:
    raise SystemExit("no examples found")
W = max(p.width for p in panels)
sheet = Image.new("RGB", (W, sum(p.height + 8 for p in panels)), "white")
y = 0
for p in panels:
    sheet.paste(p, (0, y))
    y += p.height + 8
out = Path(REPO_ROOT / "docs" / "img" / "hard_cases.png")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(f"{out}: {len(panels)} panels, {sheet.size}")
for t, r, _ in cases:
    print(" -", r["page_id"], "|", t[:60])
