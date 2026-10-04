from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def to_list(self) -> list[float | str]:
        return [self.text, round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]

    @classmethod
    def from_list(cls, v: list[float | str]) -> Word:
        return cls(str(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]))


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def cy(self) -> float:
        return sum(w.cy for w in self.words) / len(self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)


def group_lines(words: list[Word]) -> list[Line]:
    if not words:
        return []
    hs = [w.h for w in words if w.h > 0]
    med_h = median(hs) if hs else 8.0
    tol = max(2.0, 0.5 * med_h)
    ordered = sorted(words, key=lambda w: (w.cy, w.x0))
    lines: list[Line] = []
    for w in ordered:
        placed = False
        for ln in reversed(lines[-3:]):
            if abs(ln.cy - w.cy) <= tol:
                ln.words.append(w)
                placed = True
                break
        if not placed:
            lines.append(Line([w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    lines.sort(key=lambda ln: ln.cy)
    return lines


def words_to_layout_text(words: list[Word], page_width: float, pts_per_char: float = 5.0) -> str:
    lines = group_lines(words)
    out: list[str] = []
    prev_y1: float | None = None
    med_h = median([w.h for w in words if w.h > 0]) if words else 8.0
    for ln in lines:
        if prev_y1 is not None and ln.y0 - prev_y1 > 1.2 * med_h:
            out.append("")
        buf = ""
        for w in ln.words:
            col = int(round(w.x0 / pts_per_char))
            if len(buf) < col:
                buf += " " * (col - len(buf))
            elif buf:
                buf += " "
            buf += w.text
        out.append(buf.rstrip())
        prev_y1 = ln.y1
    return "\n".join(out)


def plain_text(words: list[Word]) -> str:
    return "\n".join(ln.text for ln in group_lines(words))


_NUMISH_END = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".", "(")
_NUMISH_START = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".", ")")


def _numeric_fragment(a: str, b: str) -> bool:
    if not a or not b:
        return False
    core = set("0123456789,.()-")
    if not (set(a) <= core and set(b) <= core):
        return False
    if a.endswith(")") or b.startswith("("):
        return False
    if a[-1] in _NUMISH_END and b[0] in _NUMISH_START:
        if a[-1].isdigit() and b[0].isdigit():
            return len(a.replace(",", "").replace(".", "").replace("(", "").replace("-", "")) <= 2 and "." not in a
        return True
    return False


def merge_fragments(words: list[Word], max_gap: float = 1.6) -> list[Word]:
    out: list[Word] = []
    for line in group_lines(words):
        cur: Word | None = None
        for w in line.words:
            if cur is not None and w.x0 - cur.x1 <= max_gap and _numeric_fragment(cur.text, w.text):
                cur = Word(cur.text + w.text, cur.x0, min(cur.y0, w.y0), w.x1, max(cur.y1, w.y1))
            else:
                if cur is not None:
                    out.append(cur)
                cur = w
        if cur is not None:
            out.append(cur)
    return out
