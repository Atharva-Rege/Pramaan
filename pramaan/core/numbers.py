from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

DASHES = {"-", "--", "---", "–", "—", "−", "nil", "Nil", "NIL", "NA", "N.A.", "n.a."}
_MINUS_CHARS = "−–—"

_NUM_TOKEN = re.compile(
    r"(?<![\w.,])"
    r"(?P<open>\()?\s?"
    r"(?P<sign>[-−–])?\s?"
    r"(?P<cur>(?:₹|Rs\.?|INR)\s?)?"
    r"(?P<body>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s?(?P<close>\))?"
    r"(?P<pct>%)?"
    r"(?![\w])"
)

Grouping = Literal["indian", "western", "none"]
NegStyle = Literal["paren", "minus"]


@dataclass(frozen=True)
class NumToken:
    raw: str
    value: float
    decimals: int
    start: int
    end: int
    negative: bool
    digits: str


def decimals_of(body: str) -> int:
    return len(body.split(".", 1)[1]) if "." in body else 0


def parse_number(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s or s in DASHES:
        return None
    neg = False
    s = s.replace("₹", "").replace("INR", "").replace("Rs.", "").replace("Rs", "").strip()
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    elif s.startswith("(") and not s.endswith(")"):
        neg = True
        s = s[1:].strip()
    elif s.endswith(")") and "(" not in s:
        neg = True
        s = s[:-1].strip()
    if s and s[0] in "-" + _MINUS_CHARS:
        neg = True
        s = s[1:].strip()
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()
    s = s.rstrip("%").strip()
    s = s.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", s):
        return None
    v = float(s)
    return -v if neg else v


def is_dash(text: str | None) -> bool:
    return text is not None and text.strip() in DASHES


def find_numbers(text: str) -> list[NumToken]:
    out: list[NumToken] = []
    for m in _NUM_TOKEN.finditer(text):
        body = m.group("body")
        neg = bool(m.group("sign")) or (bool(m.group("open")) and bool(m.group("close")))
        value = float(body.replace(",", ""))
        out.append(
            NumToken(
                raw=m.group(0).strip(),
                value=-value if neg else value,
                decimals=decimals_of(body),
                start=m.start(),
                end=m.end(),
                negative=neg,
                digits=re.sub(r"\D", "", body),
            )
        )
    return out


def round_half_up(value: float, decimals: int) -> float:
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(value)).quantize(q, rounding=ROUND_HALF_UP))


def group_indian(int_part: str) -> str:
    if len(int_part) <= 3:
        return int_part
    head, tail = int_part[:-3], int_part[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def group_western(int_part: str) -> str:
    parts: list[str] = []
    s = int_part
    while len(s) > 3:
        parts.insert(0, s[-3:])
        s = s[:-3]
    if s:
        parts.insert(0, s)
    return ",".join(parts)


def format_number(
    value: float,
    decimals: int = 2,
    grouping: Grouping = "indian",
    neg: NegStyle = "paren",
) -> str:
    r = round_half_up(abs(value), decimals)
    body = f"{r:.{decimals}f}"
    int_part, _, frac = body.partition(".")
    if grouping == "indian":
        int_part = group_indian(int_part)
    elif grouping == "western":
        int_part = group_western(int_part)
    s = int_part + ("." + frac if frac else "")
    if value < 0 and r != 0:
        return f"({s})" if neg == "paren" else f"-{s}"
    return s


def render_variants(
    value_inr: float,
    unit_candidates: list[float],
    decimals: list[int],
) -> dict[float, set[str]]:
    out: dict[float, set[str]] = {}
    for unit in unit_candidates:
        d = value_inr / unit
        forms: set[str] = set()
        for dec in decimals:
            groupings: tuple[Grouping, ...] = ("indian", "western", "none")
            negs: tuple[NegStyle, ...] = ("paren", "minus")
            for grouping in groupings:
                for neg in negs:
                    forms.add(format_number(d, dec, grouping, neg))
        out[unit] = forms
    return out


def matches_printed(displayed: float, token_value: float, token_decimals: int, slack_ulps: float = 0.0) -> bool:
    tol = 0.5 * 10 ** (-token_decimals) * (1 + 2 * slack_ulps) + 1e-9
    return bool(abs(displayed - token_value) <= tol)


def to_crore(value: float, unit_factor_crore: float) -> float:
    return value * unit_factor_crore


def inr_to_crore(value_inr: float) -> float:
    return value_inr / 1e7


def page_value_to_inr(value: float, unit_inr: float) -> float:
    return value * unit_inr
