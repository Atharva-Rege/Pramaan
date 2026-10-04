from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

def month_number(name: str) -> int:
    """Month number from a matched month token. Keyed on the first three letters because
    `_MON` also matches misspellings that appear in real filings (for example "sepember")."""
    return MONTHS[name.lower()[:3]]


_MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

_DMY_NAME = re.compile(
    rf"(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[\s\-./,']*{_MON}\.?(?:[\s\-./,']*(?P<y>\d{{4}})(?!\d)|(?<![\d])[\-./'](?P<y2>\d{{2}})(?!\d))",
    re.I,
)
_MDY_NAME = re.compile(
    rf"{_MON}\.?[\s\-./,']*(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[\s,.\-']*(?P<y>\d{{4}})(?!\d)",
    re.I,
)
_DMY_NUM = re.compile(r"(?<!\d)(?P<d>\d{1,2})[./\-](?P<m>\d{1,2})[./\-](?P<y>\d{4}|\d{2})(?!\d)")
_ISO = re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)")
_MY_NAME = re.compile(rf"(?<![\d\w]){_MON}\.?(?:[\s\-./,']*(?P<y>\d{{4}})|[\-./'](?P<y2>\d{{2}}))(?!\d)", re.I)

DURATION_PATTERNS: list[tuple[re.Pattern[str], int | None]] = [
    (re.compile(r"nine\s*months?(?:\s*period)?\s*ended|9\s*months?(?:\s*period)?\s*ended|nine\s*months", re.I), 9),
    (re.compile(r"half[\s\-]*year(?:ly)?(?:\s*period)?\s*ended|six\s*months?(?:\s*period)?\s*ended|6\s*months?(?:\s*period)?\s*ended|half[\s\-]*year(?:ly)?", re.I), 6),
    (re.compile(r"three\s*months?(?:\s*period)?\s*ended|3\s*months?(?:\s*period)?\s*ended|quarter(?:ly)?\s*ended|quarter\s*ending|for\s+the\s+quarter|quarter", re.I), 3),
    (re.compile(r"twelve\s*months?\s*ended|12\s*months?\s*ended|year\s*ended|year\s*ending|annual|for\s+the\s+year|financial\s+year|\bfy\b", re.I), 12),
    (re.compile(r"year[\s\-]*to[\s\-]*date|cumulative|ytd", re.I), None),
]


@dataclass(frozen=True, order=True)
class PeriodKey:
    end: date
    months: int

    def __str__(self) -> str:
        return f"{self.end.isoformat()}/{self.months}m"

    @classmethod
    def parse(cls, text: str) -> PeriodKey:
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})/(\d{1,2})m", text.strip())
        if not m:
            raise ValueError(f"bad period key {text!r}")
        return cls(date.fromisoformat(m.group(1)), int(m.group(2)))

    @property
    def start(self) -> date:
        y, mth = self.end.year, self.end.month - self.months + 1
        while mth < 1:
            mth += 12
            y -= 1
        return date(y, mth, 1)

    def shift_years(self, years: int) -> PeriodKey:
        y = self.end.year + years
        day = min(self.end.day, calendar.monthrange(y, self.end.month)[1])
        return PeriodKey(date(y, self.end.month, day), self.months)


@dataclass(frozen=True)
class DateMatch:
    value: date
    start: int
    end: int


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _year(y: str) -> int:
    n = int(y)
    return 2000 + n if n < 100 else n


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_dates(text: str) -> list[DateMatch]:
    found: list[DateMatch] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    def add(dt: date | None, a: int, b: int) -> None:
        if dt is not None and free(a, b):
            found.append(DateMatch(dt, a, b))
            taken.append((a, b))

    for m in _ISO.finditer(text):
        add(_safe_date(int(m["y"]), int(m["m"]), int(m["d"])), m.start(), m.end())
    for m in _DMY_NAME.finditer(text):
        yy = m["y"] or m["y2"]
        add(_safe_date(_year(yy), month_number(m["mon"]), int(m["d"])), m.start(), m.end())
    for m in _MDY_NAME.finditer(text):
        add(_safe_date(_year(m["y"]), month_number(m["mon"]), int(m["d"])), m.start(), m.end())
    for m in _DMY_NUM.finditer(text):
        add(_safe_date(_year(m["y"]), int(m["m"]), int(m["d"])), m.start(), m.end())
    for m in _MY_NAME.finditer(text):
        yy = _year(m["y"] or m["y2"])
        if 2000 <= yy <= 2100:
            add(month_end(yy, month_number(m["mon"])), m.start(), m.end())
    found.sort(key=lambda d: d.start)
    return found


def fiscal_ytd_months(end: date) -> int:
    return (end.month - 4) % 12 + 1


def duration_keywords(text: str) -> list[tuple[int, int, int | None]]:
    out: list[tuple[int, int, int | None]] = []
    covered: list[tuple[int, int]] = []
    for pat, months in DURATION_PATTERNS:
        for m in pat.finditer(text):
            if any(not (m.end() <= s or m.start() >= e) for s, e in covered):
                continue
            covered.append((m.start(), m.end()))
            out.append((m.start(), m.end(), months))
    out.sort()
    return out


def fiscal_year_label(end: date) -> str:
    fy = end.year + 1 if end.month >= 4 else end.year
    return f"FY{fy % 100:02d}"


def fiscal_quarter_label(end: date) -> str:
    q = {6: 1, 9: 2, 12: 3, 3: 4}.get(end.month)
    if q is None:
        return f"{end.isoformat()}"
    return f"Q{q} {fiscal_year_label(end)}"


def describe_period(key: PeriodKey) -> str:
    day = f"{key.end.day} {calendar.month_name[key.end.month]} {key.end.year}"
    unit = {3: "quarter", 6: "half year", 9: "nine months", 12: "year"}.get(key.months, f"{key.months} months")
    return f"{unit} ended {day}"


def previous_quarter_end(end: date) -> date:
    m = end.month - 3
    y = end.year
    if m < 1:
        m += 12
        y -= 1
    return month_end(y, m)
