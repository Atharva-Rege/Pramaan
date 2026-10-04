from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

IST_OFFSET = timedelta(minutes=330)

_NSE_TS_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d%m%Y%H%M%S",
)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def ist_to_utc(dt: datetime) -> datetime:
    return dt - IST_OFFSET


def utc_to_ist(dt: datetime) -> datetime:
    return dt + IST_OFFSET


def parse_ist_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text or text in {"-", "null", "None"}:
        return None
    for fmt in _NSE_TS_FORMATS:
        try:
            return ist_to_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def parse_date_flexible(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if not text or text in {"-", "null", "None"}:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        return None


def months_between(start: date, end: date) -> int:
    days = (end - start).days + 1
    return max(1, round(days / 30.4375))


def add_months_end(end: date, months: int) -> date:
    y = end.year + (end.month - 1 + months) // 12
    m = (end.month - 1 + months) % 12 + 1
    nxt = date(y + (m // 12), m % 12 + 1, 1)
    return nxt - timedelta(days=1)
