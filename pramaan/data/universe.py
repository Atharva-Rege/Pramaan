from __future__ import annotations

import csv
import io
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ListedCompany:
    symbol: str
    name: str
    isin: str
    industry: str = ""
    in_nifty200: bool = False


def parse_nifty200_csv(text: str) -> list[ListedCompany]:
    out: list[ListedCompany] = []
    for row in csv.DictReader(io.StringIO(text)):
        sym = (row.get("Symbol") or "").strip()
        if not sym:
            continue
        out.append(
            ListedCompany(
                symbol=sym,
                name=(row.get("Company Name") or "").strip(),
                isin=(row.get("ISIN Code") or "").strip(),
                industry=(row.get("Industry") or "").strip(),
                in_nifty200=True,
            )
        )
    return out


def parse_equity_list_csv(text: str) -> list[ListedCompany]:
    out: list[ListedCompany] = []
    reader = csv.reader(io.StringIO(text))
    header = [h.strip().upper() for h in next(reader)]
    idx = {h: i for i, h in enumerate(header)}
    for row in reader:
        if len(row) < len(header):
            continue
        series = row[idx.get("SERIES", 2)].strip()
        if series not in {"EQ", "BE", "SM"}:
            continue
        out.append(
            ListedCompany(
                symbol=row[idx["SYMBOL"]].strip(),
                name=row[idx["NAME OF COMPANY"]].strip(),
                isin=row[idx["ISIN NUMBER"]].strip(),
            )
        )
    return out


def name_excluded(name: str, patterns: list[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def sample_others(
    all_listed: list[ListedCompany],
    nifty: list[ListedCompany],
    seed: int,
    exclude_patterns: list[str],
) -> list[ListedCompany]:
    nifty_syms = {c.symbol for c in nifty}
    pool = [
        c
        for c in all_listed
        if c.symbol not in nifty_syms and not name_excluded(c.name, exclude_patterns)
    ]
    pool.sort(key=lambda c: c.symbol)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool
