from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from pramaan.core.config import sources_config
from pramaan.core.ids import sha256_bytes
from pramaan.core.paths import Paths, get_paths
from pramaan.core.periods import parse_dates
from pramaan.core.timeutil import (
    iso,
    parse_date_flexible,
    parse_ist_timestamp,
    parse_iso,
    utc_to_ist,
    utcnow,
)
from pramaan.data.http import Blocked, PoliteClient
from pramaan.data.universe import (
    ListedCompany,
    parse_equity_list_csv,
    parse_nifty200_csv,
    sample_others,
)
from pramaan.data.xbrl import XbrlError, has_core_items, parse_xbrl

log = logging.getLogger("pramaan.collector")

NSE_EARLIEST_INTEGRATED = date(2025, 4, 1)


@dataclass
class XbrlRecord:
    seq_id: str
    symbol: str
    name: str
    period_end: date
    scope: str | None
    audited: str | None
    url: str
    created_at: datetime | None
    broadcast_at: datetime | None
    type_sub: str
    revised_at: datetime | None = None
    remark: str | None = None


@dataclass
class Announcement:
    symbol: str
    desc: str
    text: str
    file_url: str
    file_size: str
    disseminated_at: datetime | None
    seq_id: str
    isin: str
    name: str


def _clean_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or v.upper() in {"NA", "N/A", "NULL", "-"}:
        return None
    tail = v.rstrip("/").rsplit("/", 1)[-1]
    if tail in {"-", "", "null"}:
        return None
    return v


def parse_integrated_row(row: dict[str, Any], cfg: dict[str, Any]) -> XbrlRecord | None:
    if (row.get("type") or "").strip() != cfg["financial_type"]:
        return None
    url = _clean_url(row.get("xbrl"))
    if not url or cfg["xbrl_marker"] not in url:
        return None
    pe = parse_date_flexible(row.get("qe_Date"))
    if pe is None:
        return None
    scope_raw = (row.get("consolidated") or "").strip().lower()
    scope = "consolidated" if scope_raw.startswith("consol") else ("standalone" if scope_raw.startswith("stand") else None)
    return XbrlRecord(
        seq_id=str(row.get("seq_Id") or ""),
        symbol=str(row.get("symbol") or "").strip().upper(),
        name=str(row.get("cmName") or row.get("smName") or "").strip(),
        period_end=pe,
        scope=scope,
        audited=row.get("audited"),
        url=url,
        created_at=parse_ist_timestamp(row.get("creation_Date")),
        broadcast_at=parse_ist_timestamp(row.get("broadcast_Date")),
        type_sub=str(row.get("type_Sub") or ""),
        revised_at=parse_ist_timestamp(row.get("revised_Date")),
        remark=row.get("revision_Remark"),
    )


def parse_announcement_row(row: dict[str, Any]) -> Announcement:
    return Announcement(
        symbol=str(row.get("symbol") or "").strip().upper(),
        desc=str(row.get("desc") or "").strip(),
        text=str(row.get("attchmntText") or ""),
        file_url=str(row.get("attchmntFile") or "").strip(),
        file_size=str(row.get("attFileSize") or ""),
        disseminated_at=parse_ist_timestamp(row.get("exchdisstime")) or parse_ist_timestamp(row.get("an_dt")),
        seq_id=str(row.get("seq_id") or ""),
        isin=str(row.get("sm_isin") or ""),
        name=str(row.get("sm_name") or ""),
    )


def pick_outcome(
    anns: list[Announcement],
    period_end: date,
    xbrl_created: datetime | None,
    outcome_descs: list[str],
) -> Announcement | None:
    best: tuple[float, Announcement] | None = None
    for a in anns:
        if not a.file_url.lower().endswith(".pdf"):
            continue
        text = a.text.lower()
        score = 0.0
        if a.desc in outcome_descs:
            score += 2
        if "financial result" in text or "results" in text:
            score += 3
        else:
            continue
        if "intimation" in text and "submitted" not in text:
            score -= 3
        dates = [d.value for d in parse_dates(a.text)]
        if period_end in dates:
            score += 3
        elif dates:
            score -= 2
        if xbrl_created and a.disseminated_at:
            gap_h = abs((a.disseminated_at - xbrl_created).total_seconds()) / 3600.0
            score += max(0.0, 2.0 - gap_h / 12.0)
        if best is None or score > best[0]:
            best = (score, a)
    if best is None or best[0] < 4:
        return None
    return best[1]


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def build_client(cfg: dict[str, Any] | None = None) -> PoliteClient:
    cfg = cfg or sources_config()
    return PoliteClient(
        header_sets={
            "nseindia": cfg["nse"]["headers"],
            "bseindia": cfg["bse"]["headers"],
        },
        min_interval={
            "nseindia": float(cfg["nse"]["min_interval_s"]),
            "bseindia": float(cfg["bse"]["min_interval_s"]),
            "default": 2.0,
        },
        timeout=float(cfg["nse"]["timeout_s"]),
    )


class NseSource:
    def __init__(self, client: PoliteClient, cfg: dict[str, Any] | None = None) -> None:
        self.client = client
        self.cfg = (cfg or sources_config())["nse"]

    def integrated_for_symbol(self, symbol: str, start: date, end: date) -> list[XbrlRecord]:
        url = self.cfg["base"] + self.cfg["integrated_results"]
        out: list[XbrlRecord] = []
        seen: set[str] = set()
        page = 1
        total: int | None = None
        while page < 200:
            res = self.client.get(
                url,
                params={
                    "index": "equities",
                    "symbol": symbol,
                    "from_date": start.strftime("%d-%m-%Y"),
                    "to_date": end.strftime("%d-%m-%Y"),
                    "page": page,
                },
            )
            if res.status != 200:
                raise RuntimeError(f"integrated {symbol} HTTP {res.status}")
            doc = res.json()
            rows = doc.get("data") or []
            total = doc.get("totalCount", total)
            fresh = 0
            for row in rows:
                key = str(row.get("seq_Id"))
                if key in seen:
                    continue
                seen.add(key)
                fresh += 1
                rec = parse_integrated_row(row, self.cfg)
                if rec:
                    out.append(rec)
            if not rows or fresh == 0 or (total is not None and len(seen) >= total):
                break
            page += 1
        return out

    def announcements(self, symbol: str, start: date, end: date) -> list[Announcement]:
        url = self.cfg["base"] + self.cfg["announcements"]
        res = self.client.get(
            url,
            params={
                "index": "equities",
                "symbol": symbol,
                "from_date": start.strftime("%d-%m-%Y"),
                "to_date": end.strftime("%d-%m-%Y"),
            },
        )
        if res.status != 200:
            raise RuntimeError(f"announcements {symbol} HTTP {res.status}")
        doc = res.json()
        rows = doc if isinstance(doc, list) else doc.get("data", [])
        return [parse_announcement_row(r) for r in rows]

    def download(self, url: str) -> bytes | None:
        res = self.client.get(url, accept_statuses=(200, 404))
        if res.status != 200:
            return None
        return res.content

    def text(self, url: str) -> str:
        res = self.client.get(url)
        return res.content.decode("utf-8", errors="replace")


class BseSource:
    def __init__(self, client: PoliteClient, cfg: dict[str, Any] | None = None) -> None:
        self.client = client
        self.cfg = (cfg or sources_config())["bse"]

    def result_pdfs(self, scrip_code: str, start: date, end: date) -> list[dict[str, Any]]:
        url = self.cfg["api"] + self.cfg["announcements"]
        out: list[dict[str, Any]] = []
        page = 1
        while page < 20:
            res = self.client.get(
                url,
                params={
                    "pageno": page,
                    "strCat": "Result",
                    "strPrevDate": start.strftime("%Y%m%d"),
                    "strScrip": scrip_code,
                    "strSearch": "P",
                    "strToDate": end.strftime("%Y%m%d"),
                    "strType": "C",
                    "subcategory": "-1",
                },
            )
            if res.status != 200:
                break
            doc = res.json()
            rows = doc.get("Table") or []
            out.extend(rows)
            total = (doc.get("Table1") or [{}])[0].get("ROWCNT", 0)
            if not rows or len(out) >= int(total or 0):
                break
            page += 1
        return out

    def pdf_url(self, attachment_name: str) -> str:
        return f"{self.cfg['attachment_base']}/{attachment_name}"

    def download(self, url: str) -> bytes | None:
        res = self.client.get(url, accept_statuses=(200, 404))
        return res.content if res.status == 200 else None


@dataclass
class FileRef:
    url: str
    path: str
    sha256: str
    bytes: int
    fetched_at: str


@dataclass
class XbrlFile:
    scope: str | None
    seq_id: str
    type_sub: str
    url: str
    path: str
    sha256: str
    created_at: str | None
    broadcast_at: str | None
    revised_at: str | None
    remark: str | None
    audited: str | None


@dataclass
class FilingMeta:
    filing_id: str
    exchange: str
    symbol: str
    isin: str | None
    company: str
    scrip_code: str | None
    period_end: str
    role: str
    board_meeting_date: str | None = None
    pdf: dict[str, Any] | None = None
    pdf_published_at: str | None = None
    pdf_source: str | None = None
    pdf_text: str | None = None
    xbrl: list[XbrlFile] = field(default_factory=list)
    xbrl_published_at: str | None = None
    status: str = "xbrl_only"
    in_nifty200: bool = False
    industry: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def filing_dir(paths: Paths, meta_date: date, fid: str, exchange: str = "nse") -> Path:
    return paths.raw / exchange / meta_date.isoformat() / fid


def load_meta(path: Path) -> FilingMeta | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    d["xbrl"] = [XbrlFile(**x) for x in d.get("xbrl", [])]
    return FilingMeta(**d)


def all_meta_paths(paths: Paths) -> list[Path]:
    return sorted(paths.raw.glob("*/*/*/meta.json"))


def is_pdf(data: bytes) -> bool:
    if not (data[:5] == b"%PDF-" or b"%PDF-" in data[:1024]):
        return False
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(data)
        n = len(doc)
        doc.close()
        return n > 0
    except Exception:
        return False


class Collector:
    def __init__(
        self,
        client: PoliteClient | None = None,
        paths: Paths | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or sources_config()
        self.paths = (paths or get_paths()).ensure()
        self.client = client or build_client(self.cfg)
        self.nse = NseSource(self.client, self.cfg)
        self.bse = BseSource(self.client, self.cfg)
        self.manifest = Manifest(self.paths.manifest)

    def universe(self, other_n: int | None = None, nifty_limit: int | None = None) -> list[ListedCompany]:
        ucfg = self.cfg["universe"]
        cache = self.paths.work / "data" / "catalog"
        cache.mkdir(parents=True, exist_ok=True)
        n200_path, eq_path = cache / "nifty200.csv", cache / "equity_l.csv"
        if not n200_path.exists():
            n200_path.write_text(self.nse.text(self.cfg["nse"]["nifty200_csv"]), encoding="utf-8")
        if not eq_path.exists():
            eq_url = self.cfg["nse"]["archives"] + "/content/equities/EQUITY_L.csv"
            eq_path.write_text(self.nse.text(eq_url), encoding="utf-8")
        nifty = parse_nifty200_csv(n200_path.read_text(encoding="utf-8"))
        listed = parse_equity_list_csv(eq_path.read_text(encoding="utf-8"))
        pool = sample_others(listed, nifty, int(ucfg["sample_seed"]), list(ucfg["exclude_name_patterns"]))
        if nifty_limit:
            nifty = nifty[:nifty_limit]
        n = other_n if other_n is not None else int(ucfg["other_sample_size"])
        return nifty + pool[: n * 3]

    def _store_file(self, directory: Path, name: str, data: bytes, url: str) -> FileRef:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        target.write_bytes(data)
        ref = FileRef(url, str(target.relative_to(self.paths.work)), sha256_bytes(data), len(data), iso(utcnow()) or "")
        self.manifest.append({"kind": name.rsplit(".", 1)[-1], **asdict(ref)})
        return ref

    def collect_company(
        self,
        company: ListedCompany,
        periods: dict[date, str],
        pdf_roles: set[str],
        today: date | None = None,
    ) -> list[FilingMeta]:
        today = today or utcnow().date()
        recs = self.nse.integrated_for_symbol(company.symbol, NSE_EARLIEST_INTEGRATED, today)
        out: list[FilingMeta] = []
        for period_end, role in periods.items():
            pr = [r for r in recs if r.period_end == period_end and r.scope is not None]
            if not pr:
                continue
            out.append(self._collect_period(company, period_end, role, pr, role in pdf_roles))
        return out

    def _collect_period(
        self,
        company: ListedCompany,
        period_end: date,
        role: str,
        records: list[XbrlRecord],
        want_pdf: bool,
    ) -> FilingMeta:
        fid = f"nse-{company.symbol.lower()}-{period_end.isoformat()}"
        created = min((r.created_at for r in records if r.created_at), default=None)
        anchor = (utc_to_ist(created).date() if created else period_end)
        directory = filing_dir(self.paths, anchor, fid)
        meta_path = directory / "meta.json"
        meta = load_meta(meta_path) or FilingMeta(
            filing_id=fid,
            exchange="nse",
            symbol=company.symbol,
            isin=company.isin or None,
            company=company.name or records[0].name,
            scrip_code=None,
            period_end=period_end.isoformat(),
            role=role,
            in_nifty200=company.in_nifty200,
            industry=company.industry,
        )
        have = {x.seq_id for x in meta.xbrl}
        for r in sorted(records, key=lambda x: (x.created_at or datetime.min, x.seq_id)):
            if r.seq_id in have:
                continue
            data = self.nse.download(r.url)
            if not data or b"<xbrli:xbrl" not in data[:4000] and b"xbrl" not in data[:4000].lower():
                log.warning("skip non-xbrl payload for %s %s", company.symbol, r.url)
                continue
            try:
                parsed = parse_xbrl(data)
            except XbrlError as exc:
                log.warning("bad xbrl %s: %s", r.url, exc)
                continue
            scope = r.scope or parsed.scope
            ref = self._store_file(directory, f"xbrl_{scope}_{r.seq_id}.xml", data, r.url)
            meta.xbrl.append(
                XbrlFile(
                    scope=scope,
                    seq_id=r.seq_id,
                    type_sub=r.type_sub,
                    url=r.url,
                    path=ref.path,
                    sha256=ref.sha256,
                    created_at=iso(r.created_at),
                    broadcast_at=iso(r.broadcast_at),
                    revised_at=iso(r.revised_at),
                    remark=r.remark,
                    audited=r.audited,
                )
            )
            if parsed.meta.get("scrip_code") and not meta.scrip_code:
                meta.scrip_code = parsed.meta["scrip_code"]
            if parsed.meta.get("board_meeting_date") and not meta.board_meeting_date:
                meta.board_meeting_date = parsed.meta["board_meeting_date"].isoformat()
            if parsed.meta.get("isin") and not meta.isin:
                meta.isin = parsed.meta["isin"]
            if not has_core_items(parsed):
                meta.status = "xbrl_not_standard"
        originals = [x.created_at for x in meta.xbrl if x.type_sub.lower() != "revision" and x.created_at]
        meta.xbrl_published_at = min(originals) if originals else min((x.created_at for x in meta.xbrl if x.created_at), default=None)
        if want_pdf and meta.pdf is None and meta.xbrl:
            self._fetch_pdf(meta, company, period_end, directory, parse_iso(meta.xbrl_published_at))
        if meta.pdf is not None:
            meta.status = "complete" if meta.xbrl and meta.status != "xbrl_not_standard" else meta.status
        directory.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta.to_json(), indent=1, default=str), encoding="utf-8")
        return meta

    def _fetch_pdf(
        self,
        meta: FilingMeta,
        company: ListedCompany,
        period_end: date,
        directory: Path,
        xbrl_created: datetime | None,
    ) -> None:
        anchor = xbrl_created or utcnow()
        anns = self.nse.announcements(
            company.symbol, (anchor - timedelta(days=5)).date(), (anchor + timedelta(days=2)).date()
        )
        pick = pick_outcome(anns, period_end, xbrl_created, list(self.cfg["nse"]["outcome_descs"]))
        if pick is not None:
            data = self.nse.download(pick.file_url)
            if data and is_pdf(data):
                ref = self._store_file(directory, "outcome.pdf", data, pick.file_url)
                meta.pdf = asdict(ref)
                meta.pdf_published_at = iso(pick.disseminated_at)
                meta.pdf_source = "nse"
                meta.pdf_text = pick.text
                return
        if meta.scrip_code:
            try:
                rows = self.bse.result_pdfs(meta.scrip_code, (anchor - timedelta(days=5)).date(), (anchor + timedelta(days=2)).date())
            except (RuntimeError, Blocked):
                rows = []
            for row in rows:
                att = str(row.get("ATTACHMENTNAME") or "")
                headline = str(row.get("HEADLINE") or "")
                if not att.lower().endswith(".pdf") or "result" not in headline.lower():
                    continue
                url = self.bse.pdf_url(att)
                data = self.bse.download(url)
                if data and is_pdf(data):
                    ref = self._store_file(directory, "outcome.pdf", data, url)
                    meta.pdf = asdict(ref)
                    meta.pdf_published_at = iso(parse_ist_timestamp(str(row.get("DissemDT") or row.get("NEWS_DT") or "").replace("T", " ").split(".")[0]))
                    meta.pdf_source = "bse"
                    meta.pdf_text = headline
                    return

    def run(
        self,
        target_period: date,
        prior_year_period: date,
        previous_quarter: date,
        companies: list[ListedCompany],
        want_filers: int,
        reference_pdf: bool = True,
        progress: Any = None,
    ) -> dict[str, int]:
        periods = {
            target_period: "target",
            prior_year_period: "prior_year",
            previous_quarter: "previous_quarter",
        }
        pdf_roles = {"target"} | ({"previous_quarter"} if reference_pdf else set())
        got = 0
        stats = {"companies_tried": 0, "filers": 0, "complete": 0, "no_pdf": 0, "errors": 0}
        for c in companies:
            if got >= want_filers:
                break
            stats["companies_tried"] += 1
            try:
                metas = self.collect_company(c, periods, pdf_roles)
            except Blocked:
                raise
            except Exception as exc:
                log.exception("company %s failed: %s", c.symbol, exc)
                stats["errors"] += 1
                continue
            tgt = [m for m in metas if m.role == "target"]
            if tgt and tgt[0].status != "xbrl_not_standard":
                got += 1
                stats["filers"] += 1
                if tgt[0].pdf:
                    stats["complete"] += 1
                else:
                    stats["no_pdf"] += 1
            if progress:
                progress(c.symbol, stats)
        return stats


_Q_TO_PREV = {6: (3, 31), 9: (6, 30), 12: (9, 30), 3: (12, 31)}


def previous_quarter_end(period_end: date) -> date:
    m, d = _Q_TO_PREV[period_end.month]
    y = period_end.year - 1 if period_end.month == 3 else period_end.year
    return date(y, m, d)


def prior_year_end(period_end: date) -> date:
    return date(period_end.year - 1, period_end.month, period_end.day)


def slugify_company(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def restore_pdfs(paths: Paths | None = None, echo: Any = print) -> dict[str, int]:
    paths = (paths or get_paths()).ensure()
    client = build_client()
    done = missing = failed = 0
    for mp in all_meta_paths(paths):
        meta = load_meta(mp)
        if meta is None or not meta.pdf:
            continue
        target = paths.work / meta.pdf["path"]
        if target.exists():
            continue
        missing += 1
        res = client.get(meta.pdf["url"], accept_statuses=(200, 404))
        if res.status == 200 and is_pdf(res.content):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(res.content)
            if sha256_bytes(res.content) != meta.pdf.get("sha256"):
                echo(f"hash differs for {meta.filing_id}: the exchange replaced the document")
            done += 1
        else:
            failed += 1
    return {"missing": missing, "restored": done, "failed": failed}
