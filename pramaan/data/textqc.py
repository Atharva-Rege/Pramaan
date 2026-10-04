from __future__ import annotations

import re
from dataclasses import asdict, dataclass

VOCAB = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the this to was were will with
    not no all any may shall per under over than other which who whom their these those also such been being
    particulars quarter quarterly ended year half months month period results result financial statement statements
    audited unaudited standalone consolidated revenue operations income total expenses expense profit loss tax
    current deferred before after exceptional items item net comprehensive other equity share shares capital paid
    face value earnings basic diluted cost materials consumed purchases stock trade inventories employee benefits
    finance costs depreciation amortisation amortization segment segments assets liabilities company limited
    board directors meeting regulation regulations sebi listing obligations disclosure requirements exchange
    review report auditors auditor chartered accountants partner membership registration firm date place signed
    notes note refer figures previous reclassified regrouped comparable crores crore lakhs lakh million rupees
    inr rs mumbai delhi bengaluru chennai pune kolkata hyderabad india indian office registered email website
    chairman managing director whole time chief officer secretary compliance approved considered approved
    consolidated subsidiaries associates joint ventures group parent interim reporting accounting standard
    standards companies act rules amended thereunder prescribed section recognised principles practices
    accordance management responsibility conclusion basis scope procedures applied analytical inquiries
    persons responsible matters nothing came attention believe accompanying statement prepared
    """.split()
)

_PUA = re.compile(r"[-�￾￿]")
_CID = re.compile(r"\(cid:\d+\)")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NOISE_PUNCT = re.compile(r"[!~^|\\<>{}·•¨´`‘’]")
_MIXED = re.compile(r"(?=.*\d)(?=.*[A-Za-z])")
_CLEAN_NUM = re.compile(r"^\(?[-−]?\d[\d,]*(?:\.\d+)?\)?%?$")
_DATEISH = re.compile(r"^\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}$")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5,}")


@dataclass
class TextQuality:
    n_chars: int
    n_tokens: int
    pua_share: float
    vocab_rate: float
    wordlike_rate: float
    digit_clean_rate: float
    noise_punct_rate: float
    score: float
    usable: bool
    reason: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def _wordlike(tok: str) -> bool:
    if not tok.isalpha():
        return False
    if len(tok) <= 2:
        return True
    if not _VOWEL.search(tok):
        return tok.isupper() and len(tok) <= 5
    if _CONSONANT_RUN.search(tok):
        return False
    if tok.islower() or tok.isupper() or (tok[0].isupper() and tok[1:].islower()):
        return True
    return False


def assess_text(text: str, min_chars: int = 40, min_score: float = 0.62) -> TextQuality:
    n_chars = len(text.strip())
    if n_chars < min_chars:
        return TextQuality(n_chars, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "no_text")
    tokens = text.split()
    n = max(1, len(tokens))
    pua = (len(_PUA.findall(text)) + 5 * len(_CID.findall(text)) + len(_CTRL.findall(text))) / max(1, n_chars)
    alpha = [t.strip(".,;:()[]\"'") for t in tokens]
    alpha = [t for t in alpha if len(t) >= 3 and t.isalpha()]
    vocab_rate = sum(1 for t in alpha if t.lower() in VOCAB) / max(1, len(alpha))
    wordlike_rate = sum(1 for t in alpha if _wordlike(t)) / max(1, len(alpha))
    digit_toks = [t for t in tokens if any(c.isdigit() for c in t)]
    if digit_toks:
        clean = sum(1 for t in digit_toks if _CLEAN_NUM.match(t) or _DATEISH.match(t) or t.isdigit())
        mixed_bad = sum(1 for t in digit_toks if _MIXED.search(t) and not re.match(r"^(?:\d+(?:st|nd|rd|th)|[A-Za-z]{1,3}\.?\d+|\d+[A-Za-z]{1,3}\)?)$", t))
        digit_clean = max(0.0, (clean - mixed_bad)) / len(digit_toks)
    else:
        digit_clean = 1.0
    noise_rate = len(_NOISE_PUNCT.findall(text)) / max(1, n_chars) * 25
    score = (
        0.35 * wordlike_rate
        + 0.20 * min(1.0, vocab_rate / 0.35)
        + 0.25 * digit_clean
        + 0.20 * (1.0 - min(1.0, noise_rate))
    )
    score *= 1.0 - min(1.0, pua * 20)
    reason = "ok"
    usable = score >= min_score and pua < 0.02
    if pua >= 0.02:
        reason = "unmapped_glyphs"
    elif score < min_score:
        reason = "low_plausibility"
    return TextQuality(
        n_chars=n_chars,
        n_tokens=len(tokens),
        pua_share=round(pua, 4),
        vocab_rate=round(vocab_rate, 3),
        wordlike_rate=round(wordlike_rate, 3),
        digit_clean_rate=round(digit_clean, 3),
        noise_punct_rate=round(noise_rate, 3),
        score=round(score, 3),
        usable=usable,
        reason=reason,
    )
