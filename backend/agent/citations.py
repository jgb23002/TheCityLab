"""Check that every number in a research brief is cited to matching ledger evidence.

Rule: each numeric token (ignoring years 1900-2100, the E# inside tags, list or
heading enumerators such as "1.", "1)" and "(1)", the fixed phrase "95% CI",
METHOD_CONSTANT_PATTERNS such as "80 m", ranks 1-10 in a table's "Rank" column, and
the body of the "## Question" section, which restates the user's question)
must sit in a sentence (or table row: one unit per row) with an
[E#] tag that exists in the ledger, and must match a number in that evidence's
JSON (its args or result). The governing tag is the first one after the number in
the sentence, else the last one before it. A match is within 1% relative
tolerance, a rounding of the evidence value to the digits written, or either of
those after a x100 / /100 percent conversion. Signs are compared by magnitude.
"""

from __future__ import annotations

import math
import re
from typing import Any

from backend.agent.evidence import Ledger

_TAG = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.I)
_NUMBER = re.compile(r"(?<![\w.])(\$?)(\d+(?:,\d{3})+|\d+)(\.\d+)?(%?)")
# Item enumerators at the start of a line (after optional heading hashes or a bullet):
# "1.", "1)", "(1)". They are numbering, not data.
_LIST_MARKER = re.compile(r"^(\s*(?:#{1,6}\s*)?(?:[-*+]\s+)?)(?:\(\d+\)|\d+[.)])(?=\s|$)")
_FIXED_PHRASES = re.compile(r"\b95\s*%\s*(?:CI|confidence)", re.I)

# Methodology constants that describe how evidence was gathered, not results. Only the
# exact number-plus-unit forms are exempt; a bare "80" is still checked.
METHOD_CONSTANT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "80 m", "80m", "80-m": the ~80 m aerial square vexcel.get_aerial fetches per building.
    (re.compile(r"\b80\s*-?\s*m\b", re.I), "80 m aerial image square"),
    # "80 meter(s)", "80-meter", "80 metre(s)", "80-metre": the same square, unit spelled out.
    (re.compile(r"\b80\s*-?\s*met(?:er|re)s?\b", re.I), "80 meter aerial image square"),
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
_TOLERANCE = 0.01
_EXACT_SIG_DIGITS = 5


_HEADING = re.compile(r"^\s*#{1,6}\s+(.*?)\s*#*\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
# A bare rank 1-10 (optionally "#1") in a table column headed "Rank" is ordering, not data.
_RANK_CELL = re.compile(r"^\s*#?(10|[1-9])\s*$")
_MAX_RANK = 10


def _cells(row: str) -> list[str]:
    """Split a markdown table row into cells (leading/trailing pipes optional)."""
    return row.strip().strip("|").split("|")


def _sentences(brief_md: str) -> list[str]:
    """Split markdown into checkable units: one per table row, else one per sentence.

    The body of a "## Question" section is skipped: it restates the user's question
    (e.g. "Which 10 ... buildings"), which is not a claim from evidence.
    A table row is one unit, so a [E#] tag anywhere in the row covers its numbers.
    In a table whose header has a "Rank" column, a bare rank 1-10 in that column is
    blanked. Other lines are split into sentences (decimals stay intact).
    """
    lines = brief_md.splitlines()
    out: list[str] = []
    rank_col: int | None = None
    in_question = False
    for i, line in enumerate(lines):
        heading = _HEADING.match(line)
        if heading:
            in_question = heading.group(1).strip().lower().rstrip(":") == "question"
        elif in_question:
            continue  # the "## Question" body restates the user's question, not evidence
        if line.lstrip().startswith("|"):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _TABLE_SEPARATOR.match(line):
                continue
            if _TABLE_SEPARATOR.match(nxt):  # header row: find the Rank column, check nothing else
                headers = [c.strip().lower() for c in _cells(line)]
                rank_col = next((j for j, h in enumerate(headers) if h.startswith("rank")), None)
                out.append(line)
                continue
            cells = _cells(line)
            if rank_col is not None and rank_col < len(cells) and _RANK_CELL.match(cells[rank_col]):
                cells[rank_col] = " " * len(cells[rank_col])
            out.append("|" + "|".join(cells) + "|")
            continue
        rank_col = None  # any non-table line ends the table
        line = _LIST_MARKER.sub(r"\1", line)
        out.extend(part for part in _SENTENCE_END.split(line) if part.strip())
    return out


def _evidence_numbers(value: Any) -> list[float]:
    """Collect every number in an evidence entry, including numbers inside strings."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(value) else []
    if isinstance(value, str):
        return [float((m.group(2) + (m.group(3) or "")).replace(",", "")) for m in _NUMBER.finditer(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _evidence_numbers(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _evidence_numbers(v)]
    return []


def _significant_digits(digits: str) -> int:
    """Count significant digits in a written number such as "0.090" or "1950"."""
    stripped = digits.replace(".", "").lstrip("0")
    return len(stripped.rstrip("0")) if "." not in digits else len(stripped)


def _round_sig(value: float, sig: int) -> float:
    """Round *value* to *sig* significant figures."""
    if value == 0:
        return 0.0
    return round(value, sig - 1 - int(math.floor(math.log10(abs(value)))))


def _matches(written: float, decimals: int, sig: int, evidence: list[float]) -> bool:
    """Return True when a written magnitude equals some evidence value as allowed.

    Whole numbers with 5+ significant digits (ids, exact counts) must match exactly:
    a 1% tolerance would let property 990103 pass as 990102.
    """
    if decimals == 0 and sig >= _EXACT_SIG_DIGITS:
        return any(abs(e) == written for e in evidence)
    for scale, extra in ((1.0, 0), (0.01, 2), (100.0, -2)):
        v = written * scale
        d = max(decimals + extra, 0)
        for e in evidence:
            e = abs(e)
            if math.isclose(v, e, rel_tol=_TOLERANCE, abs_tol=1e-12):
                return True
            # Rounded to the decimals shown (bare integers only when >= 10, to avoid
            # "1" matching any value from 0.5 to 1.5).
            if (decimals > 0 or abs(written) >= 10) and math.isclose(round(e, d), v, abs_tol=10 ** -(d + 3)):
                return True
            if sig >= 2 and e and math.isclose(_round_sig(e, sig), v, rel_tol=1e-9, abs_tol=1e-12):
                return True
    return False


def verify_brief(brief_md: str, ledger: Ledger) -> list[str]:
    """Return one readable problem per uncited or unmatched number (empty list = pass)."""
    problems: list[str] = []
    numbers_by_id: dict[str, list[float]] = {}
    for sentence in _sentences(brief_md):
        tags = [(m.start(), [t.strip().upper() for t in re.split(r"[,;]", m.group(1))])
                for m in _TAG.finditer(sentence)]
        for _, ids in tags:
            for eid in ids:
                if ledger.get(eid) is None:
                    problems.append(f"[{eid}] is not in the evidence ledger")
        # Blank out tags and fixed phrases so their digits are never checked.
        scrubbed = _FIXED_PHRASES.sub(lambda m: " " * len(m.group(0)), sentence)
        for pattern, _ in METHOD_CONSTANT_PATTERNS:
            scrubbed = pattern.sub(lambda m: " " * len(m.group(0)), scrubbed)
        scrubbed = _TAG.sub(lambda m: " " * len(m.group(0)), scrubbed)
        for m in _NUMBER.finditer(scrubbed):
            dollar, whole, frac, pct = m.groups()
            token = m.group(0)
            value = float((whole + (frac or "")).replace(",", ""))
            if not dollar and not frac and not pct and 1900 <= value <= 2100:
                continue  # a year
            after = [ids for pos, ids in tags if pos > m.start()]
            before = [ids for pos, ids in tags if pos < m.start()]
            ids = after[0] if after else (before[-1] if before else None)
            if ids is None:
                problems.append(f"{token} has no evidence tag in: \"{sentence.strip()[:100]}\"")
                continue
            known = [eid for eid in ids if ledger.get(eid) is not None]
            if not known:
                continue  # the missing tag was already reported above
            decimals = len(frac) - 1 if frac else 0
            sig = _significant_digits(whole.replace(",", "") + (frac or ""))
            for eid in known:
                if eid not in numbers_by_id:
                    entry = ledger.get(eid) or {}
                    numbers_by_id[eid] = _evidence_numbers({"args": entry.get("args"), "result": entry.get("result")})
            if not any(_matches(value, decimals, sig, numbers_by_id[eid]) for eid in known):
                problems.append(f"{token} not found in {', '.join(known)}")
    return list(dict.fromkeys(problems))
