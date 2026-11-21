"""PHI redaction before anything is stored, embedded or sent to a model.

Redacted spans are replaced with a type tag such as [NAME] or [AGE], so the
classifier can still tell that an identifiable patient or reporter exists
(an ICSR minimum criterion) without seeing who they are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Span:
    begin: int
    end: int
    type: str
    score: float = 1.0


@dataclass
class Redaction:
    text: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    redactor: str = ""


class Redactor(Protocol):
    name: str

    def redact(self, text: str) -> Redaction: ...


def apply_spans(text: str, spans: list[Span], redactor: str) -> Redaction:
    counts: dict[str, int] = {}
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: (s.begin, -s.end)):
        if s.begin < pos:  # overlapping span already covered
            continue
        out.append(text[pos : s.begin])
        out.append(f"[{s.type}]")
        counts[s.type] = counts.get(s.type, 0) + 1
        pos = s.end
    out.append(text[pos:])
    return Redaction(text="".join(out), entity_counts=counts, redactor=redactor)


class ComprehendMedicalRedactor:
    """Amazon Comprehend Medical DetectPHI (HIPAA PHI types)."""

    name = "comprehend-medical"
    MAX_BYTES = 20_000  # DetectPHI request limit (UTF-8 bytes)

    def __init__(self, client, min_score: float = 0.5):
        self._client = client
        self._min_score = min_score

    def _windows(self, text: str) -> list[tuple[int, str]]:
        if len(text.encode()) <= self.MAX_BYTES:
            return [(0, text)]
        windows, start = [], 0
        while start < len(text):
            end = start + self.MAX_BYTES // 4  # stays under the byte limit for any UTF-8 text
            if end < len(text):
                cut = text.rfind(" ", start, end)
                end = cut if cut > start else end
            windows.append((start, text[start:end]))
            start = end
        return windows

    def redact(self, text: str) -> Redaction:
        spans = []
        for offset, window in self._windows(text):
            resp = self._client.detect_phi(Text=window)
            spans += [
                Span(offset + e["BeginOffset"], offset + e["EndOffset"], e["Type"], e["Score"])
                for e in resp["Entities"]
                if e["Score"] >= self._min_score
            ]
        return apply_spans(text, spans, self.name)


class RegexRedactor:
    """Pattern-based fallback for offline demos and tests. It catches emails,
    phone numbers, dates and obvious IDs, not names or addresses: never rely on
    it for real PHI."""

    name = "regex"
    _PATTERNS = [
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
        ("PHONE_OR_FAX", re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
        ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
        ("ID", re.compile(r"\b(?:MRN|SSN|Patient ID)[:#\s]*[\w-]{4,}\b", re.IGNORECASE)),
        ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")),
    ]

    def redact(self, text: str) -> Redaction:
        spans = [Span(m.start(), m.end(), kind) for kind, p in self._PATTERNS for m in p.finditer(text)]
        return apply_spans(text, spans, self.name)
