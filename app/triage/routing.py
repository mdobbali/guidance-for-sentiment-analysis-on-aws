"""Deterministic routing. The model classifies; this code decides where a record
goes and whether a person must look at it.

Design rule: never route down on weak evidence. A flag the model raised without a
verifiable quote is kept and sent to human review. Over-escalating costs reviewer
time; under-escalating a safety signal costs much more.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .classify import Classification

# Safety net: words that suggest a medical event. If the model says "no adverse
# event" but the record contains one of these, a person checks it.
_SAFETY_TERMS = re.compile(
    r"(?-i:\bER\b)|\b(died|death|dead|passed away|hospital\w*|admitted|emergency|ICU|urgent care|ambulance|"
    r"seizure\w*|faint\w*|unconscious|anaphyla\w*|swell\w*|rash|hives|bleed\w*|vomit\w*|"
    r"overdos\w*|poison\w*|swallowed|pregnan\w*|allergic|reaction|side effect\w*|injur\w*|burn\w*|"
    r"headache\w*|nause\w*|dizz\w*|chest pain|"
    r"short(?:ness)? of breath|can'?t breathe)\b",
    re.IGNORECASE,
)


@dataclass
class Route:
    queue: str  # pharmacovigilance | complaint_handling | customer_service | manual_triage
    priority: str  # P1 (highest) .. P4
    needs_human_review: bool
    reasons: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def route(c: Classification, redacted_text: str) -> Route:
    if c.status != "ok":
        return Route("manual_triage", "P1", True, [f"classification {c.status}; triage manually"])

    d = c.data
    ae = d["adverse_event"]["assessment"] in ("yes", "possible")
    serious = [k for k, v in d["seriousness"].items() if v]
    reasons: list[str] = []
    follow_up: list[str] = []
    review = False

    if ae and serious:
        queue, priority = "pharmacovigilance", "P1"
        reasons.append(f"possible serious adverse event ({', '.join(serious)})")
    elif ae:
        queue, priority = "pharmacovigilance", "P2"
        reasons.append(f"adverse event assessment: {d['adverse_event']['assessment']}")
    elif d["device_malfunction"]:
        queue, priority = "complaint_handling", "P2"
        reasons.append("device malfunction: assess MDR reportability (21 CFR 803)")
    elif d["category"] == "counterfeit_tampering":
        queue, priority = "complaint_handling", "P2"
        reasons.append("suspected counterfeit or tampering")
    elif d["is_complaint"]:
        queue, priority = "complaint_handling", "P3"
        reasons.append(f"product complaint ({d['category']})")
    else:
        queue, priority = "customer_service", "P4"
        reasons.append(f"not a complaint ({d['category']})")

    if ae:
        review = True
        missing = [k for k, v in d["icsr_criteria"].items() if not v]
        if missing:
            follow_up.append("collect missing ICSR minimum criteria: " + ", ".join(missing))
    if d["device_malfunction"]:
        review = True

    verified_fields = {e.field for e in c.evidence if e.verified}
    unsupported = [f for f in c.flags() if f not in verified_fields]
    if unsupported:
        review = True
        reasons.append("flag(s) without a verified quote, kept for review: " + ", ".join(unsupported))

    if not ae and _SAFETY_TERMS.search(redacted_text):
        term = _SAFETY_TERMS.search(redacted_text).group(0)
        review = True
        reasons.append(f"safety-term check: record mentions '{term}' but no adverse event was flagged")

    return Route(queue, priority, review, reasons, follow_up)
