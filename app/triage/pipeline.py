"""redact -> classify -> route -> store -> audit, for one record at a time."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass

from . import __version__
from .audit import AuditLog
from .classify import PROMPT_SHA256, SERIOUSNESS, classify
from .redact import Redactor
from .routing import route

QUEUES = ("pharmacovigilance", "complaint_handling", "customer_service", "manual_triage")
PRIORITIES = ("P1", "P2", "P3", "P4")


@dataclass
class Record:
    case_id: str
    text: str
    received_at: str | None = None
    channel: str | None = None
    product: str | None = None


class TriagePipeline:
    def __init__(self, redactor: Redactor, generator, embedder, store, audit: AuditLog, effort=None):
        self.redactor = redactor
        self.generator = generator
        self.embedder = embedder
        self.store = store
        self.audit = audit
        self.effort = effort

    def process(self, rec: Record, *, tenant_id: str, user: str) -> dict:
        if self.store.exists(tenant_id, rec.case_id):
            return {"case_id": rec.case_id, "skipped": True}
        started = time.perf_counter()
        source_sha = hashlib.sha256(rec.text.encode()).hexdigest()
        red = self.redactor.redact(rec.text)
        c = classify(self.generator, red.text, rec.channel, rec.product)
        r = route(c, red.text)
        case = {
            "case_id": rec.case_id,
            "received_at": rec.received_at,
            "channel": rec.channel,
            "product": rec.product,
            "source_sha256": source_sha,
            "redacted_text": red.text,
            "redaction": {"redactor": red.redactor, "entities": red.entity_counts},
            "classification": {
                "status": c.status,
                **c.data,
                "evidence": [asdict(e) for e in c.evidence],
            },
            "route": r.to_dict(),
            "model": c.model,
            "prompt_sha256": PROMPT_SHA256,
        }
        [embedding] = self.embedder.embed([red.text]) if self.embedder else [None]
        self.store.save_case(tenant_id, case, embedding)
        record = self.audit.append(
            {
                "event": "case.triaged",
                "tenant_id": tenant_id,
                "user": user,
                "case_id": rec.case_id,
                "source_sha256": source_sha,
                "redaction": case["redaction"],
                "generation": {
                    "requested_model": self.generator.model_id,
                    "served_model": c.model,
                    "effort": self.effort,
                    "prompt_sha256": PROMPT_SHA256,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                },
                "classification": case["classification"],
                "route": case["route"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "app_version": __version__,
            }
        )
        return {
            **case,
            "audit_seq": record["seq"],
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
        }

    def review(
        self,
        case_id: str,
        *,
        tenant_id: str,
        reviewer: str,
        decision: str,
        queue: str,
        priority: str,
        reason: str,
        adverse_event: str | None = None,
        serious: list[str] | None = None,
    ) -> dict:
        """Record a person's decision. The model's output is kept unchanged next to it."""
        if decision not in ("confirm", "override"):
            raise ValueError("decision must be confirm or override")
        if queue not in QUEUES or priority not in PRIORITIES:
            raise ValueError(f"queue must be one of {QUEUES}; priority one of {PRIORITIES}")
        if decision == "override" and not reason.strip():
            raise ValueError("an override needs a reason")
        if serious and not set(serious) <= set(SERIOUSNESS):
            raise ValueError(f"serious must be a subset of {SERIOUSNESS}")
        case = self.store.get_case(tenant_id, case_id)
        if case is None:
            raise ValueError(f"unknown case {case_id}")
        review = {
            "reviewer": reviewer,
            "decision": decision,
            "queue": queue,
            "priority": priority,
            "adverse_event": adverse_event,
            "serious": serious or [],
            "reason": reason,
            # Mirrors the "meaning" element of a signature manifestation (21 CFR 11.50).
            "meaning": "triage review",
            "model_route": case["route"],
        }
        self.store.record_review(tenant_id, case_id, review)
        rec = self.audit.append(
            {"event": "case.reviewed", "tenant_id": tenant_id, "user": reviewer, "case_id": case_id, **review}
        )
        return {**review, "audit_seq": rec["seq"]}
