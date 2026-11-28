from __future__ import annotations

import json

import pytest

from triage.audit import AuditLog
from triage.classify import ICSR, SERIOUSNESS
from triage.llm import Generation


def make_payload(**over) -> dict:
    """A valid classification with everything negative; override what a test needs."""
    p = {
        "category": "inquiry",
        "is_complaint": False,
        "adverse_event": {"assessment": "no", "description": ""},
        "seriousness": dict.fromkeys(SERIOUSNESS, False),
        "icsr_criteria": dict.fromkeys(ICSR, False),
        "device_malfunction": False,
        "product_mentioned": "",
        "sentiment": "neutral",
        "evidence": [],
        "rationale": "",
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


class FakeGenerator:
    model_id = "fake-claude"

    def __init__(self, payload=None, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def generate(self, system, user, schema):
        self.calls.append((system, user, schema))
        payload = self.payload(user) if callable(self.payload) else self.payload
        return Generation(
            text=None if payload is None else json.dumps(payload),
            stop_reason=self.stop_reason,
            model=self.model_id,
            input_tokens=50,
            output_tokens=30,
        )


class FakeEmbedder:
    model_id = "fake-embed"

    def embed(self, texts):
        return [[1.0, 0.0, float(len(t) % 5)] for t in texts]


class FakeStore:
    def __init__(self):
        self.cases = {}

    def exists(self, tenant_id, case_id):
        return (tenant_id, case_id) in self.cases

    def save_case(self, tenant_id, case, embedding):
        assert (tenant_id, case["case_id"]) not in self.cases
        self.cases[(tenant_id, case["case_id"])] = {**case, "status": "open", "review": None}

    def get_case(self, tenant_id, case_id):
        return self.cases.get((tenant_id, case_id))

    def record_review(self, tenant_id, case_id, review):
        c = self.cases[(tenant_id, case_id)]
        if c["status"] != "open":
            raise ValueError("already reviewed")
        c.update(status="reviewed", review=review)


class FakeComprehendMedical:
    """Returns PHI entities for exact substrings, like DetectPHI does."""

    def __init__(self, phi: dict[str, str]):
        self.phi, self.calls = phi, 0

    def detect_phi(self, Text):  # noqa: N803  (boto3 parameter name)
        self.calls += 1
        ents = []
        for value, kind in self.phi.items():
            start = Text.find(value)
            if start >= 0:
                ents.append(
                    {"BeginOffset": start, "EndOffset": start + len(value), "Type": kind, "Score": 0.99}
                )
        return {"Entities": ents}


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit")
