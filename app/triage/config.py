"""Runtime settings, read from environment variables (see app/env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    aws_region: str
    classify_model: str = "anthropic.claude-opus-5-5"
    fallback_model: str = "anthropic.claude-opus-4-8"
    effort: str = "medium"
    embed_model: str = "amazon.titan-embed-text-v2:0"
    embed_dim: int = 1024
    # "comprehend-medical" (default) or "regex" (offline demos only; misses most PHI).
    redactor: str = "comprehend-medical"
    database_url: str | None = None
    db_host: str | None = None
    db_port: int = 5432
    db_name: str = "triage"
    db_secret_arn: str | None = None
    audit_dir: str = "./audit"
    audit_s3_bucket: str | None = None
    tenant_id: str = "demo"
    workers: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        region = _env("AWS_REGION") or _env("AWS_DEFAULT_REGION")
        if not region:
            raise RuntimeError("Set AWS_REGION (for example us-west-2).")
        return cls(
            aws_region=region,
            classify_model=_env("TRIAGE_MODEL", cls.classify_model),
            fallback_model=os.environ.get("TRIAGE_FALLBACK_MODEL", cls.fallback_model),
            effort=_env("TRIAGE_EFFORT", cls.effort),
            embed_model=_env("TRIAGE_EMBED_MODEL", cls.embed_model),
            embed_dim=int(_env("TRIAGE_EMBED_DIM", str(cls.embed_dim))),
            redactor=_env("TRIAGE_REDACTOR", cls.redactor),
            database_url=_env("TRIAGE_DATABASE_URL"),
            db_host=_env("TRIAGE_DB_HOST"),
            db_port=int(_env("TRIAGE_DB_PORT", "5432")),
            db_name=_env("TRIAGE_DB_NAME", cls.db_name),
            db_secret_arn=_env("TRIAGE_DB_SECRET_ARN"),
            audit_dir=_env("TRIAGE_AUDIT_DIR", cls.audit_dir),
            audit_s3_bucket=_env("TRIAGE_AUDIT_BUCKET"),
            tenant_id=_env("TRIAGE_TENANT_ID", cls.tenant_id),
            workers=int(_env("TRIAGE_WORKERS", str(cls.workers))),
        )
