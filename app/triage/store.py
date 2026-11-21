"""Aurora PostgreSQL store for triaged cases, with per-tenant row-level security
and pgvector similarity search for duplicate and trend detection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from typing import Protocol

from psycopg.types.json import Jsonb


class Store(Protocol):
    def exists(self, tenant_id: str, case_id: str) -> bool: ...
    def save_case(self, tenant_id: str, case: dict, embedding: list[float] | None) -> None: ...
    def get_case(self, tenant_id: str, case_id: str) -> dict | None: ...
    def queue(self, tenant_id: str, limit: int = 50) -> list[dict]: ...
    def record_review(self, tenant_id: str, case_id: str, review: dict) -> None: ...
    def similar(self, tenant_id: str, case_id: str, k: int = 5) -> list[dict]: ...


def database_url_from_settings(settings) -> str:
    if settings.database_url:
        return settings.database_url
    if not (settings.db_host and settings.db_secret_arn):
        raise RuntimeError("Set TRIAGE_DATABASE_URL, or TRIAGE_DB_HOST and TRIAGE_DB_SECRET_ARN.")
    import boto3
    from psycopg.conninfo import make_conninfo

    secret = boto3.client("secretsmanager", region_name=settings.aws_region).get_secret_value(
        SecretId=settings.db_secret_arn
    )
    creds = json.loads(secret["SecretString"])
    return make_conninfo(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=creds["username"],
        password=creds["password"],
        sslmode="require",
    )


_CASE_COLUMNS = (
    "case_id, received_at, channel, product, source_sha256, redacted_text, redaction, classification, "
    "route, queue, priority, needs_review, status, review, model, prompt_sha256, created_at, reviewed_at"
)


class PgStore:
    def __init__(self, conninfo: str, embed_dim: int = 1024):
        self._conninfo = conninfo
        self._embed_dim = embed_dim

    @contextmanager
    def _tx(self, tenant_id: str | None) -> Iterator:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self._conninfo, row_factory=dict_row) as conn:
            if tenant_id is not None:
                from pgvector.psycopg import register_vector

                register_vector(conn)
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            yield conn

    def init_schema(self) -> None:
        sql = resources.files("triage").joinpath("schema.sql").read_text()
        with self._tx(None) as conn:
            conn.execute(sql.replace("__EMBED_DIM__", str(int(self._embed_dim))))

    def exists(self, tenant_id, case_id) -> bool:
        with self._tx(tenant_id) as conn:
            row = conn.execute(
                "SELECT 1 FROM cases WHERE tenant_id = %s AND case_id = %s", (tenant_id, case_id)
            ).fetchone()
        return row is not None

    def save_case(self, tenant_id, case, embedding) -> None:
        import numpy as np

        r = case["route"]
        with self._tx(tenant_id) as conn:
            conn.execute(
                "INSERT INTO cases (tenant_id, case_id, received_at, channel, product, source_sha256, "
                "redacted_text, redaction, classification, route, queue, priority, needs_review, model, "
                "prompt_sha256, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    tenant_id,
                    case["case_id"],
                    case.get("received_at"),
                    case.get("channel"),
                    case.get("product"),
                    case["source_sha256"],
                    case["redacted_text"],
                    Jsonb(case["redaction"]),
                    Jsonb(case["classification"]),
                    Jsonb(r),
                    r["queue"],
                    r["priority"],
                    r["needs_human_review"],
                    case.get("model"),
                    case["prompt_sha256"],
                    None if embedding is None else np.asarray(embedding, dtype=np.float32),
                ),
            )

    def get_case(self, tenant_id, case_id) -> dict | None:
        with self._tx(tenant_id) as conn:
            return conn.execute(
                f"SELECT {_CASE_COLUMNS} FROM cases WHERE tenant_id = %s AND case_id = %s",
                (tenant_id, case_id),
            ).fetchone()

    def queue(self, tenant_id, limit=50) -> list[dict]:
        with self._tx(tenant_id) as conn:
            return conn.execute(
                f"SELECT {_CASE_COLUMNS} FROM cases WHERE tenant_id = %s AND status = 'open' "
                "AND needs_review ORDER BY priority, created_at LIMIT %s",
                (tenant_id, limit),
            ).fetchall()

    def record_review(self, tenant_id, case_id, review) -> None:
        with self._tx(tenant_id) as conn:
            cur = conn.execute(
                "UPDATE cases SET status = 'reviewed', review = %s, reviewed_at = now(), "
                "queue = %s, priority = %s WHERE tenant_id = %s AND case_id = %s AND status = 'open'",
                (Jsonb(review), review["queue"], review["priority"], tenant_id, case_id),
            )
            if cur.rowcount != 1:
                raise ValueError(f"{case_id} not found or already reviewed")

    def similar(self, tenant_id, case_id, k=5) -> list[dict]:
        with self._tx(tenant_id) as conn:
            return conn.execute(
                "SELECT c.case_id, c.queue, c.priority, c.product, c.redacted_text, "
                "1 - (c.embedding <=> t.embedding) AS similarity "
                "FROM cases t JOIN cases c ON c.tenant_id = t.tenant_id AND c.case_id <> t.case_id "
                "WHERE t.tenant_id = %s AND t.case_id = %s AND c.embedding IS NOT NULL "
                "ORDER BY c.embedding <=> t.embedding LIMIT %s",
                (tenant_id, case_id, k),
            ).fetchall()
