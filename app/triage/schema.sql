-- triage schema. __EMBED_DIM__ is replaced at apply time (Titan V2 default: 1024).
-- Only redacted text is stored; the original is kept as a SHA-256 so a record can
-- be matched to its source system without the database holding PHI.
-- Row-level security limits a session to the tenant set with
--   SELECT set_config('app.tenant_id', '<tenant>', true);
-- Roles with superuser or BYPASSRLS skip these policies: connect the app as a
-- dedicated role without them.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cases (
    tenant_id        text        NOT NULL,
    case_id          text        NOT NULL,
    received_at      timestamptz,
    channel          text,
    product          text,
    source_sha256    text        NOT NULL,
    redacted_text    text        NOT NULL,
    redaction        jsonb       NOT NULL,
    classification   jsonb       NOT NULL,
    route            jsonb       NOT NULL,
    queue            text        NOT NULL,
    priority         text        NOT NULL,
    needs_review     boolean     NOT NULL,
    status           text        NOT NULL DEFAULT 'open',  -- open | reviewed
    review           jsonb,
    model            text,
    prompt_sha256    text        NOT NULL,
    embedding        vector(__EMBED_DIM__),
    created_at       timestamptz NOT NULL DEFAULT now(),
    reviewed_at      timestamptz,
    PRIMARY KEY (tenant_id, case_id)
);

CREATE INDEX IF NOT EXISTS cases_queue ON cases (tenant_id, status, priority, created_at);
CREATE INDEX IF NOT EXISTS cases_embedding_hnsw ON cases USING hnsw (embedding vector_cosine_ops);

ALTER TABLE cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE cases FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'cases' AND policyname = 'tenant_isolation') THEN
        CREATE POLICY tenant_isolation ON cases
            USING (tenant_id = current_setting('app.tenant_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
    END IF;
END
$$;
