"""Command line: triage init-db | run | queue | show | review | similar | eval | verify-audit"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audit import AuditLog, verify
from .config import Settings
from .pipeline import PRIORITIES, QUEUES, Record, TriagePipeline


def build_redactor(settings: Settings):
    from .redact import ComprehendMedicalRedactor, RegexRedactor

    if settings.redactor == "regex":
        print(
            "WARNING: regex redactor misses names and addresses; use only with synthetic data.",
            file=sys.stderr,
        )
        return RegexRedactor()
    import boto3

    return ComprehendMedicalRedactor(boto3.client("comprehendmedical", region_name=settings.aws_region))


def build_generator(settings: Settings):
    from .llm import ClaudeBedrockGenerator

    return ClaudeBedrockGenerator(
        settings.aws_region, settings.classify_model, settings.effort, settings.fallback_model or None
    )


def build_pipeline(settings: Settings) -> TriagePipeline:
    import boto3

    from .embeddings import TitanEmbedder
    from .store import PgStore, database_url_from_settings

    return TriagePipeline(
        redactor=build_redactor(settings),
        generator=build_generator(settings),
        embedder=TitanEmbedder(
            boto3.client("bedrock-runtime", region_name=settings.aws_region),
            settings.embed_model,
            settings.embed_dim,
        ),
        store=PgStore(database_url_from_settings(settings), embed_dim=settings.embed_dim),
        audit=AuditLog(settings.audit_dir, settings.audit_s3_bucket),
        effort=settings.effort,
    )


def _user() -> str:
    return os.environ.get("TRIAGE_USER") or f"cli:{getpass.getuser()}"


def load_records(path: Path) -> list[Record]:
    """CSV or JSONL with columns/keys: id (or case_id), text, and optional received_at, channel, product."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        Record(
            case_id=str(r.get("case_id") or r["id"]),
            text=r["text"],
            received_at=r.get("received_at") or None,
            channel=r.get("channel") or None,
            product=r.get("product") or None,
        )
        for r in rows
    ]


def cmd_init_db(args) -> int:
    from .store import PgStore, database_url_from_settings

    s = Settings.from_env()
    PgStore(database_url_from_settings(s), embed_dim=s.embed_dim).init_schema()
    print("Schema ready.")
    return 0


def cmd_run(args) -> int:
    s = Settings.from_env()
    pipe = build_pipeline(s)
    tenant, user = args.tenant or s.tenant_id, _user()
    records = load_records(Path(args.path))

    def one(rec):
        return pipe.process(rec, tenant_id=tenant, user=user)

    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        for out in pool.map(one, records):
            if out.get("skipped"):
                print(f"{out['case_id']:<24} already triaged, skipped")
                continue
            r = out["route"]
            review = "REVIEW" if r["needs_human_review"] else ""
            print(
                f"{out['case_id']:<24} {r['priority']} {r['queue']:<19} {review:<6} {'; '.join(r['reasons'])}"
            )
    return 0


def cmd_queue(args) -> int:
    s = Settings.from_env()
    pipe = build_pipeline(s)
    for c in pipe.store.queue(args.tenant or s.tenant_id, args.limit):
        print(f"{c['case_id']:<24} {c['priority']} {c['queue']:<19} {c['redacted_text'][:80]!r}")
    return 0


def cmd_show(args) -> int:
    s = Settings.from_env()
    case = build_pipeline(s).store.get_case(args.tenant or s.tenant_id, args.case_id)
    print(json.dumps(case, indent=2, default=str) if case else "not found")
    return 0 if case else 1


def cmd_review(args) -> int:
    s = Settings.from_env()
    out = build_pipeline(s).review(
        args.case_id,
        tenant_id=args.tenant or s.tenant_id,
        reviewer=_user(),
        decision=args.decision,
        queue=args.queue,
        priority=args.priority,
        reason=args.reason,
        adverse_event=args.adverse_event,
        serious=args.serious,
    )
    print(f"Recorded {out['decision']} -> {out['queue']} {out['priority']} (audit #{out['audit_seq']})")
    return 0


def cmd_similar(args) -> int:
    s = Settings.from_env()
    for row in build_pipeline(s).store.similar(args.tenant or s.tenant_id, args.case_id, args.k):
        print(
            f"{row['similarity']:.3f} {row['case_id']:<24} {row['queue']:<19} {row['redacted_text'][:70]!r}"
        )
    return 0


def cmd_eval(args) -> int:
    from .evaluate import run_eval

    s = Settings.from_env()
    report = run_eval(
        build_redactor(s), build_generator(s), Path(args.dataset), Path(args.out), s.workers, s.effort
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


def cmd_verify(args) -> int:
    ok, _, msg = verify(args.path)
    print(("OK: " if ok else "FAILED: ") + msg)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="triage", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="create tables, indexes and row-level security").set_defaults(
        fn=cmd_init_db
    )

    p = sub.add_parser("run", help="triage a CSV or JSONL file of feedback records")
    p.add_argument("path")
    p.add_argument("--tenant")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("queue", help="list open cases that need human review")
    p.add_argument("--tenant")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("show", help="show one case")
    p.add_argument("case_id")
    p.add_argument("--tenant")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("review", help="record a human review decision")
    p.add_argument("case_id")
    p.add_argument("--decision", choices=["confirm", "override"], required=True)
    p.add_argument("--queue", choices=QUEUES, required=True)
    p.add_argument("--priority", choices=PRIORITIES, required=True)
    p.add_argument("--reason", default="")
    p.add_argument("--adverse-event", choices=["yes", "possible", "no"])
    p.add_argument("--serious", nargs="*", help="seriousness criteria the reviewer confirms")
    p.add_argument("--tenant")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("similar", help="find similar past cases (duplicates, trends)")
    p.add_argument("case_id")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--tenant")
    p.set_defaults(fn=cmd_similar)

    p = sub.add_parser("eval", help="evaluate on a labeled JSONL set (no database writes)")
    p.add_argument("dataset")
    p.add_argument("--out", default="eval-results")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("verify-audit", help="check an audit file's hash chain")
    p.add_argument("path")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
