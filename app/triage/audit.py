"""Append-only, hash-chained audit trail.

Every record stores the SHA-256 of the previous record, so editing or deleting
any line breaks the chain and `verify` reports where. Records are written to a
local JSONL file and, when a bucket is configured, also to S3 with Object Lock
(WORM) so they can't be overwritten during the retention period.

This supports 21 CFR 11.10(e)-style audit trails (secure, computer-generated,
time-stamped), but using it does not make a system Part 11 compliant on its own:
validation, access control and procedures are still the operator's job.

The chain is kept per process. Run one writer per audit file; a multi-instance
deployment should give each instance its own file/prefix (see `instance`).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def record_hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(_canonical(body)).hexdigest()


class AuditLog:
    def __init__(
        self,
        directory: str | os.PathLike,
        s3_bucket: str | None = None,
        s3_client=None,
        instance: str = "default",
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / f"audit-{instance}.jsonl"
        self._instance = instance
        self._bucket = s3_bucket
        self._s3 = s3_client
        if s3_bucket and s3_client is None:
            import boto3

            self._s3 = boto3.client("s3")
        self._lock = threading.Lock()
        self._seq, self._head = self._load_head()

    def _load_head(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS
        last = None
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    last = line
        if last is None:
            return 0, GENESIS
        rec = json.loads(last)
        return rec["seq"], rec["record_hash"]

    def append(self, event: dict) -> dict:
        with self._lock:
            record = {
                **event,
                "seq": self._seq + 1,
                "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
                "instance": self._instance,
                "prev_hash": self._head,
            }
            record["record_hash"] = record_hash(record)
            line = _canonical(record) + b"\n"
            # Write S3 first: if the WORM copy fails we don't advance the local chain.
            if self._bucket:
                day = record["ts"][:10]
                key = f"audit/{self._instance}/{day}/{record['seq']:012d}-{record['record_hash'][:16]}.json"
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=line,
                    ContentType="application/json",
                    ChecksumAlgorithm="SHA256",
                )
            with self.path.open("ab") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._seq, self._head = record["seq"], record["record_hash"]
            return record


def verify(path: str | os.PathLike) -> tuple[bool, int, str]:
    """Check an audit file. Returns (ok, records_checked, message)."""
    prev, seq, n = GENESIS, 0, 0
    with open(path, "rb") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            if rec.get("prev_hash") != prev:
                return False, n, f"line {lineno}: prev_hash does not match the previous record"
            if rec.get("seq") != seq + 1:
                return False, n, f"line {lineno}: sequence gap (expected {seq + 1}, got {rec.get('seq')})"
            if record_hash(rec) != rec.get("record_hash"):
                return False, n, f"line {lineno}: record_hash mismatch (record was modified)"
            prev, seq = rec["record_hash"], rec["seq"]
    return True, n, f"{n} records, chain intact"
