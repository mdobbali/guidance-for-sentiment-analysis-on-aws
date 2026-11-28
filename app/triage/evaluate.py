"""Evaluate triage on a labeled set. Nothing is written to the database.

Dataset: JSONL, one record per line:
  {"id": "...", "text": "...", "channel": "phone", "product": "...",
   "expect_ae": "yes" | "possible" | "no", "expect_serious": true | false | null,
   "expect_queue": "pharmacovigilance" | "complaint_handling" | "customer_service"}

The headline number is missed_ae: records with an adverse event (or special
situation) that were routed somewhere other than pharmacovigilance.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .classify import classify
from .routing import route


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score_item(item: dict, c, r) -> dict:
    ae_true = item["expect_ae"] in ("yes", "possible")
    got_ae = c.status == "ok" and c.data["adverse_event"]["assessment"] in ("yes", "possible")
    got_serious = c.status == "ok" and any(c.data["seriousness"].values())
    return {
        "id": item["id"],
        "expect_ae": item["expect_ae"],
        "got_ae": c.data["adverse_event"]["assessment"] if c.status == "ok" else c.status,
        "ae_true": ae_true,
        "ae_pred": got_ae,
        "expect_serious": item.get("expect_serious"),
        "serious_pred": got_serious,
        "expect_queue": item["expect_queue"],
        "queue": r.queue,
        "priority": r.priority,
        "needs_review": r.needs_human_review,
        # Unsafe: a safety-relevant record that did not reach pharmacovigilance
        # (manual_triage counts as safe: a person triages it first).
        "missed_ae": ae_true and r.queue not in ("pharmacovigilance", "manual_triage"),
        "evidence": len(c.evidence),
        "verified_evidence": sum(e.verified for e in c.evidence),
        "phi_entities": None,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
    }


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["ae_true"]]
    pred = [r for r in rows if r["ae_pred"]]
    ser = [r for r in rows if r["expect_serious"] is True]
    nonser = [r for r in rows if r["expect_serious"] is False]
    ev = sum(r["evidence"] for r in rows)
    return {
        "items": len(rows),
        "ae_positive": len(pos),
        "ae_recall": _rate(sum(r["ae_pred"] for r in pos), len(pos)),
        "ae_precision": _rate(sum(r["ae_true"] for r in pred), len(pred)),
        "missed_ae": sum(r["missed_ae"] for r in rows),
        "serious_recall": _rate(sum(r["serious_pred"] for r in ser), len(ser)),
        "serious_false_positive_rate": _rate(sum(r["serious_pred"] for r in nonser), len(nonser)),
        "queue_accuracy": _rate(sum(r["queue"] == r["expect_queue"] for r in rows), len(rows)),
        "human_review_rate": _rate(sum(r["needs_review"] for r in rows), len(rows)),
        "evidence_verification_rate": _rate(sum(r["verified_evidence"] for r in rows), ev),
        "refused_or_error": sum(r["got_ae"] in ("refused", "error") for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def run_eval(redactor, generator, dataset: Path, out_dir: Path, workers: int = 4, effort=None) -> dict:
    items = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]

    def one(item):
        t0 = time.perf_counter()
        red = redactor.redact(item["text"])
        c = classify(generator, red.text, item.get("channel"), item.get("product"))
        r = route(c, red.text)
        row = score_item(item, c, r)
        row["phi_entities"] = red.entity_counts
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, items))
    for row in rows:
        flag = "MISSED-AE" if row["missed_ae"] else ""
        print(
            f"{row['id']:<28} ae {row['expect_ae']:>8}->{row['got_ae']:<8} "
            f"queue {row['queue']:<19} {row['priority']} {flag}"
        )
    summary = summarize(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    if lat:
        summary["latency_ms_p50"] = lat[len(lat) // 2]
        summary["latency_ms_p95"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    summary.update(model=generator.model_id, effort=effort, redactor=redactor.name)
    report = {
        "dataset": str(dataset),
        "run_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return report
