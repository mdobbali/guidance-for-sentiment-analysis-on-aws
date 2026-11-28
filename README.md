# triage: complaint and adverse-event triage for life-sciences customer feedback on AWS

Customer feedback reaches pharma and medical device companies through call centres, web forms, email and chat, and some of it describes an adverse event that has to reach pharmacovigilance quickly. This project reads each record and sends it to the right team, with a person in the loop for anything safety-relevant:

1. **Redacts PHI first.** Amazon Comprehend Medical `DetectPHI` replaces names, dates, phone numbers and similar details with type tags (`[NAME]`, `[AGE]`), so the model can still tell that an identifiable patient exists. The original text is never stored, only its SHA-256.
2. **Classifies.** Claude on Amazon Bedrock fills a schema-enforced record:
   - complaint category
   - adverse event: yes, possible or no (special situations such as lack of effect, medication errors, overdose and pregnancy or accidental exposure count as *possible*)
   - the six seriousness criteria
   - the four ICSR minimum criteria
   - device malfunction
   - sentiment
   - a verbatim quote supporting every positive flag
3. **Routes in plain code.** The routing table assigns a queue (pharmacovigilance, complaint handling, customer service, or manual triage on any model failure) and a priority (P1–P4). It is built to catch every possible adverse event:
   - A flag whose quote can't be found in the text is **kept** and sent to human review, never silently dropped.
   - A keyword safety net ("hospital", "ER", "seizure", "pregnant" and so on) sends records the model called "no adverse event" to a person.
   - Any adverse event or device malfunction always needs human review, and missing ICSR criteria become follow-up tasks.
4. **Stores and audits.** Cases go to Aurora PostgreSQL with per-tenant row-level security and pgvector embeddings for "find similar cases" (duplicates, emerging trends). Every triage and review decision is written to a hash-chained audit log, optionally copied to S3 Object Lock.
5. **Puts a person in the loop.** A Streamlit review queue, or `triage review`, records who decided what and why, next to the model's original output, which is never overwritten.

> This is a reference implementation. It is not a validated pharmacovigilance or complaint-handling system and doesn't replace your PV or complaint SOPs, safety database, or regulatory reporting timelines (for example 21 CFR 314.80, 803 or 820.198). The model never closes a case. Run it on synthetic data until you have validated it.

## Architecture

```
 feedback ─► Comprehend Medical ─► Claude (Bedrock) ─► routing rules ─► Aurora PostgreSQL ─► review queue
 (CSV/JSONL)  DetectPHI redaction   JSON schema +       queue, priority,  cases (redacted)     (Streamlit / CLI)
                                    verbatim evidence   human review      pgvector, RLS
                    └──────────────────── hash-chained audit trail ──► S3 Object Lock (WORM) ────────┘
```

| Piece | Choice |
|---|---|
| PHI redaction | Amazon Comprehend Medical DetectPHI, with long texts split into windows. A regex redactor exists for offline demos with synthetic data only. |
| Classifier | Claude Opus 5.5 on Bedrock (`anthropic.claude-opus-5-5`) through the Anthropic SDK with an enforced JSON schema. If the model declines, the request is retried on `anthropic.claude-opus-4-8`. |
| Similarity | Titan Text Embeddings V2 on redacted text, with an HNSW index |
| Database | Aurora PostgreSQL Serverless v2 in isolated subnets (KMS, managed and rotated secret, row-level security) |
| Access | An SSM Session Manager host with no inbound ports. No Cloud9 or SageMaker notebook. |

## What changed from the AWS Guidance this was forked from

This fork started from [AWS Guidance for Sentiment Analysis on AWS](https://github.com/aws-solutions-library-samples/guidance-for-sentiment-analysis-on-aws) (SO9449, MIT-0), a notebook that ran Comprehend `detect_sentiment` from SQL over hotel reviews:

| Upstream (2024) | Here |
|---|---|
| Positive/negative sentiment on hotel reviews | Complaint category, adverse events, seriousness, ICSR criteria and device malfunction, with sentiment kept as one field |
| Sentiment recomputed by a Comprehend call on every `SELECT` | Classified once, stored with the model, prompt hash and evidence |
| No privacy handling | PHI redacted before storage, embedding or any model call |
| No labels or metrics | Labeled eval set reporting adverse-event recall and precision, missed adverse events, seriousness recall, queue accuracy and evidence verification |
| Notebook and copy-paste shell steps; Cloud9 (closed to new customers) and a SageMaker notebook with full-access IAM | A package with CLI, review UI, 16 unit tests and CI |
| Aurora with `PubliclyAccessible: true`, notebook security group open to 0.0.0.0/0 | Isolated subnets, no public endpoints, scoped security groups and IAM |
| Two `db.r6g.2xlarge` instances | Serverless v2, which can pause when idle |
| Hotel reviews that look derived from a public review corpus | A fictional, labeled feedback set written for this repo |

## Quick start

Requirements: an AWS account with Bedrock access to Claude Opus 5.5 (and Opus 4.8 for the fallback) and Titan Text Embeddings V2, Comprehend Medical in your region (for example us-east-1, us-east-2 or us-west-2), the AWS CLI with the Session Manager plugin, and Python 3.11+.

```bash
# 1. Infrastructure. Attach the AppPolicyArn output to the role you run as.
aws cloudformation deploy --stack-name triage-dev --template-file infra/triage.yaml \
  --capabilities CAPABILITY_NAMED_IAM --parameter-overrides EnvName=triage-dev
aws cloudformation describe-stacks --stack-name triage-dev --query 'Stacks[0].Outputs' --output table

# 2. Tunnel to the private database (run the PortForwardCommand output in its own terminal).

# 3. Install and configure
cd app && python -m venv .venv && source .venv/bin/activate && pip install -e ".[ui]"
cp env.example .env   # fill TRIAGE_DB_SECRET_ARN and TRIAGE_AUDIT_BUCKET; then:
set -a; source .env; set +a
triage init-db

# 4. Measure before you trust it (no database writes; ~24 model calls + 24 DetectPHI calls)
triage eval eval/triage_eval.jsonl

# 5. Triage, review, find similar cases
triage run eval/triage_eval.jsonl        # any CSV/JSONL with id and text columns
triage queue
triage review ae-device-injury --decision confirm --queue pharmacovigilance --priority P1 \
  --adverse-event yes --serious other_medically_important
triage similar device-malfunction-noharm
streamlit run review_app.py
triage verify-audit audit/audit-default.jsonl
```

### Reading the eval

`eval/triage_eval.jsonl` has 24 fictional records. Seven describe adverse events (five serious), one only hints at one, four are special situations, eight are product or device complaints and four are not complaints. The set includes a prompt-injection attempt, a record that explicitly denies symptoms, an HCP report and a death report. Watch:

- **`missed_ae`**: safety-relevant records that didn't reach pharmacovigilance. This should be 0.
- **`ae_recall`** before `ae_precision`: in pharmacovigilance a false alarm costs reviewer time, while a miss can cost a late report.
- **`human_review_rate`**: the reviewer workload this configuration creates.

The set is small and written for illustration. Before relying on any number, build a larger labeled set from your own de-identified records, or from public narratives such as the openFDA MAUDE device-event text, with labels from your PV and complaint teams.

## Repository layout

```
app/triage/      redact, classify (prompt + schema), routing, pipeline, store, evaluate, audit, cli
app/review_app.py
app/eval/        labeled fictional feedback (JSONL)
app/tests/       unit tests (no AWS or database needed)
infra/triage.yaml
```

## Development

```bash
cd app && pip install -e ".[ui,dev]"
ruff check . && ruff format --check . && pytest -q && cfn-lint ../infra/triage.yaml
```

## Known limits

- **Authentication.** The review app reads the user from Streamlit's OIDC login if it's configured, and from `TRIAGE_USER` otherwise. Put it behind real SSO before anyone else uses it. Reviews record the reviewer, decision, reason and meaning, but they are not 21 CFR Part 11 electronic signatures. That needs re-authentication at signing and validated controls.
- **Database roles.** The quick start connects as the RDS master user. Row-level security is forced, but roles with superuser or `BYPASSRLS` skip it. Use a dedicated least-privilege role in production.
- **Redaction.** It's only as good as Comprehend Medical. Spot-check redacted output on your own data, and don't use the regex redactor with real records.
- **Language.** English only. The prompt and safety net assume English text.
- **Scale.** Processing is synchronous with a small thread pool. For high volume, put SQS or Step Functions in front and watch Bedrock and Comprehend Medical quotas.
- **Deployment.** Not yet tested against a live AWS deployment in this fork. The unit tests use in-memory fakes for the database, models and Comprehend Medical.

## License and credits

The code is MIT-0 (see `LICENSE`), as was the upstream AWS Guidance it was forked from. This project is not affiliated with or endorsed by AWS. The eval records, products ("Zelvora", "GlucoTrack", "Pentivax") and people in them are fictional.
