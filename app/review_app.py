"""Streamlit review queue: open cases that need a person, highest priority first.

Run: streamlit run review_app.py
For anything beyond local use, put this behind real authentication (Streamlit's
OIDC `st.login`, or an ALB with Cognito) so reviews are attributed to real users.
"""

from __future__ import annotations

import os

import streamlit as st

from triage.classify import SERIOUSNESS
from triage.cli import build_pipeline
from triage.config import Settings
from triage.pipeline import PRIORITIES, QUEUES


@st.cache_resource
def pipeline():
    settings = Settings.from_env()
    return settings, build_pipeline(settings)


def current_user() -> str:
    user = getattr(st, "user", None)
    if user is not None and getattr(user, "is_logged_in", False):
        return str(user.get("email") or user.get("sub"))
    return os.environ.get("TRIAGE_USER", "local-unauthenticated")


def main() -> None:
    st.set_page_config(page_title="Triage review queue", layout="wide")
    settings, pipe = pipeline()
    tenant, user = settings.tenant_id, current_user()
    st.title("Triage review queue")
    st.caption(f"Tenant `{tenant}` · reviewer `{user}` · model `{settings.classify_model}`")

    cases = pipe.store.queue(tenant, limit=100)
    if not cases:
        st.success("Nothing waiting for review.")
        return
    labels = [f"{c['priority']} · {c['queue']} · {c['case_id']}" for c in cases]
    case = cases[st.sidebar.radio("Open cases", range(len(cases)), format_func=lambda i: labels[i])]
    cls, route = case["classification"], case["route"]

    left, right = st.columns([3, 2])
    with left:
        st.subheader(case["case_id"])
        st.caption(
            f"{case.get('channel') or 'unknown channel'} · {case.get('product') or 'no product field'} · "
            f"PHI redacted: {case['redaction'].get('entities') or 'none found'}"
        )
        st.text(case["redacted_text"])
        st.subheader("Why it is here")
        for reason in route["reasons"]:
            st.markdown(f"- {reason}")
        for item in route.get("follow_up", []):
            st.markdown(f"- Follow-up: {item}")
        if cls.get("status") == "ok":
            st.subheader("Model evidence")
            for e in cls["evidence"]:
                st.markdown(
                    f"{'✅' if e['verified'] else '⚠️ not found in text'} **{e['field']}**: “{e['quote']}”"
                )
    with right:
        if cls.get("status") == "ok":
            ae = cls["adverse_event"]
            serious = [k for k, v in cls["seriousness"].items() if v]
            st.metric("Adverse event", ae["assessment"], help=ae["description"] or None)
            st.markdown(f"**Seriousness:** {', '.join(serious) or 'none flagged'}")
            st.markdown(f"**Category:** {cls['category']} · **sentiment:** {cls['sentiment']}")
            st.markdown(f"**Device malfunction:** {cls['device_malfunction']}")
            st.caption(cls["rationale"])
        else:
            st.error(f"Classification {cls.get('status')}: triage manually.")

        st.divider()
        with st.form(f"review-{case['case_id']}"):
            st.markdown("**Your decision**")
            queue = st.selectbox("Queue", QUEUES, index=QUEUES.index(route["queue"]))
            priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(route["priority"]))
            ae_choice = st.selectbox("Adverse event", ["yes", "possible", "no"])
            serious_choice = st.multiselect("Seriousness criteria", SERIOUSNESS)
            reason = st.text_area("Reason (required if you change the route)")
            if st.form_submit_button("Record review"):
                decision = (
                    "confirm" if (queue, priority) == (route["queue"], route["priority"]) else "override"
                )
                try:
                    out = pipe.review(
                        case["case_id"],
                        tenant_id=tenant,
                        reviewer=user,
                        decision=decision,
                        queue=queue,
                        priority=priority,
                        reason=reason,
                        adverse_event=ae_choice,
                        serious=serious_choice,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Recorded {decision} (audit #{out['audit_seq']}).")
                    st.rerun()


main()
