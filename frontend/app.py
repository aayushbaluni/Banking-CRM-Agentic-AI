"""Streamlit Chat UI — Banking CRM Agentic AI."""
import streamlit as st
import httpx
import json
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Banking CRM AI", page_icon="🏦", layout="wide")
st.title("🏦 Banking CRM AI Assistant")
st.caption("Powered by Azure GPT-4o + LangGraph | Stateful multi-turn conversations")

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("thread_id", ""),
    ("messages", []),
    ("last_recommendations", []),
    ("last_trace", []),
    ("last_stats", {}),
    ("last_total_ms", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 CRM Dashboard")
    try:
        summary = httpx.get(f"{API_URL}/customers/summary", timeout=5).json()
        c1, c2 = st.columns(2)
        c1.metric("Total", summary["total_customers"])
        c2.metric("Cross-sell Pool", summary["cross_sell_pool"])
        c1.metric("Salary Holders", summary["salary_account_holders"])
        c2.metric("High Value", summary["high_value_customers"])
    except Exception:
        st.warning("Start backend: `uvicorn app.main:app --reload --port 8000`")

    st.divider()

    # ── Last run timing ───────────────────────────────────────────────────────
    if st.session_state.last_total_ms:
        st.subheader("⏱️ Last Run Timing")
        total_ms_sidebar = st.session_state.last_total_ms
        st.metric("End-to-end", f"{total_ms_sidebar / 1000:.2f}s")
        for t in st.session_state.last_trace:
            if not t.get("skipped") and t.get("duration_ms", 0) > 0:
                pct = t["duration_ms"] / total_ms_sidebar * 100
                st.progress(
                    min(pct / 100, 1.0),
                    text=f"`{t['step']}` — {t['duration_ms']}ms ({pct:.0f}%)",
                )
        st.divider()

    st.subheader("💡 Quick Queries")
    quick_queries = [
        "Find high-value customers likely to convert for a personal loan this month",
        "Generate personalized WhatsApp messages for the top 5 customers",
        "Which customers in Mumbai with salary accounts should we target for pre-approved loans?",
        "Find customers in Bangalore earning over ₹1 lakh with no personal loan",
    ]
    for q in quick_queries:
        if st.button(q[:52] + "…", use_container_width=True, key=q):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        for k in ["thread_id", "messages", "last_recommendations",
                  "last_trace", "last_stats", "last_total_ms"]:
            st.session_state[k] = [] if k not in ("thread_id",) else ""
        st.session_state.last_total_ms = 0
        st.rerun()

# ── Layout ────────────────────────────────────────────────────────────────────
col_chat, col_results = st.columns([3, 2])

with col_chat:
    st.subheader("💬 Conversation")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask me to find customers, score them, generate messages...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── API call — measure wall-clock time from submit to response ───────
        data = None
        error = None
        t_start = time.perf_counter()
        with st.spinner("Querying CRM → Scoring → Recommending…"):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={"message": user_input, "thread_id": st.session_state.thread_id},
                    timeout=180,
                )
                if resp.status_code != 200:
                    try:
                        body = resp.json()
                        detail = body.get("detail", body)
                        if isinstance(detail, list):
                            detail = detail[0].get("msg", str(detail))
                        error = f"API error ({resp.status_code}): {detail}"
                    except Exception:
                        error = f"API error ({resp.status_code}): {resp.text[:300]}"
                else:
                    data = resp.json()
                    if "thread_id" not in data:
                        error = f"Unexpected API response: {data}"
                        data = None
            except httpx.ConnectError:
                error = "Cannot connect to backend. Run: `uvicorn app.main:app --reload --port 8000`"
            except Exception as e:
                error = str(e)
        client_ms = int((time.perf_counter() - t_start) * 1000)

        # ── Render response OUTSIDE spinner so nothing gets swallowed ────────
        with st.chat_message("assistant"):
            if error:
                st.error(error)
            elif data:
                # Save state
                st.session_state.thread_id = data["thread_id"]
                st.session_state.last_recommendations = data.get("recommendations", [])
                st.session_state.last_trace = data.get("trace", [])
                st.session_state.last_stats = data.get("stats", {})
                st.session_state.last_total_ms = data.get("total_duration_ms", 0)

                # Response text
                st.markdown(data["response"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["response"]}
                )

                # ── Timing bar — client-side wall clock is ground truth ───────
                trace = data.get("trace", [])
                st.session_state.last_total_ms = client_ms

                step_pills = "  ·  ".join(
                    f"**{t['step']}** `{t.get('duration_ms', 0)}ms`"
                    for t in trace
                    if not t.get("skipped") and t.get("duration_ms", 0) > 0
                )
                st.info(f"⏱️ **Total: {client_ms / 1000:.2f}s**   {step_pills}")

                # ── Agent reasoning trace ─────────────────────────────────────
                if trace:
                    with st.expander("🧠 Agent reasoning trace", expanded=False):
                        for i, step in enumerate(trace):
                            skipped = step.get("skipped", False)
                            duration = step.get("duration_ms", 0)
                            icon = "⏭️" if skipped else "✅"
                            dur_str = f" · `{duration}ms`" if duration else ""
                            st.markdown(
                                f"{icon} **`{step['step']}`**{dur_str}  \n"
                                f"{step['decision']}"
                            )
                            if step.get("tools_called"):
                                st.code(", ".join(step["tools_called"]), language=None)
                            if step.get("result_summary"):
                                st.caption(step["result_summary"])
                            if i < len(trace) - 1:
                                st.markdown("↓")

                # ── Stats bar ─────────────────────────────────────────────────
                stats = data.get("stats", {})
                if stats:
                    label = "↩️ Follow-up" if stats.get("is_followup") else "🆕 New"
                    st.caption(
                        f"{label} · ⏱️ {client_ms / 1000:.2f}s · "
                        f"{stats.get('customers_retrieved', 0)} retrieved · "
                        f"{stats.get('customers_scored', 0)} scored · "
                        f"{stats.get('recommendations', 0)} recommended"
                    )

# ── Results Panel ─────────────────────────────────────────────────────────────
with col_results:
    recs = st.session_state.last_recommendations

    if recs:
        st.subheader(f"🎯 {len(recs)} Recommendations")
        for i, rec in enumerate(recs, 1):
            c = rec.get("customer", {})
            score = rec.get("propensity_score", 0)
            tier = rec.get("tier", "Medium")
            product = rec.get("product", {})
            msg = rec.get("whatsapp_message", "")
            compliant = rec.get("message_compliant")
            tier_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(tier, "🟡")

            with st.expander(f"{tier_icon} {i}. {c.get('name')} — {score:.0%}", expanded=i <= 3):
                cols = st.columns(2)
                cols[0].write(f"**City:** {c.get('city')}")
                cols[0].write(f"**Job:** {c.get('occupation')}")
                cols[0].write(f"**Income:** ₹{c.get('monthly_income', 0):,.0f}/mo")
                cols[1].write(f"**Credit:** {c.get('credit_score')}")
                cols[1].write(f"**Balance:** ₹{c.get('account_balance', 0):,.0f}")
                cols[1].write(f"**Salary A/C:** {'✓' if c.get('has_salary_account') else '✗'}")

                if product:
                    offered_rate = rec.get("offered_interest_rate") or product.get("interest_rate")
                    base_rate = product.get("interest_rate")
                    rate_label = (
                        f"{offered_rate}% p.a."
                        if offered_rate == base_rate
                        else f"{offered_rate}% p.a. *(base {base_rate}%)*"
                    )
                    st.info(f"**{product.get('name')}** @ {rate_label}")

                if rec.get("score_reason"):
                    st.caption(f"📊 {rec['score_reason']}")

                if msg:
                    char_count = rec.get("message_char_count", len(msg))
                    badge = "✅" if compliant else "⚠️"
                    if compliant:
                        st.success(f"💬 **WhatsApp** ({char_count} chars {badge})\n\n{msg}")
                    else:
                        st.warning(f"💬 **WhatsApp** ({char_count} chars {badge})\n\n{msg}")
                    st.code(msg, language=None)

        st.divider()
        export = [
            {
                "name": r.get("customer", {}).get("name"),
                "phone": r.get("customer", {}).get("phone"),
                "score": r.get("propensity_score"),
                "tier": r.get("tier"),
                "product": r.get("product", {}).get("name"),
                "message": r.get("whatsapp_message", ""),
                "compliant": r.get("message_compliant"),
            }
            for r in recs
        ]
        st.download_button(
            "⬇️ Export Results (JSON)",
            data=json.dumps(export, indent=2),
            file_name="crm_outreach.json",
            mime="application/json",
        )
    else:
        st.info("Results appear here after your first query.")
        st.markdown("""
**Try asking:**
- *"Find high-value customers for a personal loan"*
- *"Generate WhatsApp messages for top 5 customers"*
- *"Which Mumbai salary account holders have no loan?"*
        """)
