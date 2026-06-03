"""
Conversation Service — the only layer that knows about LangGraph.

main.py calls this service; it never touches the graph directly.
This keeps the API layer clean and makes the agent swappable.
"""
import uuid
import time
from langchain_core.messages import HumanMessage

from app.agents.supervisor import get_crm_graph
from app.models.schemas import ChatResponse, CustomerRecommendation, TraceEntry, ProductMatch, CustomerProfile


def _is_content_filter_error(exc: Exception) -> bool:
    """Azure OpenAI blocks jailbreak/injection prompts — handle gracefully."""
    msg = str(exc).lower()
    return any(k in msg for k in ("content_filter", "jailbreak", "responsibleaipolicyviolation"))


def _build_recommendation(rec: dict) -> CustomerRecommendation | None:
    """Convert a raw recommendation dict to a typed schema. Returns None if malformed."""
    try:
        c = rec.get("customer", {})
        p = rec.get("product", {})
        return CustomerRecommendation(
            customer=CustomerProfile(
                id=c.get("id", ""),
                name=c.get("name", ""),
                age=c.get("age", 0),
                occupation=c.get("occupation", ""),
                city=c.get("city", ""),
                region=c.get("region", ""),
                monthly_income=c.get("monthly_income", 0),
                account_balance=c.get("account_balance", 0),
                credit_score=c.get("credit_score", 0),
                has_personal_loan=c.get("has_personal_loan", False),
                has_salary_account=c.get("has_salary_account", False),
                phone=c.get("phone", ""),
            ),
            propensity_score=rec.get("propensity_score", 0),
            tier=rec.get("tier", "Medium"),
            score_reason=rec.get("score_reason", ""),
            product=ProductMatch(
                id=p.get("id", ""),
                name=p.get("name", ""),
                interest_rate=p.get("interest_rate", 0),
                max_amount=p.get("max_amount", 0),
                description=p.get("description", ""),
            ),
            recommended_loan_amount=rec.get("recommended_loan_amount", 0),
            whatsapp_message=rec.get("whatsapp_message"),
            message_char_count=rec.get("message_char_count"),
            message_compliant=rec.get("message_compliant"),
        )
    except Exception:
        return None


async def handle_chat(message: str, thread_id: str) -> ChatResponse:
    """
    Invoke the agent graph for one conversational turn.

    - New thread_id → full initial state
    - Existing thread_id → only the new message; all other state restored from SqliteSaver
    """
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    graph = get_crm_graph()
    existing = await graph.aget_state(config)
    is_followup = bool(existing.values)

    if is_followup:
        invoke_state = {"messages": [HumanMessage(content=message)]}
    else:
        invoke_state = {
            "messages": [HumanMessage(content=message)],
            "rm_intent": "",
            "filters": {},
            "retrieved_customers": [],
            "scored_customers": [],
            "final_recommendations": [],
            "final_response": "",
            "next_agent": "",
            "unsupported_product": False,
            "loan_category": "personal_loan",
            "trace": [],
        }

    t0 = time.perf_counter()
    try:
        result = await graph.ainvoke(invoke_state, config=config)
    except Exception as e:
        total_ms = int((time.perf_counter() - t0) * 1000)
        if _is_content_filter_error(e):
            return ChatResponse(
                thread_id=thread_id,
                response=(
                    "Your message was blocked by the AI safety filter. "
                    "Please rephrase as a normal banking CRM request (e.g. "
                    "*\"Find high-value customers for a personal loan\"*)."
                ),
                recommendations=[],
                trace=[
                    TraceEntry(
                        step="safety_filter",
                        decision="Azure content management policy blocked the prompt (jailbreak/injection detected)",
                        tools_called=[],
                        result_summary="Request blocked — no CRM data accessed",
                        skipped=True,
                        duration_ms=total_ms,
                    )
                ],
                stats={"blocked_by_filter": True, "is_followup": is_followup},
                total_duration_ms=total_ms,
            )
        raise

    total_ms = int((time.perf_counter() - t0) * 1000)

    recommendations = [
        r for r in (_build_recommendation(rec) for rec in result.get("final_recommendations", []))
        if r is not None
    ]

    trace = [
        TraceEntry(
            step=t.get("step", ""),
            decision=t.get("decision", ""),
            tools_called=t.get("tools_called", []),
            result_summary=t.get("result_summary", ""),
            skipped=t.get("skipped", False),
            duration_ms=t.get("duration_ms", 0),
        )
        for t in result.get("trace", [])
    ]

    # A turn is only a "follow-up" if it reused prior state without re-querying.
    # If data_agent ran (visible in trace), it's a new/restarted search regardless
    # of whether a checkpoint existed.
    ran_data_agent = any(
        t.get("step") == "data_agent" and not t.get("skipped")
        for t in result.get("trace", [])
    )
    actual_followup = is_followup and not ran_data_agent

    stats = {
        "customers_retrieved": len(result.get("retrieved_customers", [])),
        "customers_scored": len(result.get("scored_customers", [])),
        "recommendations": len(recommendations),
        "compliant_messages": sum(1 for r in recommendations if r.message_compliant),
        "is_followup": actual_followup,
        "total_duration_ms": total_ms,
    }

    return ChatResponse(
        thread_id=thread_id,
        response=result.get("final_response", ""),
        recommendations=recommendations,
        trace=trace,
        stats=stats,
        total_duration_ms=total_ms,
    )
