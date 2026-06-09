"""
Outreach Agent — generates personalised WhatsApp messages in parallel,
then validates each one through compliance guardrails before returning.
"""
import json
import asyncio
from app.models.state import AgentState
from app.tools.message_tools import generate_whatsapp_message
from app.services.guardrails import enforce
from app.agents.base import timed_node


async def _generate_one(rec: dict) -> dict:
    """Generate and validate a single WhatsApp message asynchronously."""
    try:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            generate_whatsapp_message.invoke,
            json.dumps(rec),
        )
        result = json.loads(raw)
    except Exception:
        return {
            "customer_id": rec.get("customer", {}).get("id"),
            "message": "",
            "char_count": 0,
            "compliant": False,
            "violations": ["message_generation_failed"],
        }
    message = result.get("message", "")
    customer = rec.get("customer", {})

    compliance = enforce(message, customer)
    return {
        "customer_id": result.get("customer_id"),
        "message": message,
        "char_count": compliance["char_count"],
        "compliant": compliance["compliant"],
        "violations": compliance["violations"],
    }


@timed_node
def outreach_agent_node(state: AgentState) -> dict:
    """Generate all WhatsApp messages in parallel, validate via guardrails."""
    recs = state.get("final_recommendations", [])
    if not recs:
        return {
            "trace": [{
                "step": "outreach_agent",
                "decision": "Skipped — no recommendations to generate messages for",
                "tools_called": [],
                "result_summary": "0 messages generated",
                "skipped": True,
            }]
        }

    async def run_all():
        return await asyncio.gather(*[_generate_one(rec) for rec in recs])

    try:
        loop = asyncio.get_running_loop()
        results = loop.run_until_complete(run_all())
    except RuntimeError:
        results = asyncio.run(run_all())

    messages_by_id = {r["customer_id"]: r for r in results if r.get("customer_id")}

    updated = []
    compliant_count = 0
    for rec in recs:
        cust_id = rec.get("customer", {}).get("id") or rec.get("customer_id")
        msg_data = messages_by_id.get(cust_id, {})
        new_rec = {
            **rec,
            "whatsapp_message": msg_data.get("message", ""),
            "message_char_count": msg_data.get("char_count", 0),
            "message_compliant": msg_data.get("compliant", False),
            "message_violations": msg_data.get("violations", []),
        }
        if new_rec["message_compliant"]:
            compliant_count += 1
        updated.append(new_rec)

    return {
        "final_recommendations": updated,
        "trace": [{
            "step": "outreach_agent",
            "decision": f"Generated {len(recs)} messages in parallel via OpenRouter, validated against compliance guardrails",
            "tools_called": ["generate_whatsapp_message (parallel)"],
            "result_summary": f"{compliant_count}/{len(recs)} compliant | avg {sum(r.get('message_char_count',0) for r in updated)//max(len(updated),1)} chars",
            "skipped": False,
        }],
    }
