"""
Data Retrieval Agent — queries the CRM database via registered tools.

Includes a clarification check: if the query is too vague to map to a tool,
the agent asks a specific follow-up question instead of silently guessing.
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from app.models.state import AgentState
from app.prompts.registry import get as get_prompt
from app.agents.base import get_llm, timed_node, human_messages_for_llm
from app.tools.crm_tools import (
    get_high_value_customers,
    get_customers_without_personal_loan,
    get_customers_for_loan,
    get_customers_by_city,
    get_customer_transactions,
    get_customer_by_id,
    get_salary_account_holders_without_loan,
)

TOOLS = [
    get_high_value_customers,
    get_customers_without_personal_loan,
    get_customers_for_loan,
    get_customers_by_city,
    get_customer_transactions,
    get_customer_by_id,
    get_salary_account_holders_without_loan,
]
TOOL_MAP = {t.name: t for t in TOOLS}

_CLARIFY_SYSTEM = """You are ARIA, a banking CRM assistant. The RM sent a query that
is not clear enough to execute. Ask ONE specific, short question to clarify.

Available actions you support:
- Find customers for any loan type: personal, home, car/auto, business, education, gold, LAP
- Score customers by conversion likelihood
- Generate WhatsApp outreach messages

Ask a question that will unblock the task. Be friendly and brief (1 sentence).
Example good questions:
- "Which city should I search in — or would you like all cities?"
- "Are you looking for personal loan customers specifically, or a different product?"
- "Should I prioritise salary account holders, or search broadly?"
"""


def _needs_clarification(state: AgentState) -> tuple[bool, str]:
    """
    Ask the LLM if the query can be executed or needs more information.
    Returns (needs_clarification, clarification_question).
    """
    last_human = next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        "",
    )

    check_prompt = (
        f'You are a banking CRM assistant. Decide if you can execute this query RIGHT NOW.\n\n'
        f'RM query: "{last_human}"\n\n'
        'You have sensible defaults — use them freely, DO NOT ask about them:\n'
        '- "high-value" = balance ≥ ₹5 lakh\n'
        '- "top N" or "N mvp/best" = top N by propensity score (mvp = most likely to convert)\n'
        '- "personal loan customers" = customers without existing personal loan\n'
        '- "car loan / auto loan customers" = candidates for vehicle financing\n'
        '- "home loan customers" = candidates for housing/mortgage loans\n'
        '- "business loan customers" = business owners needing working capital\n'
        '- "education loan customers" = candidates for student/education financing\n'
        '- "gold loan customers" = candidates for loan against gold\n'
        '- "LAP customers" = candidates for Loan Against Property\n'
        '- No city = all cities. No income = use sensible defaults per category\n\n'
        'CRITICAL: ANY mention of a loan type (personal, home, car, auto, business,\n'
        'education, gold, LAP, property) is ALWAYS clear enough to EXECUTE.\n'
        'Never ask for clarification on loan type queries.\n\n'
        'EXECUTE examples (clear intent — do not ask):\n'
        '  "find high-value customers for a personal loan" → EXECUTE\n'
        '  "find top 5 customers" → EXECUTE\n'
        '  "3 mvp for loan conversion" → EXECUTE\n'
        '  "customers in mumbai for personal loan" → EXECUTE\n'
        '  "find home loan customers" → EXECUTE\n'
        '  "show me car loan candidates" → EXECUTE\n'
        '  "find business loan prospects" → EXECUTE\n'
        '  "who qualifies for an education loan" → EXECUTE\n'
        '  "find gold loan customers" → EXECUTE\n'
        '  "show LAP eligible customers" → EXECUTE\n'
        '  "find gold loan customers with good credit" → EXECUTE\n\n'
        'CLARIFY examples (genuinely unknown — NON-BANKING jargon only):\n'
        '  "find the special ones" → CLARIFY: What criteria define special?\n'
        '  "xyz segment customers" → CLARIFY: What is xyz segment?\n\n'
        'Reply with ONLY: EXECUTE  — or —  CLARIFY: <one short question>'
    )

    try:
        llm = get_llm(temperature=0)
        resp = llm.invoke([
            SystemMessage(content="You decide whether a banking CRM query is clear or needs clarification."),
            HumanMessage(content=check_prompt),
        ])
        content = resp.content.strip()
        if content.upper().startswith("CLARIFY:"):
            question = content[len("CLARIFY:"):].strip()
            return True, question
        return False, ""
    except Exception:
        return False, ""  # on error, proceed rather than block


@timed_node
def data_agent_node(state: AgentState) -> dict:
    """Select and execute the right CRM query tool. Asks for clarification if intent is unclear."""

    # Clarification gate — ask before querying if the intent is ambiguous
    needs_clarify, question = _needs_clarification(state)
    if needs_clarify:
        return {
            "retrieved_customers": [],
            "final_response": question,
            "trace": [{
                "step": "data_agent",
                "decision": f"Query too vague — asking for clarification",
                "tools_called": [],
                "result_summary": f"Clarification requested: {question[:80]}",
                "skipped": False,
            }],
        }

    llm = get_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=get_prompt("data_agent.system"))] + human_messages_for_llm(state)

    response = llm.invoke(messages)

    tool_results: list[dict] = []
    tool_messages: list[ToolMessage] = []
    for tc in getattr(response, "tool_calls", []):
        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn:
            raw = tool_fn.invoke(tc["args"])
            tool_results.append({"tool": tc["name"], "result": raw})
            tool_messages.append(ToolMessage(content=raw, tool_call_id=tc["id"]))

    tools_called = [tc["name"] for tc in getattr(response, "tool_calls", [])]

    # If LLM still didn't call any tool, surface a helpful message
    if not tools_called:
        return {
            "retrieved_customers": [],
            "final_response": (
                "I wasn't sure how to interpret that request. Could you try:\n"
                "- *\"Find high-value customers for a personal loan\"*\n"
                "- *\"Which customers in Mumbai have no loan?\"*\n"
                "- *\"Find salary account holders with credit score above 700\"*"
            ),
            "trace": [{
                "step": "data_agent",
                "decision": "No tool matched — surfaced example queries",
                "tools_called": [],
                "result_summary": "0 customers — query not actionable",
                "skipped": False,
            }],
        }

    customers: list[dict] = []
    for tr in tool_results:
        try:
            data = json.loads(tr["result"])
            if "customers" in data:
                customers.extend(data["customers"])
            elif "id" in data:
                customers.append(data)
        except Exception:
            pass

    seen: set[str] = set()
    unique = [c for c in customers if not (c.get("id") in seen or seen.add(c.get("id")))]  # type: ignore[func-returns-value]

    return {
        "retrieved_customers": unique,
        "messages": [response] + tool_messages,
        "trace": [{
            "step": "data_agent",
            "decision": f"Called {tools_called[0]} based on RM request",
            "tools_called": tools_called,
            "result_summary": f"{len(unique)} customers retrieved",
            "skipped": False,
        }],
    }
