"""
Scoring Agent — scores retrieved customers by loan propensity.

LLM removed deliberately: this agent always calls batch_score_customers.
top_n is extracted from the RM's message so "find top 3" returns exactly 3.
"""
import re
import json
from langchain_core.messages import HumanMessage
from app.models.state import AgentState
from app.tools.scoring_tools import batch_score_customers
from app.agents.base import timed_node

_NUMBER_RE = re.compile(r"\b([1-9]|[1-2][0-9]|30)\b")


def _extract_requested_count(state: AgentState) -> int:
    """
    Extract the count the RM explicitly asked for (e.g. 'top 3', 'find 5').
    Returns 10 if no specific number is mentioned.
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            nums = _NUMBER_RE.findall(msg.content)
            if nums:
                return int(nums[0])
    return 10


@timed_node
def scoring_agent_node(state: AgentState) -> dict:
    """Directly invoke batch_score_customers — no LLM needed here."""
    customers = state.get("retrieved_customers", [])
    loan_category = state.get("loan_category", "personal_loan")
    if not customers:
        return {
            "scored_customers": [],
            "trace": [{
                "step": "scoring_agent",
                "decision": "Skipped — no customers in state to score",
                "tools_called": [],
                "result_summary": "0 customers scored",
                "skipped": True,
            }],
        }

    # Inject loan_category so scoring heuristics can apply category-specific logic
    customers_with_category = [{**c, "loan_category": loan_category} for c in customers]
    top_n = _extract_requested_count(state)

    raw = batch_score_customers.invoke({
        "customers_json": json.dumps(customers_with_category),
        "top_n": top_n,
    })
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {
            "scored_customers": [],
            "trace": [{
                "step": "scoring_agent",
                "decision": "batch_score_customers returned invalid JSON",
                "tools_called": ["batch_score_customers"],
                "result_summary": "0 customers scored (parse error)",
                "skipped": True,
            }],
        }
    scored = data.get("results", [])
    top_score = max((s.get("propensity_score", 0) for s in scored), default=0)
    min_score = min((s.get("propensity_score", 0) for s in scored), default=0)

    return {
        "scored_customers": scored,
        "trace": [{
            "step": "scoring_agent",
            "decision": (
                f"Scored {len(customers)} customers, returning top {top_n} "
                f"(RM requested {top_n})"
            ),
            "tools_called": ["batch_score_customers"],
            "result_summary": (
                f"Top {len(scored)} returned | "
                f"score range: {min_score:.0%}–{top_score:.0%}"
            ),
            "skipped": False,
        }],
    }
