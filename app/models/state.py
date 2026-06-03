from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def _append_trace(left: list, right: list) -> list:
    """Reducer: always appends trace entries, never overwrites."""
    return (left or []) + (right or [])


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    rm_intent: str
    filters: dict[str, Any]
    retrieved_customers: list[dict]
    scored_customers: list[dict]
    final_recommendations: list[dict]
    final_response: str
    next_agent: str
    unsupported_product: bool
    loan_category: str  # personal_loan | home_loan | auto_loan | business_loan | education_loan | gold_loan | lap
    # Agent execution trace — each node appends one entry
    trace: Annotated[list[dict], _append_trace]
