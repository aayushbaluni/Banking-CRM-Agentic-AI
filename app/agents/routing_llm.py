"""
LLM-only routing decisions for the CRM agent graph.

No keyword heuristics for RESTART/CONTINUE/OUTREACH or product scope.
Social greetings still use a minimal exact-match fast path in supervisor._classify_intent only.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.state import AgentState
from app.agents.base import get_llm

_SUPPORTED = (
    "Personal loans (Premium PL003, Pre-Approved PL001, Flexi PL004, Standard PL002), "
    "Home loans (Prime HL001, Affordable HL002), "
    "Car/Auto loans (New Car AL001, Used Car AL002), "
    "Business loans (SME BL001, Micro BL002), "
    "Education loans (EDL001), "
    "Gold loans (GL001), "
    "Loan Against Property / LAP (LAP001)."
)
_UNSUPPORTED = (
    "Two-wheeler loan, credit card, overdraft, forex, mutual fund, insurance policy, "
    "microfinance, peer-to-peer lending."
)

_PRODUCT_SCOPE_SYSTEM = f"""You classify which loan category the RM is asking about.

SUPPORTED categories — reply with the EXACT label shown:
- Personal loan (any variant, generic "loan", unclear) → PERSONAL_LOAN
- Home loan / housing loan / mortgage → HOME_LOAN
- Car loan / auto loan / vehicle loan (4-wheeler) → AUTO_LOAN
- Business loan / SME loan / working capital / MSME → BUSINESS_LOAN
- Education loan / student loan → EDUCATION_LOAN
- Gold loan / loan against gold → GOLD_LOAN
- Loan against property / LAP / mortgage against property → LAP

NOT SUPPORTED: {_UNSUPPORTED}
If the RM's primary intent is an unsupported product → reply UNSUPPORTED

Reply with ONLY one word: PERSONAL_LOAN / HOME_LOAN / AUTO_LOAN / BUSINESS_LOAN / EDUCATION_LOAN / GOLD_LOAN / LAP / UNSUPPORTED"""


_FOLLOW_UP_SYSTEM = f"""You classify how a Relationship Manager's follow-up message should be handled.

The RM already has customer recommendations on screen from a prior search.
Supported loan categories: personal, home, car/auto, business, education, gold, LAP.
NOT SUPPORTED: {_UNSUPPORTED}

Reply with ONLY one word:

RESTART — new search, different filters, higher quality only, dissatisfaction with scores,
          different city/segment, different count, different loan category (e.g. switching
          from personal loans to home loans), or any change that requires re-querying CRM
CONTINUE — view, re-display, or discuss the EXACT same list already shown (no new search)
OUTREACH — generate WhatsApp/outreach messages for the current recommendations
UNSUPPORTED — RM now wants a truly unsupported product (credit card, two-wheeler, insurance)

Critical:
- "scores too low", "high value only", "mvp only", "find better" → RESTART
- "what about car loans?" or "show me home loan customers" → RESTART (category switch)
- "generate messages", "whatsapp for them" → OUTREACH
- "show those results again" → CONTINUE"""


_OUTREACH_SYSTEM = """Does the RM EXPLICITLY ask for WhatsApp messages, outreach messages,
or message generation in their query?

YES examples: "generate WhatsApp messages", "send messages", "create outreach",
"write messages for them", "draft WhatsApp for top 5"

NO examples: "find customers for personal loan", "show me car loan candidates",
"find high-value customers likely to convert this month", "who qualifies for education loan"

IMPORTANT: Phrases like "likely to convert", "this month", "high-potential" are about
FINDING customers, NOT about generating messages. Only say YES if the RM explicitly
mentions messages, WhatsApp, or outreach.

Reply ONLY: YES or NO"""


def _last_human_message(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        "",
    )


def _format_results_context(state: AgentState) -> str:
    recs = state.get("final_recommendations", [])
    if not recs:
        return "No recommendations on screen yet."
    lines = [f"{len(recs)} recommendations currently shown:"]
    for i, rec in enumerate(recs[:10], 1):
        c = rec.get("customer", {})
        product = rec.get("product", {})
        lines.append(
            f"  {i}. {c.get('name', 'N/A')} | score {rec.get('propensity_score', 0):.0%} "
            f"({rec.get('tier', '?')}) | {product.get('name', 'loan')}"
        )
    return "\n".join(lines)


def _invoke_label(system: str, user: str, allowed: list[str], default: str) -> str:
    try:
        llm = get_llm(temperature=0)
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        text = (resp.content or "").strip().upper()
        token = text.split()[0] if text else ""
        for label in allowed:
            if token == label.upper() or text.startswith(label.upper()):
                return label
        return default
    except Exception:
        return default


_CATEGORY_LABELS = [
    "PERSONAL_LOAN", "HOME_LOAN", "AUTO_LOAN", "BUSINESS_LOAN",
    "EDUCATION_LOAN", "GOLD_LOAN", "LAP", "UNSUPPORTED",
]

_LABEL_TO_CATEGORY = {
    "PERSONAL_LOAN": "personal_loan",
    "HOME_LOAN": "home_loan",
    "AUTO_LOAN": "auto_loan",
    "BUSINESS_LOAN": "business_loan",
    "EDUCATION_LOAN": "education_loan",
    "GOLD_LOAN": "gold_loan",
    "LAP": "lap",
    "UNSUPPORTED": "unsupported",
}


def llm_product_scope(message: str) -> str:
    """Returns the loan category slug (e.g. 'auto_loan') or 'unsupported'."""
    label = _invoke_label(
        _PRODUCT_SCOPE_SYSTEM,
        f'RM message: "{message}"',
        _CATEGORY_LABELS,
        "PERSONAL_LOAN",
    )
    return _LABEL_TO_CATEGORY.get(label, "personal_loan")


def llm_follow_up_action(state: AgentState) -> str:
    """
    Returns 'restart', 'continue', 'outreach', or 'unsupported'.
    Default on error: restart (never echo stale results).
    """
    message = _last_human_message(state)
    context = _format_results_context(state)
    user = f"{context}\n\nRM follow-up message:\n\"{message}\""

    raw = _invoke_label(
        _FOLLOW_UP_SYSTEM,
        user,
        ["RESTART", "CONTINUE", "OUTREACH", "UNSUPPORTED"],
        "RESTART",
    )
    return raw.lower()


def llm_wants_outreach(message: str) -> bool:
    """After product_agent in the same turn — LLM decides if outreach is needed."""
    label = _invoke_label(
        _OUTREACH_SYSTEM,
        f'RM message: "{message}"',
        ["YES", "NO"],
        "NO",
    )
    return label == "YES"


def decide_router(state: AgentState) -> tuple[str, dict]:
    """
    LLM-driven router. Returns (next_node_name, state_patch).

    next_node: general_chat | data_agent | scoring_agent | product_agent |
               outreach_agent | supervisor

    Follow-up actions (OUTREACH/CONTINUE/RESTART/UNSUPPORTED) are checked
    BEFORE product scope so that "generate messages" isn't misclassified
    as unsupported when no loan category is mentioned.
    """
    message = _last_human_message(state)
    recs = state.get("final_recommendations", [])

    # ── Follow-up turn: existing recommendations on screen ──────────────
    if recs:
        action = llm_follow_up_action(state)
        if action == "unsupported":
            return "supervisor", {
                "retrieved_customers": [],
                "scored_customers": [],
                "final_recommendations": [],
                "final_response": "",
                "unsupported_product": True,
            }
        if action == "outreach":
            return "outreach_agent", {"unsupported_product": False}
        if action == "continue":
            return "supervisor", {"unsupported_product": False}
        # restart — classify category for the new search
        scope = llm_product_scope(message)
        if scope == "unsupported":
            return "supervisor", {
                "retrieved_customers": [],
                "scored_customers": [],
                "final_recommendations": [],
                "final_response": "",
                "unsupported_product": True,
            }
        return "data_agent", {
            "retrieved_customers": [],
            "scored_customers": [],
            "final_recommendations": [],
            "final_response": "",
            "unsupported_product": False,
            "loan_category": scope,
        }

    # ── Fresh turn: no recommendations yet ──────────────────────────────
    scope = llm_product_scope(message)
    if scope == "unsupported":
        return "supervisor", {
            "retrieved_customers": [],
            "scored_customers": [],
            "final_recommendations": [],
            "final_response": "",
            "unsupported_product": True,
        }

    if state.get("scored_customers"):
        return "product_agent", {"unsupported_product": False, "loan_category": scope}
    if state.get("retrieved_customers"):
        return "scoring_agent", {"unsupported_product": False, "loan_category": scope}
    return "data_agent", {"unsupported_product": False, "loan_category": scope}
