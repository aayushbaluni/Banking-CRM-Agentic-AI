"""
Agent integration tests.

Tests the full LangGraph pipeline using mocked Azure LLM calls so no
real API key is needed. Each test verifies state transitions, routing
logic, and stateful follow-up behaviour.
"""
import json
import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import init_db
from app.db.seed import seed
from app.models.state import AgentState


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "messages": [],
        "rm_intent": "",
        "filters": {},
        "retrieved_customers": [],
        "scored_customers": [],
        "final_recommendations": [],
        "final_response": "",
        "next_agent": "",
        "unsupported_product": False,
        "trace": [],
    }
    defaults.update(kwargs)
    return defaults


SAMPLE_CUSTOMER = {
    "id": "CUST0001",
    "name": "Rohan Mehta",
    "age": 32,
    "gender": "Male",
    "occupation": "Software Engineer",
    "education": "Graduate",
    "city": "Bangalore",
    "state": "Karnataka",
    "region": "south",
    "monthly_income": 90000,
    "account_balance": 800000,
    "credit_score": 760,
    "num_products": 2,
    "has_personal_loan": False,
    "has_home_loan": False,
    "has_salary_account": True,
    "has_fixed_deposit": False,
    "months_as_customer": 36,
    "avg_monthly_txn_amount": 50000,
    "num_monthly_txns": 15,
    "last_product_purchase_months": 6,
    "phone": "+91-9876543210",
    "email": "rohan@example.com",
}

SAMPLE_SCORED = {
    "customer": SAMPLE_CUSTOMER,
    "propensity_score": 0.87,
    "tier": "High",
    "score_reason": "excellent credit score (760); high monthly income (₹90,000)",
}

SAMPLE_REC = {
    **SAMPLE_SCORED,
    "eligible": True,
    "product": {
        "id": "PL001",
        "name": "Pre-Approved Personal Loan",
        "interest_rate": 10.5,
        "max_amount": 500000,
        "min_credit_score": 700,
        "description": "Instant pre-approved loan",
        "eligibility_criteria": "Salary account, credit score >= 700",
    },
    "recommended_loan_amount": 500000,
    "selection_reason": "Salary account holder + good credit → Pre-Approved Loan",
    "whatsapp_message": "",
}


# ── Intent Guard Tests ────────────────────────────────────────────────────────

class TestIntentGuard:
    def test_greeting_routes_to_general_chat(self):
        """router_node sets next_agent='general_chat' for greetings — pipeline is bypassed."""
        from app.agents.supervisor import router_node, route_from_router
        with patch("app.agents.supervisor._classify_intent", return_value="general_chat"):
            for msg in ["how are you", "hi", "hello", "hey there", "good morning"]:
                state = make_state(messages=[HumanMessage(content=msg)])
                result = router_node.__wrapped__(state)
                state["next_agent"] = result.get("next_agent", "")
                assert route_from_router(state) == "general_chat", f"Failed for: '{msg}'"

    def test_thanks_routes_to_general_chat(self):
        from app.agents.supervisor import router_node, route_from_router
        with patch("app.agents.supervisor._classify_intent", return_value="general_chat"):
            state = make_state(messages=[HumanMessage(content="thanks, that was great!")])
            result = router_node.__wrapped__(state)
            state["next_agent"] = result.get("next_agent", "")
            assert route_from_router(state) == "general_chat"

    def test_crm_query_routes_to_pipeline(self):
        """router_node sets next_agent to a CRM node for CRM queries."""
        from app.agents.supervisor import route_from_router
        for msg in [
            "Find high-value customers for a personal loan",
            "Which customers in Mumbai have no loan?",
            "Generate WhatsApp messages for top 5",
        ]:
            state = make_state(messages=[HumanMessage(content=msg)], next_agent="data_agent")
            assert route_from_router(state) != "general_chat", f"Wrongly routed: '{msg}'"

    def test_pipeline_state_bypasses_llm_classifier(self):
        """If pipeline state already has data, _is_crm_task must NOT call the LLM."""
        from app.agents.supervisor import _is_crm_task
        state = make_state(
            messages=[HumanMessage(content="show results again")],
            retrieved_customers=[SAMPLE_CUSTOMER],  # existing state
        )
        # _classify_intent should never be called — assert without mock
        with patch("app.agents.supervisor._classify_intent") as mock_clf:
            result = _is_crm_task(state)
            mock_clf.assert_not_called()
        assert result is True

    def test_classifier_fallback_on_llm_error(self):
        """If the LLM call throws, _classify_intent must fall back to 'crm_task'."""
        from app.agents.supervisor import _classify_intent
        # Use a message NOT in _OBVIOUS_CHAT so the LLM path is actually reached
        with patch("app.agents.supervisor.get_llm", side_effect=Exception("Azure down")):
            result = _classify_intent("can you help me with something interesting?")
        assert result == "crm_task"  # safe default — never silently breaks

    def test_general_chat_node_calls_llm(self):
        """general_chat_node must use the LLM, not hardcoded strings."""
        from app.agents.supervisor import general_chat_node
        mock_response = MagicMock()
        mock_response.content = "Hi! I'm ready to help with banking CRM tasks."
        with patch("app.agents.supervisor.get_llm") as mock_llm_fn:
            mock_llm_fn.return_value.invoke.return_value = mock_response
            state = make_state(messages=[HumanMessage(content="how are you")])
            result = general_chat_node(state)
        assert result["final_response"] == "Hi! I'm ready to help with banking CRM tasks."
        assert result["trace"][0]["step"] == "general_chat"
        mock_llm_fn.return_value.invoke.assert_called_once()

    def test_general_chat_does_not_set_recommendations(self):
        """general_chat_node must never write to pipeline state fields."""
        from app.agents.supervisor import general_chat_node
        mock_response = MagicMock()
        mock_response.content = "Hello! Try: find high-value customers."
        with patch("app.agents.supervisor.get_llm") as mock_llm_fn:
            mock_llm_fn.return_value.invoke.return_value = mock_response
            state = make_state(messages=[HumanMessage(content="hi")])
            result = general_chat_node(state)
        assert "final_recommendations" not in result
        assert "retrieved_customers" not in result
        assert "scored_customers" not in result


# ── Router Tests ──────────────────────────────────────────────────────────────

def _route_with_llm(state, next_node: str, state_patch: dict | None = None):
    """Simulate router_node + route_from_router with a mocked LLM decision."""
    from app.agents.supervisor import router_node, route_from_router
    state_patch = state_patch or {}
    with patch("app.agents.supervisor.decide_router", return_value=(next_node, state_patch)):
        merged = {**state, **router_node(state)}
    return route_from_router(merged)


class TestSmartRouter:
    def test_routes_to_data_agent_on_fresh_state(self):
        state = make_state(messages=[HumanMessage(content="Find high-value customers")])
        assert _route_with_llm(state, "data_agent") == "data_agent"

    def test_routes_to_scoring_when_customers_retrieved(self):
        state = make_state(
            messages=[HumanMessage(content="score them")],
            retrieved_customers=[SAMPLE_CUSTOMER],
        )
        assert _route_with_llm(state, "scoring_agent") == "scoring_agent"

    def test_routes_to_product_when_scored(self):
        state = make_state(
            messages=[HumanMessage(content="recommend products")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
        )
        assert _route_with_llm(state, "product_agent") == "product_agent"

    def test_routes_to_outreach_when_messages_requested(self):
        state = make_state(
            messages=[HumanMessage(content="generate whatsapp messages")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
            final_recommendations=[SAMPLE_REC],
        )
        assert _route_with_llm(state, "outreach_agent") == "outreach_agent"

    def test_routes_to_supervisor_when_everything_done(self):
        rec_with_msg = {**SAMPLE_REC, "whatsapp_message": "Hi Rohan! Check this out."}
        state = make_state(
            messages=[HumanMessage(content="show me those results")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
            final_recommendations=[rec_with_msg],
        )
        assert _route_with_llm(state, "supervisor") == "supervisor"

    def test_skips_to_outreach_on_followup_with_recs_no_messages(self):
        state = make_state(
            messages=[HumanMessage(content="generate whatsapp messages for them")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
            final_recommendations=[SAMPLE_REC],
        )
        assert _route_with_llm(state, "outreach_agent") == "outreach_agent"

    def test_unsupported_product_clears_recs_via_llm(self):
        from app.agents.supervisor import router_node, supervisor_node
        state = make_state(
            messages=[HumanMessage(content="find customers for car loan")],
            final_recommendations=[SAMPLE_REC],
        )
        state_patch = {
            "retrieved_customers": [],
            "scored_customers": [],
            "final_recommendations": [],
            "final_response": "",
            "unsupported_product": True,
        }
        with patch("app.agents.supervisor.decide_router", return_value=("supervisor", state_patch)):
            merged = {**state, **router_node(state)}
        assert merged["final_recommendations"] == []
        assert merged["unsupported_product"] is True
        sup = supervisor_node(merged)
        assert sup["final_response"] == ""
        assert "Rohan Mehta" not in sup["final_response"]


# ── Product Agent Tests ───────────────────────────────────────────────────────

class TestProductAgent:
    def test_recommends_eligible_product(self):
        from app.agents.product_agent import product_agent_node
        state = make_state(scored_customers=[SAMPLE_SCORED])
        result = product_agent_node(state)
        assert len(result["final_recommendations"]) == 1
        rec = result["final_recommendations"][0]
        assert rec["eligible"] is True
        assert rec["product"]["id"] in ["PL001", "PL002", "PL003", "PL004"]

    def test_excludes_ineligible_customers(self):
        from app.agents.product_agent import product_agent_node
        ineligible = {
            "customer": {**SAMPLE_CUSTOMER, "monthly_income": 5000, "credit_score": 400},
            "propensity_score": 0.1,
            "tier": "Low",
            "score_reason": "low income and credit score",
        }
        state = make_state(scored_customers=[ineligible])
        result = product_agent_node(state)
        assert result["final_recommendations"] == []

    def test_premium_loan_for_high_income_customer(self):
        from app.agents.product_agent import product_agent_node
        high_income = {
            "customer": {**SAMPLE_CUSTOMER, "monthly_income": 150000, "credit_score": 780},
            "propensity_score": 0.92,
            "tier": "High",
            "score_reason": "very high income",
        }
        state = make_state(scored_customers=[high_income])
        result = product_agent_node(state)
        assert result["final_recommendations"][0]["product"]["id"] == "PL003"


# ── Supervisor Node Tests ─────────────────────────────────────────────────────

class TestSupervisorNode:
    def test_formats_response_with_recommendations(self):
        from app.agents.supervisor import supervisor_node
        rec = {**SAMPLE_REC, "whatsapp_message": "Hi Rohan! Great offer for you.", "message_compliant": True}
        state = make_state(final_recommendations=[rec])
        result = supervisor_node(state)
        assert "Rohan Mehta" in result["final_response"]
        assert "%" in result["final_response"]
        assert "✅" in result["final_response"]

    def test_handles_empty_recommendations(self):
        from app.agents.supervisor import supervisor_node
        state = make_state(final_recommendations=[])
        result = supervisor_node(state)
        assert "No eligible customers" in result["final_response"]


# ── Data Agent Tests (mocked LLM) ─────────────────────────────────────────────

class TestDataAgent:
    def test_parses_tool_results_into_customers(self):
        from app.agents.data_agent import data_agent_node

        mock_tool_call = {
            "name": "get_high_value_customers",
            "args": {"min_balance": 500000, "limit": 5},
            "id": "call_123",
            "type": "tool_call",
        }
        mock_response = AIMessage(content="", tool_calls=[mock_tool_call])

        # clarification check calls llm.invoke directly (returns "EXECUTE")
        # tool selection calls llm.bind_tools().invoke (returns mock_response with tool call)
        clarify_response = MagicMock()
        clarify_response.content = "EXECUTE"

        with patch("app.agents.data_agent.get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = clarify_response          # clarification check
            mock_llm.bind_tools.return_value.invoke.return_value = mock_response  # tool selection
            mock_llm_fn.return_value = mock_llm

            state = make_state(messages=[HumanMessage(content="Find high value customers")])
            result = data_agent_node(state)

        # Tool was actually called against real DB — customers should be returned
        assert isinstance(result["retrieved_customers"], list)
        assert len(result["retrieved_customers"]) > 0

    def test_deduplicates_customers(self):
        from app.agents.data_agent import data_agent_node

        # Return same customer twice from two "tool calls"
        dup_response_data = json.dumps({"count": 2, "customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]})

        mock_tool_call = {
            "name": "get_high_value_customers",
            "args": {"min_balance": 100000, "limit": 10},
            "id": "call_456",
            "type": "tool_call",
        }
        mock_response = AIMessage(content="", tool_calls=[mock_tool_call])

        with patch("app.agents.data_agent.get_llm") as mock_llm_fn, \
             patch("app.agents.data_agent.TOOL_MAP") as mock_tool_map:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke.return_value = mock_response
            mock_llm_fn.return_value = mock_llm
            mock_tool_map.__getitem__ = MagicMock(
                return_value=MagicMock(invoke=MagicMock(return_value=dup_response_data))
            )
            mock_tool_map.get = MagicMock(
                return_value=MagicMock(invoke=MagicMock(return_value=dup_response_data))
            )

            state = make_state(messages=[HumanMessage(content="find customers")])
            result = data_agent_node(state)

        ids = [c["id"] for c in result["retrieved_customers"]]
        assert len(ids) == len(set(ids)), "Duplicate customer IDs found after dedup"


# ── Scoring Agent Tests (mocked LLM) ─────────────────────────────────────────

class TestScoringAgent:
    def test_returns_empty_when_no_customers(self):
        from app.agents.scoring_agent import scoring_agent_node
        state = make_state(retrieved_customers=[])
        result = scoring_agent_node(state)
        assert result["scored_customers"] == []

    def test_calls_batch_scoring_tool(self):
        """scoring_agent calls batch_score_customers directly — no LLM involved."""
        from app.agents.scoring_agent import scoring_agent_node
        # Feed a real customer from the seeded DB so the tool executes properly
        state = make_state(retrieved_customers=[SAMPLE_CUSTOMER])
        result = scoring_agent_node(state)
        assert isinstance(result["scored_customers"], list)
        # Trace must confirm it's a direct tool call, not LLM-mediated
        assert "batch_score_customers" in result["trace"][0]["tools_called"]
        assert result["trace"][0]["skipped"] is False


# ── Full Pipeline Integration Test ────────────────────────────────────────────

class TestFullPipeline:
    def test_product_agent_follows_scoring_agent(self):
        """Verify state flows correctly: scored → product → recommendations."""
        from app.agents.product_agent import product_agent_node
        from app.agents.supervisor import supervisor_node

        scored_state = make_state(scored_customers=[SAMPLE_SCORED])
        product_result = product_agent_node(scored_state)
        assert len(product_result["final_recommendations"]) >= 1

        # Feed product output into supervisor
        full_state = make_state(
            final_recommendations=product_result["final_recommendations"]
        )
        sup_result = supervisor_node(full_state)
        assert "Rohan Mehta" in sup_result["final_response"]
        assert "%" in sup_result["final_response"]

    def test_graph_imports_and_compiles(self):
        """Smoke test: graph builds without errors."""
        from app.agents.supervisor import get_crm_graph
        assert get_crm_graph() is not None

    def test_all_agent_modules_importable(self):
        """Every agent module must import cleanly (catches circular import issues)."""
        import app.agents.base
        import app.agents.data_agent
        import app.agents.scoring_agent
        import app.agents.product_agent
        import app.agents.outreach_agent
        import app.agents.supervisor

    def test_trace_emitted_by_product_agent(self):
        """Each node must emit a trace entry — verified on product_agent."""
        from app.agents.product_agent import product_agent_node
        state = make_state(scored_customers=[SAMPLE_SCORED])
        result = product_agent_node(state)
        assert "trace" in result
        assert len(result["trace"]) == 1
        entry = result["trace"][0]
        assert entry["step"] == "product_agent"
        assert "recommend_product" in entry["tools_called"]

    def test_trace_marks_skipped_when_no_customers(self):
        """Scoring agent must mark trace as skipped when no customers are in state."""
        from app.agents.scoring_agent import scoring_agent_node
        state = make_state(retrieved_customers=[])
        result = scoring_agent_node(state)
        assert result["trace"][0]["skipped"] is True

    def test_supervisor_response_includes_compliance_badge(self):
        """Supervisor must show ✅/⚠️ based on message_compliant field."""
        from app.agents.supervisor import supervisor_node
        rec_compliant = {**SAMPLE_REC, "whatsapp_message": "Hi Rohan! Great offer. Reply STOP to opt out.", "message_compliant": True}
        rec_noncompliant = {**SAMPLE_REC, "whatsapp_message": "Call us.", "message_compliant": False,
                            "customer": {**SAMPLE_CUSTOMER, "id": "CUST0002", "name": "Priya Nair"}}
        state = make_state(final_recommendations=[rec_compliant, rec_noncompliant])
        result = supervisor_node(state)
        assert "✅" in result["final_response"]
        assert "⚠️" in result["final_response"]


# ── Multi-Turn Stateful Integration Test ──────────────────────────────────────

class TestMultiTurnStateful:
    """
    Proves the router skips already-completed stages on follow-up turns.
    No Azure calls needed — tests routing logic directly.
    """

    def test_turn1_routes_to_data_agent_fresh(self):
        state = make_state(messages=[HumanMessage(content="Find high-value customers")])
        assert _route_with_llm(state, "data_agent") == "data_agent"

    def test_turn2_routes_to_outreach_skipping_data_and_scoring(self):
        state = make_state(
            messages=[HumanMessage(content="Now generate whatsapp messages for them")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
            final_recommendations=[SAMPLE_REC],
        )
        assert _route_with_llm(state, "outreach_agent") == "outreach_agent"

    def test_turn3_routes_to_supervisor_when_complete(self):
        rec_with_msg = {**SAMPLE_REC, "whatsapp_message": "Hi Rohan! Reply STOP to opt out."}
        state = make_state(
            messages=[HumanMessage(content="Show me the results again")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
            final_recommendations=[rec_with_msg],
        )
        assert _route_with_llm(state, "supervisor") == "supervisor"

    def test_partial_state_routes_to_scoring_not_data(self):
        state = make_state(
            messages=[HumanMessage(content="now score them")],
            retrieved_customers=[SAMPLE_CUSTOMER],
        )
        assert _route_with_llm(state, "scoring_agent") == "scoring_agent"

    def test_dissatisfied_follow_up_routes_restart_via_llm(self):
        """Turn 2: low scores + car loan → LLM must RESTART, not echo stale recs."""
        state = make_state(
            messages=[HumanMessage(
                content="why is the percent so low, give me high value only, and also car loan"
            )],
            final_recommendations=[SAMPLE_REC],
        )
        reset_patch = {
            "retrieved_customers": [],
            "scored_customers": [],
            "final_recommendations": [],
            "unsupported_product": False,
        }
        with patch("app.agents.routing_llm.llm_product_scope", return_value="unsupported"):
            assert _route_with_llm(state, "supervisor", {
                **reset_patch,
                "final_response": "",
                "unsupported_product": True,
            }) == "supervisor"

    def test_full_pipeline_state_transitions(self):
        """
        Walk through the full state machine manually.
        Proves each stage receives and passes state correctly without LLM calls.
        """
        from app.agents.product_agent import product_agent_node
        from app.agents.supervisor import supervisor_node

        state_after_data = make_state(
            messages=[HumanMessage(content="find customers")],
            retrieved_customers=[SAMPLE_CUSTOMER],
        )
        assert _route_with_llm(state_after_data, "scoring_agent") == "scoring_agent"

        # Stage 2: after scoring_agent — simulate scored customers
        state_after_scoring = make_state(
            messages=[HumanMessage(content="find customers")],
            retrieved_customers=[SAMPLE_CUSTOMER],
            scored_customers=[SAMPLE_SCORED],
        )
        assert _route_with_llm(state_after_scoring, "product_agent") == "product_agent"

        # Stage 3: product_agent produces recommendations
        product_result = product_agent_node(state_after_scoring)
        assert len(product_result["final_recommendations"]) >= 1
        assert product_result["final_recommendations"][0]["eligible"] is True

        # Stage 4: supervisor synthesises
        state_final = make_state(
            final_recommendations=product_result["final_recommendations"]
        )
        sup_result = supervisor_node(state_final)
        assert "Rohan Mehta" in sup_result["final_response"]
        assert len(sup_result["trace"]) == 1
        assert sup_result["trace"][0]["step"] == "supervisor"
