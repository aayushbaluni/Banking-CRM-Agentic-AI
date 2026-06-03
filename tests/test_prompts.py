"""
Verify prompt registry integrity:
- All expected prompt keys exist
- No prompt is empty or contains unresolved placeholder text
- Template prompts format correctly with valid substitution values
- Required structural sections are present per agent
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.prompts.registry import REGISTRY, get

EXPECTED_KEYS = [
    "supervisor.orchestrator",
    "supervisor.synthesis",
    "data_agent.system",
    "scoring_agent.system",
    "product_agent.system",
    "outreach_agent.system",
    "message_tools.whatsapp_generation",
    "scoring_tools.heuristic_reason",
]

FORBIDDEN_PLACEHOLDERS = [
    "[TASK_SLUG]",
    "[SPECIFIC EXPERT PERSONA",
    "[REQUIRED]",
    "[TYPE]",
]

REQUIRED_SECTIONS = {
    "supervisor.orchestrator": ["# Role", "# Mission", "# Reasoning Protocol", "# Constraints"],
    "data_agent.system": ["# Role", "# Mission", "# Tool Selection Rules", "# Constraints"],
    "scoring_agent.system": ["# Role", "# Mission", "# Reasoning Protocol"],
    "product_agent.system": ["# Role", "# Mission", "# Constraints"],
    "outreach_agent.system": ["# Role", "# Mission", "# Constraints"],
    "message_tools.whatsapp_generation": [
        "# Role", "# Mission", "# Reasoning Protocol",
        "# Examples", "# Hard Constraints", "# Self-Critique Checklist"
    ],
}


def test_all_expected_keys_present():
    for key in EXPECTED_KEYS:
        assert key in REGISTRY, f"Missing prompt key: '{key}'"


def test_no_prompt_is_empty():
    for key, prompt in REGISTRY.items():
        assert len(prompt.strip()) > 100, f"Prompt '{key}' is suspiciously short"


def test_no_unresolved_placeholders():
    for key, prompt in REGISTRY.items():
        for placeholder in FORBIDDEN_PLACEHOLDERS:
            assert placeholder not in prompt, (
                f"Prompt '{key}' contains unresolved placeholder: '{placeholder}'"
            )


def test_required_sections_present():
    for key, sections in REQUIRED_SECTIONS.items():
        prompt = REGISTRY[key]
        for section in sections:
            assert section in prompt, (
                f"Prompt '{key}' is missing required section: '{section}'"
            )


def test_get_function_returns_correct_prompt():
    for key in EXPECTED_KEYS:
        result = get(key)
        assert isinstance(result, str)
        assert len(result) > 0


def test_get_raises_on_unknown_key():
    with pytest.raises(KeyError) as exc_info:
        get("nonexistent.prompt")
    assert "registry" in str(exc_info.value)
    assert "Available prompts" in str(exc_info.value)


def test_whatsapp_template_formats_correctly():
    template = get("message_tools.whatsapp_generation")
    filled = template.format(
        customer_name="Rohan Mehta",
        product_name="Pre-Approved Personal Loan",
        occupation="Software Engineer",
        city="Bangalore",
        monthly_income=85000,
        account_balance=500000,
        has_salary_account=True,
        region="south",
        interest_rate=10.5,
        loan_amount=500000,
    )
    assert "Rohan Mehta" in filled
    assert "Pre-Approved Personal Loan" in filled
    assert "Software Engineer" in filled


def test_all_agent_prompts_mention_constraints():
    agent_keys = [
        "data_agent.system",
        "scoring_agent.system",
        "product_agent.system",
        "outreach_agent.system",
    ]
    for key in agent_keys:
        prompt = get(key)
        assert "NEVER" in prompt or "Hard" in prompt, (
            f"Prompt '{key}' has no hard constraints — add a Constraints section"
        )


def test_supervisor_prompt_mentions_state_awareness():
    prompt = get("supervisor.orchestrator")
    assert "AgentState" in prompt or "STATE" in prompt, (
        "Supervisor prompt must reference AgentState for stateful routing"
    )
