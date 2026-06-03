"""
WhatsApp message compliance guardrails.

Enforces banking-specific rules before messages reach the RM.
Violations are logged; non-compliant messages are flagged but not silently dropped.
"""
import re

MAX_CHARS = 300

# Phrases that imply guaranteed approval — not allowed without actual pre-approval flag
_GUARANTEED_APPROVAL = re.compile(
    r"\b(guaranteed|definitely approved|100%\s*approved|instant approval)\b",
    re.IGNORECASE,
)

# Internal data that must never surface in customer-facing messages
_INTERNAL_PATTERNS = re.compile(
    r"\b(CUST\d{4}|propensity|feature importance|credit risk score|internal)\b",
    re.IGNORECASE,
)

# Must-have elements
_HAS_OPT_OUT = re.compile(r"reply\s+stop\s+to\s+opt\s+out", re.IGNORECASE)
_HAS_CTA = re.compile(r"(reply\s+yes|call\s+\d|click\s+\w|tap\s+here)", re.IGNORECASE)


def validate(message: str, customer: dict | None = None) -> tuple[bool, list[str]]:
    """
    Validate a WhatsApp message against compliance rules.

    Returns:
        (is_compliant, list_of_violations)
    """
    violations: list[str] = []

    if len(message) > MAX_CHARS:
        violations.append(f"Exceeds {MAX_CHARS} chars ({len(message)} found)")

    if not _HAS_OPT_OUT.search(message):
        violations.append("Missing opt-out line ('Reply STOP to opt out')")

    if not _HAS_CTA.search(message):
        violations.append("Missing call-to-action (Reply YES / Call / Click)")

    if _GUARANTEED_APPROVAL.search(message):
        violations.append("Contains guaranteed-approval language (regulatory risk)")

    if _INTERNAL_PATTERNS.search(message):
        violations.append("Contains internal identifiers or jargon")

    if customer:
        name = customer.get("name", "")
        first_name = name.split()[0] if name else ""
        if first_name and first_name.lower() not in message.lower():
            violations.append(f"Missing customer first name ('{first_name}')")

    return len(violations) == 0, violations


def enforce(message: str, customer: dict | None = None) -> dict:
    """
    Run validation and return a structured compliance report.
    """
    compliant, violations = validate(message, customer)
    return {
        "message": message,
        "char_count": len(message),
        "compliant": compliant,
        "violations": violations,
    }
