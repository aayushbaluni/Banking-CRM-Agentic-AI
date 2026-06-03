"""Unit tests for WhatsApp message compliance guardrails."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.guardrails import validate, enforce

CUSTOMER = {"name": "Rohan Mehta"}

# ── Passing messages ──────────────────────────────────────────────────────────

def test_valid_message_passes():
    msg = "Hi Rohan! Get ₹5L at 9.5% p.a. — tailored for you. Reply YES. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert ok, violations

def test_valid_message_under_300_chars():
    msg = "Hi Rohan! Big goals need smart funding. ₹5L at 9.5% p.a. Reply YES. Reply STOP to opt out."
    assert len(msg) < 300
    ok, _ = validate(msg, CUSTOMER)
    assert ok

# ── Failing messages ──────────────────────────────────────────────────────────

def test_missing_opt_out_fails():
    msg = "Hi Rohan! Get ₹5L at 9.5% p.a. Reply YES to confirm."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("opt-out" in v.lower() for v in violations)

def test_over_300_chars_fails():
    msg = "Hi Rohan! " + ("Get ₹5L at 9.5% p.a. — tailored for you. " * 10) + "Reply YES. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("300" in v for v in violations)

def test_missing_cta_fails():
    msg = "Hi Rohan! We have a great loan offer for you. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("call-to-action" in v.lower() for v in violations)

def test_guaranteed_approval_language_fails():
    msg = "Hi Rohan! You are 100% approved for ₹5L. Reply YES. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("guaranteed" in v.lower() or "approval" in v.lower() for v in violations)

def test_internal_id_in_message_fails():
    msg = "Hi Rohan! CUST0042 is eligible for ₹5L at 9.5%. Reply YES. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("internal" in v.lower() or "identifier" in v.lower() for v in violations)

def test_missing_customer_name_fails():
    msg = "Hi there! Get ₹5L at 9.5% p.a. Reply YES. Reply STOP to opt out."
    ok, violations = validate(msg, CUSTOMER)
    assert not ok
    assert any("Rohan" in v for v in violations)

# ── enforce() helper ──────────────────────────────────────────────────────────

def test_enforce_returns_structured_report():
    msg = "Hi Rohan! ₹5L at 9.5% p.a. Reply YES. Reply STOP to opt out."
    report = enforce(msg, CUSTOMER)
    assert "compliant" in report
    assert "char_count" in report
    assert "violations" in report
    assert report["char_count"] == len(msg)
