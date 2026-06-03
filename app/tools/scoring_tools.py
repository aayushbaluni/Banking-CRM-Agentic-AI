"""Loan propensity scoring tools using a trained XGBoost model + heuristic fallback."""
import json
import os
import numpy as np
from langchain_core.tools import tool
from app.config import settings


_cached_model = None
_model_loaded = False


def _load_model():
    global _cached_model, _model_loaded
    if _model_loaded:
        return _cached_model
    import joblib
    if os.path.exists(settings.model_path):
        _cached_model = joblib.load(settings.model_path)
    _model_loaded = True
    return _cached_model


def _safe_num(val, default=0):
    """Coerce None/non-numeric to a safe default for feature vectors."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(val) -> int:
    """Coerce None/non-bool to 0/1 for feature vectors."""
    if val is None:
        return 0
    return int(bool(val))


def _extract_features(customer: dict) -> list[float]:
    """Convert customer dict to model feature vector (must match training order)."""
    edu_map = {"Undergraduate": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3, "Doctorate": 4}
    return [
        _safe_num(customer.get("monthly_income"), 0),
        _safe_num(customer.get("account_balance"), 0),
        _safe_num(customer.get("credit_score"), 600),
        _safe_num(customer.get("age"), 35),
        _safe_num(customer.get("num_products"), 0),
        _safe_bool(customer.get("has_salary_account")),
        _safe_bool(customer.get("has_fixed_deposit")),
        _safe_num(customer.get("months_as_customer"), 12),
        _safe_num(customer.get("avg_monthly_txn_amount"), 0),
        _safe_num(customer.get("num_monthly_txns"), 10),
        _safe_num(customer.get("last_product_purchase_months"), 12),
        float(edu_map.get(customer.get("education", "Graduate"), 1)),
    ]


_BUSINESS_OCCUPATIONS = {"Business Owner", "Consultant", "Architect", "Lawyer"}


def _heuristic_score(customer: dict) -> float:
    """Rule-based propensity score. Category-aware: reads loan_category from customer dict."""
    score = 0.0
    income = customer.get("monthly_income", 0)
    balance = customer.get("account_balance", 0)
    credit = customer.get("credit_score", 600)
    loan_category = customer.get("loan_category", "personal_loan")

    # Universal income/credit base
    if income > 100000:
        score += 0.25
    elif income > 50000:
        score += 0.15
    elif income > 25000:
        score += 0.08

    if credit >= 750:
        score += 0.25
    elif credit >= 700:
        score += 0.15
    elif credit >= 650:
        score += 0.08

    if customer.get("has_salary_account"):
        score += 0.15

    if balance > 1000000:
        score += 0.15
    elif balance > 300000:
        score += 0.08

    if customer.get("months_as_customer", 0) > 24:
        score += 0.10

    if customer.get("num_products", 0) >= 2:
        score += 0.05

    # Category-specific boosts and penalties
    if loan_category == "personal_loan":
        if customer.get("has_personal_loan"):
            score -= 0.30
    elif loan_category == "home_loan":
        if customer.get("has_home_loan"):
            score -= 0.25
        if customer.get("months_as_customer", 0) > 36:
            score += 0.08  # stable long-term customer
    elif loan_category == "auto_loan":
        if customer.get("has_auto_loan"):
            score -= 0.25
        if 25 <= customer.get("age", 35) <= 45:
            score += 0.05  # prime car-buying age
    elif loan_category == "business_loan":
        if customer.get("has_business_loan"):
            score -= 0.20
        if customer.get("occupation", "") in _BUSINESS_OCCUPATIONS:
            score += 0.10  # business owner profile
    elif loan_category == "education_loan":
        if customer.get("has_education_loan"):
            score -= 0.20
        if customer.get("age", 35) < 35:
            score += 0.08  # younger = student or young parent
    elif loan_category == "gold_loan":
        if customer.get("has_gold_loan"):
            score -= 0.20
    elif loan_category == "lap":
        if customer.get("has_home_loan"):
            score += 0.10  # property owner — strong LAP candidate
        if balance > 500000:
            score += 0.05

    return min(max(score, 0.0), 1.0)


def _score_reason(customer: dict, score: float) -> str:
    reasons = []
    credit = customer.get("credit_score", 0) or 0
    income = customer.get("monthly_income", 0) or 0
    if credit >= 750:
        reasons.append(f"excellent credit score ({credit})")
    elif credit >= 700:
        reasons.append(f"good credit score ({credit})")
    if income > 100000:
        reasons.append(f"high monthly income (₹{income:,.0f})")
    elif income > 50000:
        reasons.append(f"stable income (₹{income:,.0f}/month)")
    if customer.get("has_salary_account"):
        reasons.append("existing salary account relationship")
    if customer.get("account_balance", 0) > 500000:
        reasons.append(f"high account balance (₹{customer['account_balance']:,.0f})")
    if customer.get("months_as_customer", 0) > 36:
        reasons.append(f"loyal customer ({customer['months_as_customer']} months)")
    if not reasons:
        reasons.append("meets basic eligibility criteria")
    return "; ".join(reasons)


@tool
def score_loan_propensity(customer_json: str) -> str:
    """
    Score a customer's likelihood to convert for a personal loan.
    Uses an XGBoost model if available, falls back to rule-based heuristics.
    Returns a propensity score (0–1), tier (High/Medium/Low), and reason codes.

    Args:
        customer_json: JSON string of customer profile dict
    """
    try:
        customer = json.loads(customer_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid customer_json: {e}"})
    model = _load_model()
    loan_category = customer.get("loan_category", "personal_loan")

    # ML model was trained on personal loan data only; use heuristics for other categories
    if model is not None and loan_category == "personal_loan":
        features = np.array([_extract_features(customer)])
        score = min(max(float(model.predict_proba(features)[0][1]), 0.0), 1.0)
    else:
        score = _heuristic_score(customer)

    if score >= 0.65:
        tier = "High"
    elif score >= 0.35:
        tier = "Medium"
    else:
        tier = "Low"

    return json.dumps({
        "customer_id": customer.get("id"),
        "name": customer.get("name"),
        "propensity_score": round(score, 3),
        "tier": tier,
        "score_reason": _score_reason(customer, score),
        "scoring_method": "ml_model" if (model and loan_category == "personal_loan") else "heuristic",
    })


@tool
def batch_score_customers(customers_json: str, top_n: int = 10) -> str:
    """
    Score a list of customers and return the top N by propensity score.
    Efficient batch scoring for the supervisor to identify the best targets.

    Args:
        customers_json: JSON string of a list of customer profile dicts
        top_n: Number of top customers to return (default 10)
    """
    try:
        customers = json.loads(customers_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid customers_json: {e}", "results": []})
    if not isinstance(customers, list):
        return json.dumps({"error": "customers_json must be a JSON array", "results": []})
    model = _load_model()
    scored = []

    for customer in customers:
        loan_category = customer.get("loan_category", "personal_loan")
        if model is not None and loan_category == "personal_loan":
            features = np.array([_extract_features(customer)])
            score = min(max(float(model.predict_proba(features)[0][1]), 0.0), 1.0)
        else:
            score = _heuristic_score(customer)

        if score >= 0.65:
            tier = "High"
        elif score >= 0.35:
            tier = "Medium"
        else:
            tier = "Low"

        scored.append({
            "customer": customer,
            "propensity_score": round(score, 3),
            "tier": tier,
            "score_reason": _score_reason(customer, score),
        })

    scored.sort(key=lambda x: x["propensity_score"], reverse=True)
    top = scored[:top_n]

    return json.dumps({
        "total_scored": len(scored),
        "top_n": top_n,
        "results": top,
    })
