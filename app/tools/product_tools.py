"""Product recommendation tools — rule engine + DB lookup."""
import json
from langchain_core.tools import tool
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.schema import Product


def _get_product(db: Session, product_id: str) -> dict | None:
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        return None
    return {
        "id": p.id,
        "name": p.name,
        "category": p.category,
        "interest_rate": p.interest_rate,
        "max_amount": p.max_amount,
        "min_credit_score": p.min_credit_score,
        "description": p.description,
        "eligibility_criteria": p.eligibility_criteria,
    }


def _offered_rate(base_rate: float, credit_score: int) -> float:
    """Adjust base rate upward for credit risk. Higher credit = lower offered rate."""
    if credit_score >= 800:
        premium = -0.25
    elif credit_score >= 750:
        premium = 0.0
    elif credit_score >= 700:
        premium = 0.25
    elif credit_score >= 650:
        premium = 0.50
    elif credit_score >= 600:
        premium = 0.75
    else:
        premium = 1.00
    return round(base_rate + premium, 2)


_INCOME_MULTIPLIERS = {
    "personal_loan": 24,
    "home_loan": 60,
    "auto_loan": 12,
    "business_loan": 36,
    "education_loan": 20,
    "gold_loan": 5,
    "lap": 60,
}

_BUSINESS_OCCUPATIONS = {"Business Owner", "Consultant", "Architect", "Lawyer"}


def _personal_loan_match(income, credit, has_salary, age):
    if income >= 100000 and credit >= 750:
        return "PL003", "High income + excellent credit → Premium Personal Loan (lowest rate)"
    if has_salary and credit >= 700:
        return "PL001", "Salary account holder + good credit → Pre-Approved Loan (instant disbursal)"
    if age <= 40 and income >= 40000 and credit >= 680:
        return "PL004", "Young professional profile → Flexi Personal Loan (revolving credit)"
    if income >= 25000 and credit >= 650:
        return "PL002", "Meets standard eligibility → Standard Personal Loan"
    return None, None


def _home_loan_match(income, credit):
    if income >= 50000 and credit >= 700:
        return "HL001", "Income ≥ ₹50k + strong credit → Prime Home Loan (8.5% p.a.)"
    if income >= 35000 and credit >= 660:
        return "HL002", "Meets first-home buyer criteria → Affordable Home Loan (9.2% p.a.)"
    return None, None


def _auto_loan_match(income, credit):
    if income >= 35000 and credit >= 680:
        return "AL001", "Stable income + good credit → New Car Loan (9.0% p.a.)"
    if income >= 25000 and credit >= 640:
        return "AL002", "Meets used-car loan criteria → Used Car Loan (11.5% p.a.)"
    return None, None


def _business_loan_match(income, credit, occupation):
    if income >= 80000 and credit >= 700 and occupation in _BUSINESS_OCCUPATIONS:
        return "BL001", "High-income entrepreneur + good credit → SME Business Loan (11.5% p.a.)"
    if income >= 40000 and credit >= 640:
        return "BL002", "Meets micro-business criteria → Micro Business Loan (14.0% p.a.)"
    return None, None


def _education_loan_match(income, credit):
    if income >= 20000 and credit >= 600:
        return "EDL001", "Meets co-applicant criteria → Education Loan (9.5% p.a.)"
    return None, None


def _gold_loan_match(income, credit):
    if income >= 15000 and credit >= 550:
        return "GL001", "Low threshold secured product → Gold Loan (10.5% p.a.)"
    return None, None


def _lap_match(income, credit):
    if income >= 50000 and credit >= 680:
        return "LAP001", "Property owner with strong profile → Loan Against Property (9.0% p.a.)"
    return None, None


@tool
def recommend_product(scored_customer_json: str) -> str:
    """
    Recommend the most suitable loan product for a customer based on their profile
    and the requested loan category. Uses a deterministic rule engine for compliance.
    Supports: personal_loan, home_loan, auto_loan, business_loan, education_loan,
    gold_loan, lap.

    Args:
        scored_customer_json: JSON string containing customer profile, propensity_score,
                              and optional loan_category field
    """
    try:
        data = json.loads(scored_customer_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid scored_customer_json: {e}", "eligible": False})
    customer = data.get("customer", data)
    score = data.get("propensity_score", 0.5)
    loan_category = data.get("loan_category", "personal_loan")

    income = customer.get("monthly_income", 0)
    credit = customer.get("credit_score", 600)
    has_salary = customer.get("has_salary_account", False)
    age = customer.get("age", 35)
    occupation = customer.get("occupation", "")

    dispatch = {
        "personal_loan": lambda: _personal_loan_match(income, credit, has_salary, age),
        "home_loan": lambda: _home_loan_match(income, credit),
        "auto_loan": lambda: _auto_loan_match(income, credit),
        "business_loan": lambda: _business_loan_match(income, credit, occupation),
        "education_loan": lambda: _education_loan_match(income, credit),
        "gold_loan": lambda: _gold_loan_match(income, credit),
        "lap": lambda: _lap_match(income, credit),
    }
    matcher = dispatch.get(loan_category, dispatch["personal_loan"])
    product_id, reason = matcher()

    if not product_id:
        return json.dumps({
            "customer_id": customer.get("id"),
            "eligible": False,
            "reason": f"Customer does not meet minimum eligibility for {loan_category}.",
        })

    db: Session = SessionLocal()
    try:
        product = _get_product(db, product_id)
        if not product:
            return json.dumps({
                "customer_id": customer.get("id"),
                "eligible": False,
                "reason": f"Product {product_id} not found in catalogue.",
            })
        multiplier = _INCOME_MULTIPLIERS.get(loan_category, 24)
        max_loan = round(income * multiplier, 2)
        offered_rate = _offered_rate(product["interest_rate"], credit)
        return json.dumps({
            "customer_id": customer.get("id"),
            "customer_name": customer.get("name"),
            "eligible": True,
            "product": product,
            "recommended_loan_amount": min(max_loan, product["max_amount"]),
            "offered_interest_rate": offered_rate,
            "selection_reason": reason,
            "propensity_score": score,
        })
    finally:
        db.close()


@tool
def list_available_products() -> str:
    """
    Return all loan products available in the bank's product catalogue across all
    categories (personal, home, auto, business, education, gold, LAP), including
    eligibility criteria and interest rates.
    """
    db: Session = SessionLocal()
    try:
        products = db.query(Product).all()
        result = [
            {
                "id": p.id,
                "name": p.name,
                "interest_rate": p.interest_rate,
                "max_amount": p.max_amount,
                "min_credit_score": p.min_credit_score,
                "description": p.description,
                "eligibility_criteria": p.eligibility_criteria,
            }
            for p in products
        ]
        return json.dumps({"products": result})
    finally:
        db.close()
