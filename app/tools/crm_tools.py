"""CRM database query tools — every function is a real SQLAlchemy query."""
import json
from langchain_core.tools import tool
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.schema import Customer, Transaction, LoanApplication

_MAX_LIMIT = 100


def _clamp_limit(limit: int) -> int:
    """Prevent DoS via absurd limit values from LLM tool calls."""
    return max(1, min(int(limit), _MAX_LIMIT))


def _customer_to_dict(c: Customer) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "age": c.age,
        "gender": c.gender,
        "occupation": c.occupation,
        "education": c.education,
        "city": c.city,
        "state": c.state,
        "region": c.region,
        "monthly_income": c.monthly_income,
        "account_balance": c.account_balance,
        "credit_score": c.credit_score,
        "num_products": c.num_products,
        "has_personal_loan": c.has_personal_loan,
        "has_home_loan": c.has_home_loan,
        "has_auto_loan": c.has_auto_loan,
        "has_business_loan": c.has_business_loan,
        "has_education_loan": c.has_education_loan,
        "has_gold_loan": c.has_gold_loan,
        "has_salary_account": c.has_salary_account,
        "has_fixed_deposit": c.has_fixed_deposit,
        "months_as_customer": c.months_as_customer,
        "avg_monthly_txn_amount": c.avg_monthly_txn_amount,
        "num_monthly_txns": c.num_monthly_txns,
        "last_product_purchase_months": c.last_product_purchase_months,
        "phone": c.phone,
        "email": c.email,
    }


@tool
def get_high_value_customers(
    min_balance: float = 500000,
    min_income: float = 50000,
    limit: int = 50,
) -> str:
    """
    Retrieve customers with high account balance and income who may be
    candidates for personal loan outreach. Returns up to `limit` customers
    sorted by account balance descending.

    Args:
        min_balance: Minimum account balance filter (default ₹5 lakh)
        min_income: Minimum monthly income filter (default ₹50,000)
        limit: Maximum number of customers to return (default 50)
    """
    db: Session = SessionLocal()
    try:
        customers = (
            db.query(Customer)
            .filter(
                Customer.account_balance >= min_balance,
                Customer.monthly_income >= min_income,
            )
            .order_by(Customer.account_balance.desc())
            .limit(_clamp_limit(limit))
            .all()
        )
        result = [_customer_to_dict(c) for c in customers]
        return json.dumps({"count": len(result), "customers": result})
    finally:
        db.close()


@tool
def get_customers_without_personal_loan(
    min_income: float = 25000,
    min_credit_score: int = 650,
    limit: int = 50,
) -> str:
    """
    Retrieve customers who do NOT currently have a personal loan — the primary
    cross-sell opportunity pool. Filters by minimum income and credit score.

    Args:
        min_income: Minimum monthly income (default ₹25,000)
        min_credit_score: Minimum credit score (default 650)
        limit: Maximum customers to return (default 50)
    """
    db: Session = SessionLocal()
    try:
        customers = (
            db.query(Customer)
            .filter(
                Customer.has_personal_loan == False,
                Customer.monthly_income >= min_income,
                Customer.credit_score >= min_credit_score,
            )
            .order_by(Customer.monthly_income.desc())
            .limit(_clamp_limit(limit))
            .all()
        )
        result = [_customer_to_dict(c) for c in customers]
        return json.dumps({"count": len(result), "customers": result})
    finally:
        db.close()


@tool
def get_customers_by_city(
    city: str,
    has_salary_account: bool = False,
    limit: int = 30,
) -> str:
    """
    Retrieve customers filtered by city. Optionally filter to only salary
    account holders (stronger relationship with the bank).

    Args:
        city: City name (e.g., 'Mumbai', 'Bangalore', 'Delhi')
        has_salary_account: If True, return only salary account holders
        limit: Maximum customers to return (default 30)
    """
    db: Session = SessionLocal()
    try:
        query = db.query(Customer).filter(Customer.city == city)
        if has_salary_account:
            query = query.filter(Customer.has_salary_account == True)
        customers = query.order_by(Customer.account_balance.desc()).limit(_clamp_limit(limit)).all()
        result = [_customer_to_dict(c) for c in customers]
        return json.dumps({"count": len(result), "city": city, "customers": result})
    finally:
        db.close()


@tool
def get_customer_transactions(customer_id: str, months: int = 6) -> str:
    """
    Retrieve recent transaction history for a specific customer to understand
    their financial behaviour (spending patterns, EMI load, income regularity).

    Args:
        customer_id: The customer's ID (e.g., 'CUST0042')
        months: How many months of history to retrieve (default 6)
    """
    from datetime import date, timedelta
    db: Session = SessionLocal()
    try:
        cutoff = date.today() - timedelta(days=months * 30)
        txns = (
            db.query(Transaction)
            .filter(
                Transaction.customer_id == customer_id,
                Transaction.txn_date >= cutoff,
            )
            .order_by(Transaction.txn_date.desc())
            .all()
        )
        result = [
            {
                "amount": t.amount,
                "type": t.txn_type,
                "category": t.category,
                "date": str(t.txn_date),
                "description": t.description,
            }
            for t in txns
        ]
        total_credit = sum(t["amount"] for t in result if t["type"] == "credit")
        total_debit = sum(t["amount"] for t in result if t["type"] == "debit")
        return json.dumps({
            "customer_id": customer_id,
            "period_months": months,
            "total_transactions": len(result),
            "total_credit": total_credit,
            "total_debit": total_debit,
            "transactions": result,
        })
    finally:
        db.close()


@tool
def get_customer_by_id(customer_id: str) -> str:
    """
    Retrieve full profile of a specific customer by their ID.

    Args:
        customer_id: The customer's ID (e.g., 'CUST0001')
    """
    db: Session = SessionLocal()
    try:
        c = db.query(Customer).filter(Customer.id == customer_id).first()
        if not c:
            return json.dumps({"error": f"Customer {customer_id} not found"})
        return json.dumps(_customer_to_dict(c))
    finally:
        db.close()


@tool
def get_customers_for_loan(
    loan_category: str,
    min_income: float = 25000,
    min_credit_score: int = 620,
    limit: int = 50,
) -> str:
    """
    Retrieve customers who are eligible targets for a specific non-personal loan category.
    Applies category-appropriate income/credit filters and excludes existing holders.
    Use this tool when the RM asks for home loan, car loan, business loan, education loan,
    gold loan, or loan against property (LAP) customers.

    Args:
        loan_category: One of 'home_loan', 'auto_loan', 'business_loan', 'education_loan', 'gold_loan', 'lap'
        min_income: Minimum monthly income filter (category minimums apply if higher)
        min_credit_score: Minimum credit score filter (category minimums apply if higher)
        limit: Maximum customers to return (default 50)
    """
    db: Session = SessionLocal()
    BUSINESS_OCCS = ["Business Owner", "Consultant", "Architect", "Lawyer"]
    try:
        query = db.query(Customer)
        if loan_category == "home_loan":
            query = query.filter(
                Customer.has_home_loan == False,
                Customer.monthly_income >= max(min_income, 35000),
                Customer.credit_score >= max(min_credit_score, 660),
            )
        elif loan_category == "auto_loan":
            query = query.filter(
                Customer.has_auto_loan == False,
                Customer.monthly_income >= max(min_income, 25000),
                Customer.credit_score >= max(min_credit_score, 640),
            )
        elif loan_category == "business_loan":
            query = query.filter(
                Customer.has_business_loan == False,
                Customer.monthly_income >= max(min_income, 40000),
                Customer.credit_score >= max(min_credit_score, 640),
                Customer.occupation.in_(BUSINESS_OCCS),
            )
        elif loan_category == "education_loan":
            query = query.filter(
                Customer.has_education_loan == False,
                Customer.monthly_income >= max(min_income, 20000),
                Customer.credit_score >= max(min_credit_score, 600),
            )
        elif loan_category == "gold_loan":
            query = query.filter(
                Customer.has_gold_loan == False,
                Customer.monthly_income >= max(min_income, 15000),
                Customer.credit_score >= max(min_credit_score, 550),
            )
        elif loan_category == "lap":
            query = query.filter(
                Customer.monthly_income >= max(min_income, 50000),
                Customer.credit_score >= max(min_credit_score, 680),
            )
        else:
            query = query.filter(
                Customer.monthly_income >= min_income,
                Customer.credit_score >= min_credit_score,
            )
        customers = query.order_by(Customer.account_balance.desc()).limit(_clamp_limit(limit)).all()
        result = [_customer_to_dict(c) for c in customers]
        return json.dumps({"count": len(result), "loan_category": loan_category, "customers": result})
    finally:
        db.close()


@tool
def get_salary_account_holders_without_loan(city: str = "", limit: int = 30) -> str:
    """
    Retrieve customers who have a salary account with the bank but no personal
    loan — ideal candidates for pre-approved loan offers. Optionally filter by city.

    Args:
        city: City filter (leave empty for all cities)
        limit: Maximum customers to return (default 30)
    """
    db: Session = SessionLocal()
    try:
        query = db.query(Customer).filter(
            Customer.has_salary_account == True,
            Customer.has_personal_loan == False,
            Customer.credit_score >= 680,
        )
        if city:
            query = query.filter(Customer.city == city)
        customers = query.order_by(Customer.account_balance.desc()).limit(_clamp_limit(limit)).all()
        result = [_customer_to_dict(c) for c in customers]
        return json.dumps({"count": len(result), "customers": result})
    finally:
        db.close()
