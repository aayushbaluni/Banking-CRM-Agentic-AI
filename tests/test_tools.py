"""Unit tests for tool functions — each tool tested against the real SQLite DB."""
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import init_db
from app.db.seed import seed
from app.tools.crm_tools import (
    get_high_value_customers,
    get_customers_without_personal_loan,
    get_customers_by_city,
    get_salary_account_holders_without_loan,
)
from app.tools.scoring_tools import score_loan_propensity, batch_score_customers
from app.tools.product_tools import recommend_product, list_available_products


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Ensure DB is seeded before running tests."""
    init_db()
    seed()


def test_get_high_value_customers():
    result = json.loads(get_high_value_customers.invoke({"min_balance": 500000, "limit": 10}))
    assert "customers" in result
    assert result["count"] <= 10
    for c in result["customers"]:
        assert c["account_balance"] >= 500000


def test_get_customers_without_loan():
    result = json.loads(get_customers_without_personal_loan.invoke({"limit": 20}))
    assert "customers" in result
    for c in result["customers"]:
        assert c["has_personal_loan"] == False


def test_get_customers_by_city():
    result = json.loads(get_customers_by_city.invoke({"city": "Mumbai", "limit": 10}))
    assert "customers" in result
    for c in result["customers"]:
        assert c["city"] == "Mumbai"


def test_salary_holders_without_loan():
    result = json.loads(get_salary_account_holders_without_loan.invoke({}))
    for c in result["customers"]:
        assert c["has_salary_account"] == True
        assert c["has_personal_loan"] == False


def test_score_loan_propensity():
    customer = {
        "id": "TEST001",
        "name": "Test User",
        "monthly_income": 80000,
        "account_balance": 500000,
        "credit_score": 750,
        "age": 32,
        "num_products": 2,
        "has_salary_account": True,
        "has_fixed_deposit": False,
        "months_as_customer": 36,
        "avg_monthly_txn_amount": 50000,
        "num_monthly_txns": 15,
        "last_product_purchase_months": 6,
        "education": "Graduate",
        "has_personal_loan": False,
    }
    result = json.loads(score_loan_propensity.invoke({"customer_json": json.dumps(customer)}))
    assert "propensity_score" in result
    assert 0.0 <= result["propensity_score"] <= 1.0
    assert result["tier"] in ["High", "Medium", "Low"]
    assert "score_reason" in result


def test_batch_score_returns_top_n():
    data = json.loads(get_customers_without_personal_loan.invoke({"limit": 30}))
    customers = data["customers"]
    result = json.loads(batch_score_customers.invoke({
        "customers_json": json.dumps(customers),
        "top_n": 5,
    }))
    assert len(result["results"]) <= 5
    scores = [r["propensity_score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)


def test_recommend_product_eligible():
    scored = {
        "customer": {
            "id": "TEST001",
            "name": "Test User",
            "monthly_income": 120000,
            "credit_score": 780,
            "has_salary_account": True,
            "age": 35,
            "has_personal_loan": False,
        },
        "propensity_score": 0.8,
    }
    result = json.loads(recommend_product.invoke({"scored_customer_json": json.dumps(scored)}))
    assert result["eligible"] == True
    assert "product" in result
    assert result["product"]["id"] in ["PL001", "PL002", "PL003", "PL004"]


def test_list_products():
    result = json.loads(list_available_products.invoke({}))
    products = result["products"]
    assert len(products) >= 13
    assert any(p["id"].startswith("PL") for p in products)
    assert any(p["id"].startswith("HL") for p in products)
