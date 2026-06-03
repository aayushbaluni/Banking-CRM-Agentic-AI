"""Generate 600 realistic Indian banking customer profiles and seed the SQLite DB."""
import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.db.database import init_db, SessionLocal
from app.db.schema import Customer, Transaction, LoanApplication, Product

fake = Faker("en_IN")
random.seed(42)

CITIES = {
    "Mumbai": ("Maharashtra", "west"),
    "Delhi": ("Delhi", "north"),
    "Bangalore": ("Karnataka", "south"),
    "Chennai": ("Tamil Nadu", "south"),
    "Hyderabad": ("Telangana", "south"),
    "Pune": ("Maharashtra", "west"),
    "Kolkata": ("West Bengal", "east"),
    "Ahmedabad": ("Gujarat", "west"),
    "Jaipur": ("Rajasthan", "north"),
    "Lucknow": ("Uttar Pradesh", "north"),
}

OCCUPATIONS = [
    "Software Engineer", "Doctor", "Business Owner", "Teacher",
    "Sales Manager", "Accountant", "Marketing Executive", "Engineer",
    "Consultant", "Banker", "Lawyer", "Architect", "Pharmacist",
    "Government Employee", "Retail Manager",
]

EDUCATION = ["Graduate", "Post Graduate", "Professional", "Undergraduate", "Doctorate"]

TXN_CATEGORIES = ["salary", "emi", "shopping", "utilities", "investment", "transfer", "insurance"]

LOAN_APPLICATION_TYPES = [
    "personal_loan", "home_loan", "auto_loan",
    "business_loan", "education_loan", "gold_loan", "lap",
]

BUSINESS_OCCUPATIONS = {"Business Owner", "Consultant", "Architect", "Lawyer"}


def _income_for_occupation(occ: str) -> float:
    ranges = {
        "Software Engineer": (60000, 200000),
        "Doctor": (100000, 350000),
        "Business Owner": (80000, 500000),
        "Teacher": (25000, 60000),
        "Sales Manager": (50000, 150000),
        "Accountant": (35000, 90000),
        "Marketing Executive": (40000, 120000),
        "Engineer": (50000, 180000),
        "Consultant": (80000, 250000),
        "Banker": (45000, 150000),
        "Lawyer": (70000, 300000),
        "Architect": (60000, 200000),
        "Pharmacist": (40000, 100000),
        "Government Employee": (30000, 80000),
        "Retail Manager": (30000, 70000),
    }
    lo, hi = ranges.get(occ, (30000, 100000))
    return round(random.uniform(lo, hi), 2)


def _generate_customer(idx: int) -> Customer:
    city = random.choice(list(CITIES.keys()))
    state, region = CITIES[city]
    occ = random.choice(OCCUPATIONS)
    income = _income_for_occupation(occ)
    age = random.randint(22, 62)
    months = random.randint(3, 120)
    balance = round(income * random.uniform(0.5, 30), 2)
    credit_score = random.randint(550, 850)
    num_products = random.randint(0, 5)
    avg_txn = round(income * random.uniform(0.3, 0.9), 2)

    has_personal = random.random() < 0.35
    has_home = random.random() < 0.20
    has_auto = random.random() < 0.15
    has_business = (
        random.random() < 0.25 if occ in BUSINESS_OCCUPATIONS
        else random.random() < 0.04
    )
    has_education = (
        random.random() < 0.08 if age < 35
        else random.random() < 0.02
    )
    has_gold = random.random() < 0.08
    has_salary = random.random() < 0.60
    has_fd = random.random() < 0.25

    return Customer(
        id=f"CUST{idx:04d}",
        name=fake.name(),
        age=age,
        gender=random.choice(["Male", "Female"]),
        occupation=occ,
        education=random.choice(EDUCATION),
        city=city,
        state=state,
        region=region,
        monthly_income=income,
        account_balance=balance,
        credit_score=credit_score,
        num_products=num_products,
        has_personal_loan=has_personal,
        has_home_loan=has_home,
        has_auto_loan=has_auto,
        has_business_loan=has_business,
        has_education_loan=has_education,
        has_gold_loan=has_gold,
        has_salary_account=has_salary,
        has_fixed_deposit=has_fd,
        months_as_customer=months,
        avg_monthly_txn_amount=avg_txn,
        num_monthly_txns=random.randint(5, 40),
        last_product_purchase_months=random.randint(0, 36),
        phone=fake.phone_number(),
        email=fake.email(),
    )


def _generate_transactions(customer: Customer, n: int = 12) -> list[Transaction]:
    txns = []
    for _ in range(n):
        days_ago = random.randint(0, 180)
        category = random.choice(TXN_CATEGORIES)
        if category == "salary":
            amount = customer.monthly_income
            txn_type = "credit"
        elif category == "emi":
            amount = round(customer.monthly_income * random.uniform(0.05, 0.3), 2)
            txn_type = "debit"
        else:
            amount = round(random.uniform(500, customer.monthly_income * 0.5), 2)
            txn_type = random.choice(["debit", "credit"])

        txns.append(Transaction(
            customer_id=customer.id,
            amount=amount,
            txn_type=txn_type,
            category=category,
            txn_date=date.today() - timedelta(days=days_ago),
            description=f"{category.title()} transaction",
        ))
    return txns


PRODUCTS = [
    # ── Personal Loans ─────────────────────────────────────────────────────────
    Product(
        id="PL001", name="Pre-Approved Personal Loan", category="personal_loan",
        interest_rate=10.5, min_income=30000, max_amount=500000, min_credit_score=700,
        description="Instant pre-approved loan for existing salary account holders.",
        eligibility_criteria="Salary account with min 6 months tenure, credit score ≥ 700",
    ),
    Product(
        id="PL002", name="Standard Personal Loan", category="personal_loan",
        interest_rate=12.5, min_income=25000, max_amount=300000, min_credit_score=650,
        description="Standard personal loan with flexible repayment.",
        eligibility_criteria="Monthly income ≥ ₹25,000, credit score ≥ 650",
    ),
    Product(
        id="PL003", name="Premium Personal Loan", category="personal_loan",
        interest_rate=9.5, min_income=100000, max_amount=2000000, min_credit_score=750,
        description="Low-rate loan for high-income professionals.",
        eligibility_criteria="Monthly income ≥ ₹1,00,000, credit score ≥ 750",
    ),
    Product(
        id="PL004", name="Flexi Personal Loan", category="personal_loan",
        interest_rate=11.5, min_income=40000, max_amount=800000, min_credit_score=680,
        description="Revolving credit line with flexible withdrawal.",
        eligibility_criteria="Age 24–45, income ≥ ₹40,000, credit score ≥ 680",
    ),
    # ── Home Loans ─────────────────────────────────────────────────────────────
    Product(
        id="HL001", name="Prime Home Loan", category="home_loan",
        interest_rate=8.5, min_income=50000, max_amount=10000000, min_credit_score=700,
        description="Low-rate home loan for salaried professionals.",
        eligibility_criteria="Income ≥ ₹50,000/mo, credit score ≥ 700, age 24–55",
    ),
    Product(
        id="HL002", name="Affordable Home Loan", category="home_loan",
        interest_rate=9.2, min_income=35000, max_amount=5000000, min_credit_score=660,
        description="Home loan for first-time buyers with moderate income.",
        eligibility_criteria="Income ≥ ₹35,000/mo, credit score ≥ 660",
    ),
    # ── Car / Auto Loans ───────────────────────────────────────────────────────
    Product(
        id="AL001", name="New Car Loan", category="auto_loan",
        interest_rate=9.0, min_income=35000, max_amount=1500000, min_credit_score=680,
        description="Finance your new car at competitive rates with quick disbursal.",
        eligibility_criteria="Income ≥ ₹35,000/mo, credit score ≥ 680, age 21–60",
    ),
    Product(
        id="AL002", name="Used Car Loan", category="auto_loan",
        interest_rate=11.5, min_income=25000, max_amount=500000, min_credit_score=640,
        description="Pre-owned car financing with fast approval.",
        eligibility_criteria="Income ≥ ₹25,000/mo, credit score ≥ 640",
    ),
    # ── Business Loans ─────────────────────────────────────────────────────────
    Product(
        id="BL001", name="SME Business Loan", category="business_loan",
        interest_rate=11.5, min_income=80000, max_amount=5000000, min_credit_score=700,
        description="Collateral-free loan for SMEs and established entrepreneurs.",
        eligibility_criteria="Business income ≥ ₹80,000/mo, credit score ≥ 700, 2+ yrs in business",
    ),
    Product(
        id="BL002", name="Micro Business Loan", category="business_loan",
        interest_rate=14.0, min_income=40000, max_amount=1000000, min_credit_score=640,
        description="Quick funds for micro and small businesses.",
        eligibility_criteria="Income ≥ ₹40,000/mo, credit score ≥ 640",
    ),
    # ── Education Loan ─────────────────────────────────────────────────────────
    Product(
        id="EDL001", name="Education Loan", category="education_loan",
        interest_rate=9.5, min_income=20000, max_amount=2000000, min_credit_score=600,
        description="Fund higher education in India or abroad with moratorium period.",
        eligibility_criteria="Co-applicant income ≥ ₹20,000/mo, credit score ≥ 600, student age ≤ 30",
    ),
    # ── Gold Loan ──────────────────────────────────────────────────────────────
    Product(
        id="GL001", name="Gold Loan", category="gold_loan",
        interest_rate=10.5, min_income=15000, max_amount=500000, min_credit_score=550,
        description="Instant loan against gold jewellery — disbursal in 30 minutes.",
        eligibility_criteria="Gold assets ≥ ₹50,000, credit score ≥ 550, income ≥ ₹15,000/mo",
    ),
    # ── Loan Against Property ──────────────────────────────────────────────────
    Product(
        id="LAP001", name="Loan Against Property", category="lap",
        interest_rate=9.0, min_income=50000, max_amount=20000000, min_credit_score=680,
        description="Unlock the value of your property for business or personal needs.",
        eligibility_criteria="Residential/commercial property owned, income ≥ ₹50,000/mo, credit score ≥ 680",
    ),
]


def seed():
    from app.config import settings  # local import to allow sys.path patch above

    db_path = settings.db_path

    # Auto-upgrade: if DB has old schema (< 13 products), delete and rebuild
    if os.path.exists(db_path):
        try:
            _check = create_engine(f"sqlite:///{db_path}")
            with _check.connect() as conn:
                n_products = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
                n_customers = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
            _check.dispose()
            if n_customers > 0 and n_products >= 13:
                print("Database already up to date — skipping seed.")
                return
            if n_customers > 0:
                print(f"Stale database ({n_products} products, {n_customers} customers) — rebuilding...")
                os.remove(db_path)
        except Exception:
            pass  # table may not exist yet — let init_db handle it

    init_db()
    db: Session = SessionLocal()
    try:
        print(f"Seeding {len(PRODUCTS)} products...")
        for p in PRODUCTS:
            db.merge(p)

        print("Seeding 600 customers and transactions...")
        for i in range(1, 601):
            cust = _generate_customer(i)
            db.add(cust)
            db.flush()
            for txn in _generate_transactions(cust):
                db.add(txn)

            if random.random() < 0.25:
                loan_type = random.choice(LOAN_APPLICATION_TYPES)
                multipliers = {
                    "personal_loan": (5, 20),
                    "home_loan": (40, 100),
                    "auto_loan": (8, 15),
                    "business_loan": (20, 50),
                    "education_loan": (10, 30),
                    "gold_loan": (2, 8),
                    "lap": (60, 120),
                }
                lo, hi = multipliers.get(loan_type, (5, 20))
                db.add(LoanApplication(
                    customer_id=cust.id,
                    loan_type=loan_type,
                    amount_requested=round(cust.monthly_income * random.uniform(lo, hi), 2),
                    status=random.choice(["approved", "rejected", "pending"]),
                    applied_date=date.today() - timedelta(days=random.randint(30, 365)),
                ))

        db.commit()
        print(f"Seeding complete: 600 customers, {len(PRODUCTS)} products inserted.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
