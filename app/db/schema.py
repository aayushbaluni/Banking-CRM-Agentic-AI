from sqlalchemy import Column, String, Integer, Float, Boolean, Date, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    occupation = Column(String)
    education = Column(String)
    city = Column(String)
    state = Column(String)
    monthly_income = Column(Float)
    account_balance = Column(Float)
    credit_score = Column(Integer)
    num_products = Column(Integer, default=0)
    has_personal_loan = Column(Boolean, default=False)
    has_home_loan = Column(Boolean, default=False)
    has_auto_loan = Column(Boolean, default=False)
    has_business_loan = Column(Boolean, default=False)
    has_education_loan = Column(Boolean, default=False)
    has_gold_loan = Column(Boolean, default=False)
    has_salary_account = Column(Boolean, default=False)
    has_fixed_deposit = Column(Boolean, default=False)
    months_as_customer = Column(Integer)
    avg_monthly_txn_amount = Column(Float)
    num_monthly_txns = Column(Integer)
    last_product_purchase_months = Column(Integer)
    region = Column(String)  # north/south/east/west
    phone = Column(String)
    email = Column(String)

    transactions = relationship("Transaction", back_populates="customer")
    loan_applications = relationship("LoanApplication", back_populates="customer")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    amount = Column(Float)
    txn_type = Column(String)  # credit/debit
    category = Column(String)  # salary/emi/shopping/utilities/investment
    txn_date = Column(Date)
    description = Column(Text)

    customer = relationship("Customer", back_populates="transactions")


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    loan_type = Column(String)
    amount_requested = Column(Float)
    status = Column(String)  # approved/rejected/pending
    applied_date = Column(Date)

    customer = relationship("Customer", back_populates="loan_applications")


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)
    name = Column(String)
    category = Column(String)
    interest_rate = Column(Float)
    min_income = Column(Float)
    max_amount = Column(Float)
    min_credit_score = Column(Integer)
    description = Column(Text)
    eligibility_criteria = Column(Text)
