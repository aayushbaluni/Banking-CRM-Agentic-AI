from pydantic import BaseModel
from typing import Optional


class CustomerProfile(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    occupation: str
    education: str
    city: str
    state: str
    region: str
    monthly_income: float
    account_balance: float
    credit_score: int
    num_products: int
    has_personal_loan: bool
    has_home_loan: bool
    has_salary_account: bool
    has_fixed_deposit: bool
    months_as_customer: int
    avg_monthly_txn_amount: float
    num_monthly_txns: int
    last_product_purchase_months: int
    phone: str
    email: str


class ScoredCustomer(BaseModel):
    customer: CustomerProfile
    propensity_score: float
    tier: str  # High / Medium / Low
    score_reason: str
    recommended_product_id: Optional[str] = None
    recommended_product_name: Optional[str] = None
    whatsapp_message: Optional[str] = None
