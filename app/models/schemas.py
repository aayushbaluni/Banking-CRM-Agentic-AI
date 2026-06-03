"""
Typed Pydantic schemas for all API request/response contracts.
Replaces loose list[dict] throughout the codebase.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, field_validator


# ── Inbound ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        if len(v) > 10_000:
            raise ValueError("message exceeds maximum length of 10000 characters")
        return v


# ── Domain models ─────────────────────────────────────────────────────────────

class CustomerProfile(BaseModel):
    id: str
    name: str
    age: int
    occupation: str
    city: str
    region: str
    monthly_income: float
    account_balance: float
    credit_score: int
    has_personal_loan: bool
    has_salary_account: bool
    phone: str


class ProductMatch(BaseModel):
    id: str
    name: str
    interest_rate: float
    max_amount: float
    description: str


class CustomerRecommendation(BaseModel):
    customer: CustomerProfile
    propensity_score: float
    tier: Literal["High", "Medium", "Low"]
    score_reason: str
    product: ProductMatch
    recommended_loan_amount: float
    whatsapp_message: Optional[str] = None
    message_char_count: Optional[int] = None
    message_compliant: Optional[bool] = None

    @field_validator("propensity_score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("propensity_score must be between 0 and 1")
        return round(v, 3)


class TraceEntry(BaseModel):
    step: str
    decision: str
    tools_called: list[str] = []
    result_summary: str = ""
    skipped: bool = False
    duration_ms: int = 0


# ── Outbound ──────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    thread_id: str
    response: str
    recommendations: list[CustomerRecommendation] = []
    trace: list[TraceEntry] = []
    stats: dict = {}
    total_duration_ms: int = 0
