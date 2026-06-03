"""FastAPI backend — main entrypoint. Agent logic lives in services/."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ChatRequest, ChatResponse
from app.services.conversation_service import handle_chat
from app.agents.supervisor import init_crm_graph
from app.db.database import init_db, SessionLocal
from app.db.schema import Customer, Product


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await init_crm_graph()
    yield


app = FastAPI(
    title="Banking CRM Agentic AI",
    description="Agentic AI for Relationship Managers — customer targeting and outreach",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational endpoint with full state persistence per thread_id.
    Returns typed recommendations, agent trace, and compliance stats.
    """
    try:
        return await handle_chat(request.message, request.thread_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")


@app.get("/customers/summary")
def customer_summary():
    db = SessionLocal()
    try:
        total = db.query(Customer).count()
        with_loan = db.query(Customer).filter(Customer.has_personal_loan == True).count()
        salary_holders = db.query(Customer).filter(Customer.has_salary_account == True).count()
        high_value = db.query(Customer).filter(Customer.account_balance >= 500000).count()
        return {
            "total_customers": total,
            "with_personal_loan": with_loan,
            "salary_account_holders": salary_holders,
            "high_value_customers": high_value,
            "cross_sell_pool": total - with_loan,
        }
    finally:
        db.close()


@app.get("/products")
def list_products():
    db = SessionLocal()
    try:
        return [
            {
                "id": p.id, "name": p.name, "interest_rate": p.interest_rate,
                "max_amount": p.max_amount, "description": p.description,
            }
            for p in db.query(Product).all()
        ]
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok", "service": "Banking CRM Agentic AI"}
