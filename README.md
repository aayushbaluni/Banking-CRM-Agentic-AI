---
title: Banking CRM AI
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Banking CRM Agentic AI

A **production-hardened, Supervisor–Subagent multi-agent system** built with LangGraph that lets Relationship Managers (RMs) use natural language to identify high-potential customers across **seven loan categories**, score conversion propensity, match compliant products, and generate personalized WhatsApp outreach — with **LLM-driven routing**, **durable conversation state**, and **180 automated tests** validating security and edge cases.

---

## Highlights

| Capability | Implementation |
|------------|----------------|
| Intent routing | Two-tier: O(1) fast-path for greetings, **Azure GPT-4o** for everything else |
| Product scope | LLM classifies **7 loan categories** + unsupported products (not keyword-based) |
| Follow-up decisions | **RESTART / CONTINUE / OUTREACH / UNSUPPORTED** — purely LLM-driven |
| Conversation memory | **AsyncSqliteSaver** — state survives server restarts |
| Product catalogue | **7 loan types**, **13 product variants** (personal, home, auto, business, education, gold, LAP) |
| Compliance | WhatsApp guardrails module validates message length, tone, and prohibited content |
| Quality assurance | **155 offline pytest** + **25 live production** tests (abuse, injection, concurrency) |
| Hardening audit | **30 production bugs fixed** — crash prevention, security, performance, state safety |

---

## Architecture

```mermaid
flowchart TD
    RM[🧑 Relationship Manager\nStreamlit Chat UI] --> API[FastAPI /chat\nlifespan + thread_id]
    API --> CKPT[(AsyncSqliteSaver\nDurable Checkpoint)]
    API --> ROUTER[Router Node\nTwo-Tier Intent Classification]

    ROUTER -->|O(1) fast-path| GREET[general_chat\nObvious greetings]
    ROUTER -->|LLM classify| INTENT{Intent?\nrouting_llm.py}

    INTENT -->|social / chitchat| GREET
    INTENT -->|CRM pipeline| SUP[Supervisor StateGraph\nsupervisor.py]

    SUP -->|LLM product scope| SCOPE[7 Loan Categories\n+ Unsupported]
    SCOPE -->|personal · home · auto · business\neducation · gold · LAP| DA[Data Agent\ncrm_tools selection]
    SUP -->|batch scoring| SA[Scoring Agent\nXGBoost + heuristic]
    SUP -->|rule engine| PA[Product Agent\n13 variants]
    SUP -->|parallel async| OA[Outreach Agent\nGPT-4o messages]

    DA --> CRM[(SQLite CRM\nWAL + busy_timeout\n600 customers)]
    SA --> ML[(model.pkl\nclamped 0–1 scores)]
    PA --> PROD[Rule Engine\nauditable recommendations]
    OA --> GR[guardrails.py\nWhatsApp compliance]
    GR -->|pass / fail| SUP

    SUP -->|LLM follow-up action| FU[RESTART · CONTINUE\nOUTREACH · UNSUPPORTED]
    FU --> CKPT
    SUP --> API
    API --> RM

    subgraph Modules["Shared Infrastructure"]
        BASE[base.py\nLLM factory · timed_node · tool executor]
        PROMPTS[registry.py\nCentralized prompts]
    end

    ROUTER -.-> BASE
    SUP -.-> BASE
    SUP -.-> PROMPTS
```

### LLM-Driven Routing Layer

All classification decisions that affect graph traversal live in `app/agents/routing_llm.py` — not brittle keyword matchers.

| Decision | Strategy | Outcome |
|----------|----------|---------|
| **Intent** | Fast-path O(1) for obvious greetings; LLM for all other utterances | `general_chat` vs CRM pipeline |
| **Product scope** | LLM maps natural language → loan category | 7 categories + `unsupported` |
| **Follow-up action** | LLM reads conversation + checkpoint state | `RESTART`, `CONTINUE`, `OUTREACH`, or `UNSUPPORTED` |

The supervisor (`supervisor.py`) orchestrates subagents; shared LLM clients, tool-call execution, and node timing come from `app/agents/base.py`.

### Loan Categories & Product Variants

| Category | Example products | Code prefix |
|----------|------------------|-------------|
| Personal | Standard, Pre-Approved, Top-Up | PL |
| Home | New Purchase, Balance Transfer, Construction | HL |
| Auto | New Car, Used Car, Two-Wheeler | AL |
| Business | Working Capital, Term Loan, MSME | BL |
| Education | Domestic, Study Abroad | EL |
| Gold | Gold Loan | GL |
| LAP | Loan Against Property | LAP |

---

## Execution Flow

```
1. RM: "Find Mumbai salary-account holders for a home loan balance transfer"
   │
2. FastAPI receives request → thread_id → AsyncSqliteSaver restores prior state (if any)
   │
3. Router Node:
   ├─ Fast-path? → skip LLM for "hi", "hello", etc.
   └─ Else → routing_llm.classify_intent() → CRM pipeline
   │
4. routing_llm.classify_product_scope() → "home" (not keyword regex)
   │
5. Data Agent: LLM selects get_salary_account_holders_without_loan(city='Mumbai', ...)
   → SQLAlchemy (WAL mode, query limit clamped ≤ 100) → customer records
   │
6. Scoring Agent: batch_score_customers()
   → XGBoost (cached model) or heuristic fallback
   → _safe_num/_safe_bool coercion, scores clamped [0, 1]
   │
7. Product Agent: recommend_product() per customer
   → Rule engine → HL002 Balance Transfer (or nearest eligible variant)
   │
8. [If outreach requested] Outreach Agent: async batch_generate_messages()
   → guardrails.py validates each message (<300 chars, compliance rules)
   │
9. Supervisor synthesizes response → FastAPI returns structured JSON + trace
   │
10. Streamlit renders ranked cards, product match, messages, export

    [Follow-up]: "Send WhatsApp to the top 3 from that list"
    → routing_llm.classify_followup_action() → CONTINUE or OUTREACH
    → Checkpoint read (no in-place mutation) → skips re-query if state intact
    → Outreach only, using existing final_recommendations
```

---

## New Modules (Hardening Audit)

| Module | Responsibility |
|--------|----------------|
| `app/agents/routing_llm.py` | Intent, product scope, and follow-up action — all LLM classification |
| `app/agents/base.py` | Shared LLM factory, singleton client, `@timed_node`, tool-call executor |
| `app/services/guardrails.py` | WhatsApp compliance validation before messages reach the RM |
| `app/prompts/registry.py` | Centralized prompt templates — single source of truth for LLM prompts |

Subagents are split for clarity and testability: `data_agent.py`, `scoring_agent.py`, `product_agent.py`, `outreach_agent.py`.

---

## Production Hardening (30 Bugs Fixed)

### Crash Prevention
- All `json.loads` wrapped with safe fallbacks
- None-safe LLM content extraction before parsing
- Null product guards in recommendation pipeline
- NaN feature protection in ML inference path

### Security
- Generic API error messages — no `str(e)` leakage to clients
- SQLite **WAL mode** + `busy_timeout` for concurrent access safety
- CRM query limits clamped (maximum **100** rows per tool call)

### Performance
- XGBoost model loaded **once** and cached in process memory
- LLM client **singleton** — no redundant client construction per request
- Eliminated double LLM calls on routing hot paths

### State Safety
- No in-place LangGraph checkpoint mutation
- Trace reducer handles `None` entries gracefully
- Async outreach with synchronous fallback on event-loop edge cases

### Scoring Integrity
- `_safe_num` / `_safe_bool` coercion for malformed CRM fields
- ML scores clamped to **[0, 1]**
- Correct `scoring_method` label (`xgboost` vs `heuristic`) in API responses

---

## Tool Design

| Tool | Layer | Type | What it does |
|------|-------|------|-------------|
| `get_high_value_customers` | CRM | SQLAlchemy | Balance ≥ threshold, sorted by balance |
| `get_customers_without_personal_loan` | CRM | SQLAlchemy | Cross-sell pool — no existing personal loan |
| `get_customers_by_city` | CRM | SQLAlchemy | City filter + optional salary account flag |
| `get_salary_account_holders_without_loan` | CRM | SQLAlchemy | Pre-approved loan candidates by city |
| `get_customer_transactions` | CRM | SQLAlchemy | Spending patterns for a specific customer |
| `get_customer_by_id` | CRM | SQLAlchemy | Single customer profile lookup |
| `get_customers_for_loan` | CRM | SQLAlchemy | Category-aware cohort retrieval (7 loan types) |
| `score_loan_propensity` | ML | XGBoost | Single-customer score (0–1) + tier |
| `batch_score_customers` | ML | XGBoost | Batch score, return top N (clamped) |
| `recommend_product` | Rules | Deterministic | Segment → one of 13 loan variants |
| `list_available_products` | DB | SQLAlchemy | Product catalogue lookup |
| `generate_whatsapp_message` | LLM | GPT-4o | Single personalized message |
| `batch_generate_messages` | LLM | GPT-4o | Bulk generation + guardrail pass |

All CRM tools use `@tool` decorators with typed inputs, descriptive docstrings for LLM tool selection, and JSON string returns.

---

## Demo Use Cases

### Use Case 1 — Personal Loan (Primary Flow)
> *"Find high-value customers likely to convert for a personal loan this month"*

- Router → CRM pipeline; product scope → **personal**
- Data Agent: `get_high_value_customers` + cross-sell queries
- Scoring → top 10 by propensity; Product Agent → PL001/PL002/PL003
- Response: Ranked table with scores, tiers, product match, reason codes

### Use Case 2 — Home Loan Balance Transfer
> *"Which customers in Bangalore with existing home loans are good candidates for balance transfer?"*

- Product scope → **home**; Data Agent filters by city and loan history
- Scoring ranks by propensity; Product Agent → **HL002** Balance Transfer
- Guardrails-ready outreach on request

### Use Case 3 — Auto Loan (New Car)
> *"Show me salaried customers in Pune earning over ₹8L who might need a new car loan"*

- Product scope → **auto**; CRM tools segment by city and income band
- Product Agent → **AL001** New Car Loan where eligibility rules match

### Use Case 4 — Business / MSME
> *"Identify business account holders without a working capital loan"*

- Product scope → **business**; rule engine maps to **BL001** Working Capital or **BL003** MSME

### Use Case 5 — Education (Study Abroad)
> *"Target young professionals in Delhi for education loans — study abroad segment"*

- Product scope → **education**; recommends **EL002** Study Abroad where credit and age criteria fit

### Use Case 6 — Gold Loan
> *"High-balance customers who might benefit from a gold loan against holdings"*

- Product scope → **gold**; **GL001** Gold Loan for eligible high-balance segments

### Use Case 7 — LAP (Loan Against Property)
> *"Property owners in Mumbai with strong credit — LAP candidates"*

- Product scope → **LAP**; **LAP001** Loan Against Property for qualifying profiles

### Use Case 8 — Stateful Follow-up (Durable Memory)
> *"Generate personalized WhatsApp messages for the top 5 from that list"*

- Same `thread_id` → **AsyncSqliteSaver** restores state after restart
- `routing_llm` → **OUTREACH** or **CONTINUE** — skips data/scoring
- Outreach Agent + **guardrails.py** → compliant messages with name, city, product CTA

### Use Case 9 — Category Switch (RESTART)
> *"Actually, forget personal — show me auto loan prospects in Chennai instead"*

- Follow-up classifier → **RESTART**; fresh CRM query and scoring for **auto**

---

## Testing

### Offline Suite — 155 pytest tests

```bash
pytest tests/ -v
```

| Test file | Focus |
|-----------|-------|
| `tests/test_tools.py` | CRM, scoring, product, message tools |
| `tests/test_agents.py` | Supervisor graph, subagent integration |
| `tests/test_security.py` | Error leakage, query limit abuse, injection |
| `tests/test_brutal.py` | Malformed JSON, None payloads, NaN features |
| `tests/test_guardrails.py` | WhatsApp compliance edge cases |
| `tests/test_prompts.py` | Prompt registry completeness |

**Coverage themes:** CRM tool abuse · scoring abuse · guardrail edge cases · API abuse · intent guard · smart router · data agent · multi-category routing · concurrency assumptions

### Live Production Suite — 25 E2E tests

```bash
python scripts/brutal_production_test.py
```

Requires a running backend with valid Azure OpenAI credentials. Exercises:

- Vague and ambiguous natural-language queries
- Prompt-injection and tool-abuse attempts
- Multi-turn conversations with **CONTINUE** / **OUTREACH** / **RESTART**
- Concurrent request stress
- Mid-conversation **category switching** (personal → home → auto)

---

## Key Design Decisions

### 1. LLM-Driven Router (vs. keyword routing)
**Decision:** Two-tier intent (fast-path + LLM), LLM product scope, LLM follow-up actions in `routing_llm.py`.  
**Why:** Banking RMs phrase requests inconsistently; keyword routers misroute "balance transfer" vs "home loan" vs chitchat. LLM classification generalizes without maintaining regex forests.  
**Trade-off:** Adds latency and token cost on non-greeting turns; mitigated by singleton client and eliminating duplicate LLM calls.

### 2. Supervisor–Subagent Pattern
**Decision:** Linear supervisor routing (data → score → product → outreach) with dedicated subagent modules.  
**Why:** Each stage depends on prior output; compliance requires eligibility before recommendation; modules are independently testable.  
**Trade-off:** Sequential latency (~15–30s for 10 customers). Outreach uses async batching where possible.

### 3. Heuristic + ML Hybrid Scoring
**Decision:** XGBoost with SMOTE + deterministic heuristic fallback.  
**Why:** Explainability for auditors; system never fails if `model.pkl` is missing. `_safe_num` / score clamping prevent garbage-in crashes.  
**Trade-off:** Model trained on synthetic data — production needs retraining pipelines.

### 4. AsyncSqliteSaver (vs. MemorySaver)
**Decision:** LangGraph `AsyncSqliteSaver` keyed by `thread_id`.  
**Why:** RMs resume conversations after deploys and server restarts; in-memory checkpoints were a demo-only liability.  
**Trade-off:** Single-node SQLite; production would use PostgresSaver + connection pooling.

### 5. Deterministic Product Rule Engine
**Decision:** Rule engine over LLM product recommendation.  
**Why:** Interest rates and eligibility must be auditable; LLM scope classification only picks the *category*, rules pick the *variant*.  
**Trade-off:** Rules require updates when product sheets change.

### 6. Guardrails Before Delivery
**Decision:** `guardrails.py` validates outreach content before API response.  
**Why:** WhatsApp Business API requires template compliance; catching violations at generation time protects the bank brand.  
**Trade-off:** May require regeneration loops for borderline messages in production.

---

## Limitations

- **Synthetic data:** 600 Faker-generated customers; production requires PII governance and RBI/GDPR compliance.
- **No live WhatsApp send:** Messages are generated and validated, not dispatched via Meta Business API.
- **Model drift:** Propensity model is static; production needs monitoring and retraining.
- **Single-node checkpoint DB:** AsyncSqliteSaver suits demo/HF Spaces; scale-out needs Postgres-backed checkpoints.
- **Sequential scoring:** Batch scoring is synchronous; very large cohorts need queue-based workers.

---

## Setup & Run

### Prerequisites
- Python 3.11+
- Azure OpenAI resource with GPT-4o deployment

### Install

```bash
git clone https://github.com/your-username/banking-crm-agent
cd banking-crm-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Set Azure OpenAI credentials:
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com
# AZURE_OPENAI_DEPLOYMENT=gpt-4o
# AZURE_OPENAI_API_VERSION=2024-02-01
```

### Seed the Database

```bash
python app/db/seed.py
# Output: Seeding complete: 600 customers, <N> products inserted.
```

### Train the Propensity Model

```bash
python app/ml/train_propensity.py
# Output: ROC-AUC: ~0.85, model saved to app/ml/model.pkl
```

### Start Backend

```bash
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Start Frontend

```bash
streamlit run frontend/app.py
# Opens: http://localhost:8501
```

### Run Tests

```bash
# Offline (155 tests)
pytest tests/ -v

# Live production harness (25 tests, backend must be running)
python scripts/brutal_production_test.py
```

### Docker (HF Spaces)

```bash
docker build -t banking-crm-agent .
docker run -p 7860:7860 --env-file .env banking-crm-agent
# Or use start.sh as defined in the image entrypoint
```

---

## Project Structure

```
banking-crm-agent/
├── app/
│   ├── main.py                  # FastAPI entrypoint with lifespan init
│   ├── config.py                # Pydantic Settings
│   ├── agents/
│   │   ├── supervisor.py        # LangGraph StateGraph + nodes
│   │   ├── routing_llm.py       # LLM-driven routing decisions
│   │   ├── base.py              # Shared LLM factory + utilities
│   │   ├── data_agent.py        # CRM tool selection + execution
│   │   ├── scoring_agent.py     # Batch propensity scoring
│   │   ├── product_agent.py     # Rule-engine product matching
│   │   └── outreach_agent.py    # Parallel WhatsApp message generation
│   ├── tools/
│   │   ├── crm_tools.py         # 6 CRM query tools (@tool)
│   │   ├── scoring_tools.py     # XGBoost + heuristic scoring
│   │   ├── product_tools.py     # 13-product rule engine
│   │   └── message_tools.py     # LLM message generator
│   ├── models/
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── schemas.py           # Pydantic API models
│   │   └── customer.py          # Customer profile model
│   ├── services/
│   │   ├── conversation_service.py  # Graph invocation + response building
│   │   └── guardrails.py            # WhatsApp compliance validation
│   ├── prompts/
│   │   └── registry.py          # Centralized prompt management
│   ├── db/
│   │   ├── database.py          # SQLAlchemy + WAL mode
│   │   ├── schema.py            # ORM models
│   │   └── seed.py              # 600 synthetic customers
│   └── ml/
│       ├── train_propensity.py  # XGBoost training script
│       └── model.pkl            # Trained model artifact
├── frontend/
│   └── app.py                   # Streamlit chat UI
├── tests/
│   ├── test_tools.py            # Tool unit tests
│   ├── test_agents.py           # Agent integration tests
│   ├── test_security.py         # Security + abuse tests
│   ├── test_brutal.py           # Brutal edge-case tests
│   ├── test_guardrails.py       # Compliance guardrail tests
│   └── test_prompts.py          # Prompt registry tests
├── scripts/
│   └── brutal_production_test.py  # Live E2E test harness
├── Dockerfile
├── start.sh
├── requirements.txt
├── .env.example
├── PRD.md
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph 0.2 (StateGraph) |
| LLM | Azure OpenAI GPT-4o |
| API | FastAPI + Uvicorn |
| Chat UI | Streamlit |
| Database | SQLite + SQLAlchemy (WAL mode) |
| ML Model | XGBoost + SMOTE |
| Checkpointing | AsyncSqliteSaver (durable) |
| Config | Pydantic Settings |
| Testing | pytest (155 tests) |
| Deployment | Docker (HF Spaces) |

---

## Documentation

- **PRD.md** — Product requirements and acceptance criteria
- **app/prompts/PROMPTFORGE.md** — Prompt engineering notes (if present)

Built for panel review: auditable routing, durable state, compliance guardrails, and a test matrix that treats adversarial input as a first-class requirement.
