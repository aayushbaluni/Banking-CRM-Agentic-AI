# Demo Video Script — Banking CRM Agentic AI

**Duration:** 5–8 minutes | **Format:** Screen recording with voiceover | **No face cam needed**

---

## Setup Before Recording

- Open browser: `https://ayushbaluni-banking-crm-ai.hf.space`
- Keep GitHub README open in another tab: `https://github.com/aayushbaluni/Banking-CRM-Agentic-AI`
- Zoom browser to 100–110% so text is readable
- Close unnecessary tabs/notifications
- Test mic levels

---

## INTRO (30 seconds)

**Show:** The app landing page with sidebar dashboard

**Say:**

> "Hi, this is a Banking CRM Agentic AI system. It's a multi-agent system built with LangGraph that helps bank Relationship Managers — or RMs — do their job faster.
>
> An RM can type a natural language request, and the system autonomously retrieves customers from the CRM database, scores their conversion likelihood using machine learning, matches them to the right loan product using a rule engine, and generates personalized WhatsApp outreach messages — all in one conversational flow.
>
> Let me walk through three use cases."

**Point out on screen:**
- Sidebar: "600 customers in the CRM, 360 cross-sell pool, 484 high-value"
- Quick query buttons on the left

---

## USE CASE 1 — Personal Loan Discovery (2 minutes)

**Click:** New Conversation

**Type:** `Find high-value customers likely to convert for a personal loan`

**Wait for results (5–15 seconds)**

**Say while waiting:**

> "The system is now running through its full pipeline — the router classifies the intent and loan category, the data agent queries the CRM database, the scoring agent runs the XGBoost propensity model, and the product agent matches each customer to the right personal loan variant."

**When results appear, narrate:**

> "We got [X] recommendations. Each card shows the customer's name, city, occupation, income, credit score, and account balance — all pulled from the real SQLite database, not hardcoded.
>
> The propensity score — like 92% here — comes from our XGBoost model trained on customer features. The tier — High, Medium, or Low — is assigned based on the score threshold.
>
> The product match — Premium Personal Loan at 9.5% — comes from a deterministic rule engine, not the LLM. So we can audit exactly why this customer got this product."

**Now expand the Agent Reasoning Trace:**

> "This is the full agent trace. You can see every step:
> - Router classified this as a CRM task and routed to the data agent
> - Data agent called `get_high_value_customers` — that's a real SQLAlchemy query
> - Scoring agent ran `batch_score_customers` — the XGBoost model scored and ranked them
> - Product agent ran `recommend_product` for each customer — rule-based matching
> - Supervisor synthesized the final response
>
> Each step shows its duration in milliseconds. Total end-to-end was about [X] seconds."

---

## USE CASE 2 — Follow-up Outreach (2 minutes)

**Stay in the same conversation. Type:** `Generate personalized WhatsApp messages for these customers`

**Wait for results**

**Say while waiting:**

> "This is where stateful conversation comes in. The system remembers the customers from the previous turn — it doesn't re-query the database or re-score. It uses LangGraph's AsyncSqliteSaver checkpoint to restore the exact state."

**When messages appear:**

> "Now each card has a WhatsApp message. Let me expand one.
>
> [Read the message] — Notice it uses the customer's first name, has a hook based on their occupation, mentions the specific loan amount and interest rate, has a clear call-to-action — 'Reply YES' — and ends with 'Reply STOP to opt out.'
>
> That opt-out line is mandatory under TRAI regulations. Our guardrails module automatically validates every message for:
> - Length under 300 characters
> - Presence of a CTA
> - Presence of opt-out
> - No guaranteed-approval language
> - Customer's first name is included
>
> The green checkmark means this message passed all compliance checks."

**Expand the trace:**

> "Look at the trace — this time only the router and outreach agent ran. Data, scoring, and product were skipped because the state was already there. That's the benefit of durable checkpointing."

---

## USE CASE 3 — Category Switch (1.5 minutes)

**Stay in the same conversation. Type:** `Actually show me car loan customers instead`

**Wait for results**

**Say:**

> "Now the LLM follow-up classifier detected this is a RESTART — the RM wants a completely different loan category. So the system clears the previous state and runs the full pipeline again — but this time for auto loans.
>
> You can see the products changed — we're now seeing New Car Loan at 9.0% instead of Personal Loan. The customer list is different too — these are customers who don't have an existing auto loan.
>
> We support 7 loan categories total: personal, home, car, business, education, gold, and loan against property. Each has its own eligibility rules, product variants, and scoring heuristics."

---

## ARCHITECTURE WALKTHROUGH (2 minutes)

**Switch to the GitHub README tab. Scroll to the Architecture Diagram.**

**Say:**

> "Let me walk through the architecture.
>
> The RM interacts through a Streamlit chat UI that talks to a FastAPI backend. The backend invokes a LangGraph StateGraph — that's the multi-agent orchestration layer.
>
> The first node is the Router — it uses a three-tier classification system:
> - Tier 1 is an instant O(1) lookup for obvious greetings like 'hi' or 'thanks' — no LLM call needed
> - Tier 2 checks for loan-related keywords — if the message mentions 'loan', 'customer', 'score', it's immediately a CRM task
> - Tier 3 is the LLM — Azure GPT-4o — for anything ambiguous
>
> Once in the CRM pipeline, the LLM classifies the loan category — personal, home, auto, business, education, gold, or LAP. Then data flows through four specialized agents — data retrieval, scoring, product matching, and optionally outreach — each with their own tools."

**Scroll to Tool Design and Usage:**

> "Every tool is a real integration — CRM tools run SQLAlchemy queries against the database, the scoring tool runs XGBoost inference, the product tool uses a deterministic rule engine, and the message tool calls Azure GPT-4o with compliance validation."

---

## TRADE-OFFS (1.5 minutes)

**Stay on README or narrate freely:**

**Trade-off 1 — Linear Pipeline:**

> "The pipeline is sequential — data, then scoring, then products, then outreach. Each step depends on the previous output. A parallel approach would add complexity without benefit because the dependencies are strictly sequential. The trade-off is latency — 5 to 15 seconds end-to-end. In production with thousands of customers, you'd add async workers and queue-based scoring."

**Trade-off 2 — Deterministic Product Rules:**

> "Product matching uses hardcoded rules, not the LLM. The LLM picks the category — 'this is a home loan query' — but rules pick the exact variant and rate. Why? Because interest rates must be auditable. A bank can't tell a regulator 'the AI decided the rate.' Rules give you a paper trail. The trade-off is that when the product team changes rates, someone has to update the rules. In production, this would be config-driven."

**Trade-off 3 — Hybrid ML + Heuristic Scoring:**

> "Personal loans use a trained XGBoost model. The other six categories use domain-expert heuristics. Why not ML for everything? Because we don't have real training data for those categories — training on fake labels would be dishonest. The heuristics weight category-specific factors like business occupation for business loans, age for education loans. The system also never crashes if the model file is missing — heuristics activate automatically as a fallback."

**Trade-off 4 — Compliance Guardrails:**

> "Every WhatsApp message passes through a guardrails module before reaching the RM. It checks length, CTA, opt-out, prohibited language, and personalization. Messages that fail are flagged with a warning badge, not silently dropped. The trade-off is that in production, you'd add a regeneration loop — if a message fails, retry with tighter constraints rather than just flagging it."

---

## CLOSING (30 seconds)

**Switch back to the app.**

**Say:**

> "To summarize — this is a production-hardened multi-agent system with:
> - Real database queries, not hardcoded outputs
> - ML-based scoring with heuristic fallback
> - A deterministic, auditable product engine
> - Compliance-validated WhatsApp outreach
> - Durable conversation state across server restarts
> - And 155 automated tests covering tools, agents, security, guardrails, and edge cases
>
> The system was tested with prompt injection, SQL injection, garbage inputs, and adversarial attacks — zero crashes, zero data leaks.
>
> The code is on GitHub and the live demo is on Hugging Face Spaces. Thank you."

---

## TIMING SUMMARY

| Section | Duration |
|---|---|
| Intro | 30 sec |
| Use Case 1 — Personal Loan | 2 min |
| Use Case 2 — Follow-up Outreach | 2 min |
| Use Case 3 — Category Switch | 1.5 min |
| Architecture Walkthrough | 2 min |
| Trade-offs | 1.5 min |
| Closing | 30 sec |
| **Total** | **~8 min** |

---

## TIPS

- Don't read the script word-for-word — use it as a guide, speak naturally
- Pause briefly when results appear so the panel can read the cards
- If something takes long to load, say "the system is running through its pipeline" — don't sit in silence
- If something errors (unlikely), say "let me try that again" and move on — don't apologize excessively
- Keep energy steady — confident, not rushed, not monotone
- End strong — the closing summary is what the panel remembers
