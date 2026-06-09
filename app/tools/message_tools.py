"""
WhatsApp message generation tools.
All prompts sourced from app/prompts/registry.py.
"""
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from app.config import settings
from app.prompts.registry import get as get_prompt


_llm_instance = None


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.7,
            max_tokens=512,
        )
    return _llm_instance


@tool
def generate_whatsapp_message(recommendation_json: str) -> str:
    """
    Generate a personalised WhatsApp outreach message for a customer based on
    their profile and recommended loan product. Messages are under 300 characters,
    personalised with occupation-relevant hooks, and include a CTA + opt-out line.

    Args:
        recommendation_json: JSON string containing customer profile and product recommendation
    """
    try:
        data = json.loads(recommendation_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid recommendation JSON: {e}", "message": ""})
    customer = data.get("customer", {})
    product = data.get("product", {})
    loan_amount = data.get("recommended_loan_amount", 500000)
    # Use the customer-specific offered rate if available, otherwise fall back to base product rate
    interest_rate = data.get("offered_interest_rate") or product.get("interest_rate", 12)

    prompt = get_prompt("message_tools.whatsapp_generation").format(
        customer_name=customer.get("name", "Customer"),
        product_name=product.get("name", "Personal Loan"),
        occupation=customer.get("occupation", "Professional"),
        city=customer.get("city", "India"),
        monthly_income=int(customer.get("monthly_income", 50000)),
        account_balance=int(customer.get("account_balance", 200000)),
        has_salary_account=customer.get("has_salary_account", False),
        region=customer.get("region", "south"),
        interest_rate=interest_rate,
        loan_amount=int(loan_amount),
    )

    llm = _get_llm()
    response = llm.invoke(prompt)
    message = (response.content or "").strip()

    return json.dumps({
        "customer_id": customer.get("id"),
        "customer_name": customer.get("name"),
        "phone": customer.get("phone"),
        "message": message,
        "product_name": product.get("name"),
        "char_count": len(message),
    })


@tool
def batch_generate_messages(recommendations_json: str) -> str:
    """
    Generate personalised WhatsApp messages for multiple customers in one call.
    Each message is generated individually to ensure personalisation quality.

    Args:
        recommendations_json: JSON string containing a list of recommendation dicts
    """
    try:
        recommendations = json.loads(recommendations_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid recommendations JSON: {e}", "total": 0, "messages": []})
    results = []

    for rec in recommendations:
        try:
            msg_result = generate_whatsapp_message.invoke(json.dumps(rec))
            results.append(json.loads(msg_result))
        except Exception as e:
            results.append({
                "customer_id": rec.get("customer", {}).get("id", "unknown"),
                "error": str(e),
            })

    return json.dumps({"total": len(results), "messages": results})
