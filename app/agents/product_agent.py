"""Product Agent — matches scored customers to eligible loan products."""
import json
from app.models.state import AgentState
from app.tools.product_tools import recommend_product
from app.agents.base import timed_node


@timed_node
def product_agent_node(state: AgentState) -> dict:
    """Apply the rule engine to recommend a loan product for each scored customer."""
    scored = state.get("scored_customers", [])
    loan_category = state.get("loan_category", "personal_loan")
    recommendations: list[dict] = []
    ineligible = 0

    for item in scored:
        try:
            item_with_category = {**item, "loan_category": loan_category}
            raw = recommend_product.invoke({"scored_customer_json": json.dumps(item_with_category)})
            rec = json.loads(raw)
            if rec.get("eligible"):
                recommendations.append({**item, **rec})
            else:
                ineligible += 1
        except Exception:
            ineligible += 1

    products_matched = {r.get("product", {}).get("id") for r in recommendations}

    return {
        "final_recommendations": recommendations,
        "trace": [{
            "step": "product_agent",
            "decision": "Applied deterministic rule engine (income + credit + account type → product)",
            "tools_called": ["recommend_product"],
            "result_summary": (
                f"{len(recommendations)} eligible | {ineligible} ineligible | "
                f"products: {', '.join(sorted(products_matched)) or 'none'}"
            ),
            "skipped": False,
        }],
    }
