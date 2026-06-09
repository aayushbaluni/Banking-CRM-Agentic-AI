"""Shared LLM factory, tool-call executor, and timing utilities."""
import time
import functools
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from app.config import settings
from app.models.state import AgentState


def human_messages_for_llm(state: AgentState) -> list:
    """
    Return only HumanMessage turns for LLM calls.

    Checkpoint history may contain AIMessages with tool_calls but no ToolMessages
    (e.g. after a partial graph run). Passing those causes 400 errors.
    """
    return [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]


def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=temperature,
        max_tokens=1024,
    )


def run_tool_calls(response: AIMessage, tool_map: dict) -> list[dict]:
    """Execute every tool call in an AIMessage and return raw results."""
    results = []
    for tc in getattr(response, "tool_calls", []):
        tool_fn = tool_map.get(tc["name"])
        if tool_fn:
            raw = tool_fn.invoke(tc["args"])
            results.append({"tool": tc["name"], "result": raw})
    return results


def timed_node(fn):
    """
    Decorator for LangGraph node functions.
    Injects duration_ms into every trace entry the node emits.
    Usage: @timed_node on any node that returns {"trace": [...]}
    """
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state) or {}
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        for entry in result.get("trace", []):
            entry["duration_ms"] = elapsed_ms
        return result
    return wrapper
