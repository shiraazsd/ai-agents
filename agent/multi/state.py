# --------------------------------------------------------------------------------------
# MULTI-AGENT STATE
# --------------------------------------------------------------------------------------
# Extends single-agent fields with planning + review artifacts.
# Uses Annotated reducers (LangGraph >=0.6) like the base GraphState.
# --------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Annotated
import operator

def _merge_dict(a: Dict[str, Any] | None, b: Dict[str, Any] | None):
    if not a:
        return b or {}
    if not b:
        return a
    c = a.copy()
    c.update(b)
    return c

class MultiAgentState(TypedDict, total=False):
    # Core IO
    user_input: str
    original_user_input: str
    route: str
    answer: str
    reviewed_answer: str
    executor_answer: str
    draft_answer: str

    # Planning / tools
    plan: str
    planned_tools: List[str]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    used_tools: Annotated[List[str], operator.add]
    tool_errors: Annotated[List[str], operator.add]

    # Retrieval (single & multi-agent)
    retrieved_docs: List[Dict[str, Any]]     # single-agent style retrieved snippets
    memory_query: str
    memory_docs: List[Dict[str, Any]]
    memory_used: bool
    retrieval_cache_hit: bool
    retrieval_latency_s: float

    # Governance / moderation
    redacted: bool
    moderation: Dict[str, Any]
    hitl_approved: bool
    dry_run: bool
    halt: str

    # Post-execution / audit
    audit: Dict[str, Any]
    rolled_back: bool
    content_hash: str

    # Error / retries
    error: str
    retry_count: int
    max_retries: int

    # Metadata
    meta: Dict[str, Any]
    metadata: Dict[str, Any]

    # Observability / aggregation
    trace: Annotated[List[Dict[str, Any]], operator.add]
    timings: Annotated[Dict[str, float], _merge_dict]
    artifacts: Annotated[Dict[str, Any], _merge_dict]
    decisions: Annotated[Dict[str, Any], _merge_dict]
    metrics: Annotated[Dict[str, Any], _merge_dict]