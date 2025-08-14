# --------------------------------------------------------------------------------------
# ROUTER NODE
# --------------------------------------------------------------------------------------
# PURPOSE:
#   Decide which downstream subgraph should process the user input:
#     - "direct": simple LLM answer without external context/tools.
#     - "rag"   : perform retrieval over documents before answering.
#     - "tool"  : execute a (restricted) shell command and return output.
#
# STRATEGY:
#   A very naive keyword heuristic (fine for a POC). In production you'd likely:
#     - Train / prompt a classifier LLM.
#     - Use vector similarity against route exemplars.
#     - Combine with guardrails (tool invocation policy).
#
# OUTPUT:
#   Returns partial state containing:
#     route : chosen route literal
#     meta  : diagnostic info (who routed, maybe reasoning later)
# --------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Literal
from ..state import GraphState

def route_node(state: GraphState) -> GraphState:
    user_input = state["user_input"].lower()

    # SIMPLE HEURISTICS:
    #   - Contains "shell" or begins with typical shell commands  -> tool
    #   - Mentions documents / pdf / citation cues                -> rag
    #   - Otherwise                                                -> direct
    if "shell" in user_input or user_input.startswith("ls ") or user_input.startswith("pwd"):
        route: Literal["tool"] = "tool"
    elif "document" in user_input or "pdf" in user_input or "according to" in user_input:
        route = "rag"
    else:
        route = "direct"

    return {"route": route, "meta": {"routed_by": "router_node"}}