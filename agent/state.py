from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional, Annotated

# ---------------------------------------------------------------------------------------
# PURPOSE OF THIS MODULE
# ---------------------------------------------------------------------------------------
# This module defines:
#   1. The typed "GraphState" shared across all LangGraph nodes.
#   2. A small "reducer" system (reduce_state) that merges partial updates coming
#      from different nodes (including parallel branches) into one coherent state.
#
# WHY A REDUCER?
# LangGraph lets nodes return *partial* state updates. When multiple edges converge
# (e.g., parallel fan‑out -> merge) we need objective rules for how to merge fields.
# This reducer acts like a simplified "Redux-style" merge:
#   - Some fields are overwritten (scalars / last-writer-wins).
#   - Some fields are appended (lists).
#   - Some fields are shallow-merged (dicts).
#
# DESIGN GOALS:
#   - Make merging explicit & predictable.
#   - Keep state small, serializable, and explainable (good for checkpointing / replay).
#   - Avoid accidental mutation of previously saved snapshots (always copy before merge).
#
# EXTENDING:
#   If you add new keys to GraphState, also update reduce_state so merging semantics
#   remain intentional (not accidental overwrites).
# ---------------------------------------------------------------------------------------

Route = Literal["direct", "rag", "tool"]

# Merge helpers (prev, new) -> merged
def _merge_list(prev: Optional[List], new: Optional[List]):
    if prev is None: prev = []
    if new is None: new = []
    return prev + new

def _merge_dict(prev: Optional[Dict], new: Optional[Dict]):
    merged = {}
    if prev: merged.update(prev)
    if new: merged.update(new)
    return merged

class GraphState(TypedDict, total=False):
    user_input: str                 # Original user query
    route: Route                    # Selected route
    retrieved_docs: Annotated[List[Dict[str, Any]], _merge_list]
    tool_results: Annotated[List[Dict[str, Any]], _merge_list]
    parallel_parts: Annotated[Dict[str, Any], _merge_dict]
    answer: str
    meta: Annotated[Dict[str, Any], _merge_dict]