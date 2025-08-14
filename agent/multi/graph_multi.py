# --------------------------------------------------------------------------------------
# MULTI-AGENT SUPERVISOR GRAPH (with governance & safety layers)
# --------------------------------------------------------------------------------------
from __future__ import annotations
from typing import AsyncIterator
import os, re
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

from .state import MultiAgentState
from ..checkpoint import JSONCheckpointStore
from .nodes.planner import planner_node
from .nodes.researcher import researcher_node
from .nodes.executor import executor_node
from .nodes.tool_exec import tool_executor_node
from .nodes.reviewer import reviewer_node
from .nodes.governance import governance_node
from .nodes.audit import audit_node
from .instrumentation import instrument

MULTI_THREAD_ID = os.getenv("MULTI_AGENT_THREAD_ID", "multi-default-thread")

builder = StateGraph(MultiAgentState)
# Wrap nodes with instrumentation
builder.add_node("planner", instrument("planner", planner_node))
builder.add_node("governance", instrument("governance", governance_node))
builder.add_node("researcher", instrument("researcher", researcher_node))
builder.add_node("tool_exec", instrument("tool_exec", tool_executor_node))
builder.add_node("executor", instrument("executor", executor_node))
builder.add_node("audit", instrument("audit", audit_node))
builder.add_node("reviewer", instrument("reviewer", reviewer_node))
builder.set_entry_point("planner")

# Revised edges with gates
builder.add_edge("planner", "governance")
builder.add_edge("governance", "researcher")
builder.add_edge("governance", "tool_exec")
builder.add_edge("researcher", "executor")
builder.add_edge("tool_exec", "executor")
builder.add_edge("executor", "audit")
builder.add_edge("audit", "reviewer")
builder.add_edge("reviewer", END)

memory = MemorySaver() if MemorySaver else None
graph_multi = builder.compile(checkpointer=memory) if memory else builder.compile()
_ckpt = JSONCheckpointStore(path=".agent_ckpts_multi")

SAFE_INPUT_REGEX = re.compile(r'^[\s\S]{1,5000}$')

def _config():
    return {"configurable": {"thread_id": MULTI_THREAD_ID}} if memory else {}

def _validate_input(user_input: str):
    if not user_input or not SAFE_INPUT_REGEX.match(user_input):
        raise ValueError("Invalid input.")
    return user_input

async def run_multi_stream(user_input: str) -> AsyncIterator[str]:
    user_input = _validate_input(user_input)
    initial: MultiAgentState = {"user_input": user_input}
    final_answer = None
    async for evt in graph_multi.astream_events(initial, config=_config(), version="v1"):
        etype = evt.get("type") or evt.get("event")
        st = evt.get("state") or evt.get("data", {}).get("state")
        if st and etype in {"on_chain_end","graph:node:end"}:
            _ckpt.append(st, node=";".join(evt.get("tags", [])) or evt.get("name") or "node")
            if "reviewed_answer" in st:
                final_answer = st["reviewed_answer"]
            elif st.get("answer"):
                final_answer = st["answer"]
    if final_answer:
        yield final_answer

def run_multi(user_input: str) -> MultiAgentState:
    user_input = _validate_input(user_input)
    final = graph_multi.invoke({"user_input": user_input}, config=_config())
    _ckpt.append(final, node="FINAL")
    return final