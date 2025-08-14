# --------------------------------------------------------------------------------------
# LANGGRAPH GRAPH (Updated for langgraph 0.6.x API: no 'reducer' arg)
# Per-field merging specified in state.py via Annotated reducers.
# --------------------------------------------------------------------------------------
from __future__ import annotations
from typing import AsyncIterator
import os
from langgraph.graph import StateGraph, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

from .state import GraphState
from .checkpoint import JSONCheckpointStore
from .nodes.router import route_node
from .nodes.direct import direct_answer
from .nodes.rag import rag_retrieve, rag_generate
from .nodes.tool import tool_shell
from .nodes.parallel import branch_summary, branch_citations, merge_parallel

THREAD_ID = os.getenv("AGENT_THREAD_ID", "default-thread")  # required when using a checkpointer

builder = StateGraph(GraphState)
builder.add_node("router", route_node)
builder.add_node("direct_answer", direct_answer)
builder.add_node("rag_retrieve", rag_retrieve)
builder.add_node("rag_generate", rag_generate)
builder.add_node("tool_shell", tool_shell)
builder.add_node("branch_summary", branch_summary)
builder.add_node("branch_citations", branch_citations)
builder.add_node("merge_parallel", merge_parallel)
builder.set_entry_point("router")

def _route_decider(state: GraphState):
    r = state.get("route")
    if r == "direct":
        return "direct_answer"
    if r == "rag":
        return "rag_retrieve"
    return "tool_shell"

builder.add_conditional_edges(
    "router",
    _route_decider,
    {"direct_answer": "direct_answer", "rag_retrieve": "rag_retrieve", "tool_shell": "tool_shell"},
)
builder.add_edge("direct_answer", END)
builder.add_edge("tool_shell", END)
builder.add_edge("rag_retrieve", "branch_summary")
builder.add_edge("rag_retrieve", "branch_citations")
builder.add_edge("branch_summary", "merge_parallel")
builder.add_edge("branch_citations", "merge_parallel")
builder.add_edge("merge_parallel", "rag_generate")
builder.add_edge("rag_generate", END)

memory = MemorySaver() if MemorySaver else None
graph = builder.compile(checkpointer=memory) if memory else builder.compile()
_ckpt = JSONCheckpointStore()

def _config():
    # Only supply thread_id if a checkpointer is active
    return {"configurable": {"thread_id": THREAD_ID}} if memory else {}

async def run_agent_stream(user_input: str) -> AsyncIterator[str]:
    initial: GraphState = {"user_input": user_input}
    yielded_any = False
    final_answer = None

    async for event in graph.astream_events(initial, config=_config(), version="v1"):
        etype = event.get("type") or event.get("event")
        if not etype:
            continue

        st = event.get("state") or event.get("data", {}).get("state")
        if st and isinstance(st, dict):
            if etype in {"on_chain_end", "chain.end", "graph:node:end", "graph:step:end"}:
                node = (
                    ",".join(event.get("tags", []))
                    or event.get("name")
                    or event.get("node_name")
                    or "node"
                )
                _ckpt.append(st, node=node)
            if "answer" in st:
                final_answer = st["answer"]

        data = event.get("data") or {}
        chunk_text = None

        if "chunk" in data:
            c = data["chunk"]
            raw = getattr(c, "content", None) or getattr(c, "text", None)
            if raw is None and isinstance(c, dict):
                raw = c.get("content") or c.get("text")
            # If attribute is a method (callable), call it
            if callable(raw):
                try:
                    raw = raw()
                except Exception:
                    raw = None
            if isinstance(raw, (list, tuple)):
                # Some providers emit list of parts; join them
                raw = "".join(str(p) for p in raw)
            if raw is not None:
                chunk_text = str(raw)

        elif "delta" in data and isinstance(data["delta"], dict):
            d = data["delta"]
            raw = d.get("content") or d.get("text")
            if callable(raw):
                try:
                    raw = raw()
                except Exception:
                    raw = None
            if raw is not None:
                chunk_text = str(raw)

        if chunk_text:
            yielded_any = True
            yield chunk_text

    if not yielded_any and final_answer:
        yield str(final_answer)

def run_agent(user_input: str) -> GraphState:
    final = graph.invoke({"user_input": user_input}, config=_config())
    _ckpt.append(final, node="FINAL")
    return final

def rollback_to(checkpoint_id: str):
    return _ckpt.rollback(checkpoint_id)

def time_travel(index: int):
    return _ckpt.time_travel(index)