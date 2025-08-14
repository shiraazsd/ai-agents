# --------------------------------------------------------------------------------------
# RESEARCHER (RAG retrieval if planner indicates)
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os, requests
from ..state import MultiAgentState
from langchain_ollama import ChatOllama

TOOLS_URL = os.getenv("TOOLS_URL", "http://localhost:8000")
MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=MODEL, temperature=0)

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    print('researcher-entered', state)
    if state.get("route") != "rag":
        return {}
    q = state["user_input"]
    r = requests.post(f"{TOOLS_URL}/rag/search", json={"query": q, "k": 6}, timeout=60)
    txt = r.json().get("result", "")
    docs = [{"raw": l} for l in txt.splitlines() if l.strip()]
    print('researcher-search', docs)
    return {
        "retrieved_docs": docs,
        "meta": {"research_hits": len(docs)}
    }