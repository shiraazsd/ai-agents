# --------------------------------------------------------------------------------------
# RAG (Retrieval Augmented Generation) NODES
# --------------------------------------------------------------------------------------
# NODES:
#   1. rag_retrieve : Calls external tool_server to perform vector similarity search.
#                     Stores raw string lines as structured doc objects.
#   2. rag_generate : Consumes retrieved snippets + question to synthesize final answer.
#
# WHY SPLIT?
#   Separation allows:
#     - Parallel fan‑out branches (summary, citations) to read retrieval output.
#     - Potential caching of retrieval results.
#
# DATA CONTRACT:
#   rag_retrieve adds:
#     retrieved_docs: List[{"raw": <string line>}]
#     meta.rag_hits : Number of lines parsed (for diagnostics).
#
#   rag_generate reads:
#     state['retrieved_docs'] (list)
#     state['user_input']
#
# PROMPTING:
#   rag_generate instructs the model to use ONLY provided snippets (helps honesty).
#
# EXTENSIONS:
#   - Include provenance (source/page) parsed from each line into structured fields.
#   - Add ranking / scoring metadata.
#   - Guard against empty retrieval (fallback to direct answer or ask clarifying).
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os, requests
from ..state import GraphState
from langchain_ollama import ChatOllama

TOOLS_URL = os.getenv("TOOLS_URL", "http://localhost:8000")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0)

def rag_retrieve(state: GraphState) -> GraphState:
    q = state["user_input"]
    r = requests.post(f"{TOOLS_URL}/rag/search", json={"query": q, "k": 4}, timeout=60)
    txt = r.json().get("result", "")
    docs = [{"raw": line} for line in txt.splitlines() if line.strip()]
    return {
        "retrieved_docs": docs,
        "meta": {"rag_hits": len(docs)}
    }

def rag_generate(state: GraphState) -> GraphState:
    snippets = "\n".join(d.get("raw","") for d in state.get("retrieved_docs", [])[:6])
    prompt = (
        "Use ONLY these snippets. If answer absent, say so.\n"
        f"Snippets:\n{snippets}\n\nQuestion: {state['user_input']}\nAnswer:"
    )
    resp = _llm.invoke(prompt)
    return {
        "answer": resp.content,
        "meta": {"rag_model": OLLAMA_CHAT_MODEL}
    }