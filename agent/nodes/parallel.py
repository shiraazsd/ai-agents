# --------------------------------------------------------------------------------------
# PARALLEL BRANCH NODES + MERGE
# --------------------------------------------------------------------------------------
# OBJECTIVE:
#   Demonstrate parallel fan‑out from a shared state (post-retrieval) into two
#   independent analytic transformations, then converge and merge outputs before
#   final RAG synthesis.
#
# BRANCHES:
#   - branch_summary  : Summarizes the retrieved snippets.
#   - branch_citations: Produces stylized citation bullets (or references).
#
# MERGE:
#   - merge_parallel: Combines 'summary' + 'citations' into a composite answer field
#     IF no answer already exists. Even if 'answer' is present, we append citations
#     to show a merged structure.
#
# WHY STORE IN parallel_parts?
#   Ensures branch outputs do not overwrite each other (each writes a distinct key).
#   The reducer (see state.py) shallow-merges dicts allowing accumulation.
#
# EXTENSIONS:
#   - Add scoring, sentiment, cluster labeling, etc. as more branches.
#   - Track timing metadata in meta for performance analysis.
# --------------------------------------------------------------------------------------
# Parallel branches (Ollama)
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os
from ..state import GraphState
from langchain_ollama import ChatOllama

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0)

def branch_summary(state: GraphState) -> GraphState:
    docs = "\n".join(d.get("raw","") for d in state.get("retrieved_docs", [])[:8])
    resp = _llm.invoke(f"Summarize key points concisely:\n{docs}")
    return {"parallel_parts": {"summary": resp.content}}

def branch_citations(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    numbered = "\n".join(f"{i+1}. {d.get('raw','')}" for i, d in enumerate(docs[:8]))
    resp = _llm.invoke(
        "Produce bullet citations (no fabrication). If unknown source, label Generic.\n\n"
        f"{numbered}"
    )
    return {"parallel_parts": {"citations": resp.content}}

def merge_parallel(state: GraphState) -> GraphState:
    parts = state.get("parallel_parts", {})
    base = state.get("answer") or parts.get("summary", "(no summary)")
    return {"answer": f"{base}\n\nCitations:\n{parts.get('citations','(none)')}"}