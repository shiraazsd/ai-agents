# --------------------------------------------------------------------------------------
# EXECUTOR
# --------------------------------------------------------------------------------------
# Synthesizes draft answer:
#   - If RAG: incorporate snippets
#   - Else: direct reasoning
# --------------------------------------------------------------------------------------
from __future__ import annotations
from ..state import MultiAgentState
from langchain_ollama import ChatOllama
import os

MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=MODEL, temperature=0)

def executor_node(state: MultiAgentState) -> MultiAgentState:
    q = state["user_input"]
    if state.get("route") == "rag":
        snippets = "\n".join(d.get("raw","") for d in state.get("retrieved_docs", [])[:8])
        prompt = (
            f"Snippets:\n{snippets}\n\nQuestion: {q}\nDraft:"
        )
    else:
        prompt = f"Provide a concise, structured answer:\nQuestion: {q}\nDraft:"
    print('promt-executor', prompt)
    resp = _llm.invoke(prompt)
    print('answer-executor', resp)
    return {
        "draft_answer": resp.content,
        "meta": {"executor_model": MODEL}
    }