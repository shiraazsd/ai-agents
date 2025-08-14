# --------------------------------------------------------------------------------------
# REVIEWER
# --------------------------------------------------------------------------------------
# Takes draft_answer (+ optional tool_results + retrieved_docs) and produces:
#   reviewed_answer, critique
# Provides light quality improvements & factuality reminder.
# --------------------------------------------------------------------------------------
from __future__ import annotations
# FIX: import from local multi state module (previously pointed to parent state)
from ..state import MultiAgentState
from langchain_ollama import ChatOllama
import os

MODEL = os.getenv("OLLAMA_CHAT_MODEL","llama3.2:1b")
_llm = ChatOllama(model=MODEL, temperature=0)

def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    draft = state.get("draft_answer", "")
    snippets = "\n".join(d.get("raw","") for d in state.get("retrieved_docs", [])[:4])
    prompt = (
        "You are a reviewer. Improve clarity, keep facts grounded to the provided "
        "snippets (if any). If unsupported claims exist, flag them. "
        "Return improved answer, then a short critique.\n"
        f"Snippets (may be empty):\n{snippets}\n\nDraft:\n{draft}\n\n"
        "Respond with:\nANSWER:\n<improved>\nCRITIQUE:\n<notes>"
    )
    resp = _llm.invoke(prompt).content
    answer, critique = resp, ""
    if "CRITIQUE:" in resp:
        parts = resp.split("CRITIQUE:",1)
        answer = parts[0].replace("ANSWER:","").strip()
        critique = parts[1].strip()
    return {
        "reviewed_answer": answer,
        "critique": critique,
        "meta": {"review_model": MODEL}
    }