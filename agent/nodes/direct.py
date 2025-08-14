# --------------------------------------------------------------------------------------
# DIRECT (local Ollama)
# --------------------------------------------------------------------------------------
# PURPOSE:
#   Provide a concise answer without retrieval or tool calls.
#
# NOTES:
#   - Uses ChatOpenAI (LangChain wrapper). You can swap with any chat model provider.
#   - Keep temperature low for determinism (important when learning / testing).
#   - Only consumes 'user_input'; ignores previous intermediate fields.
#
# EXTENSIONS:
#   - Add system prompt for style consistency.
#   - Track token usage and push into state['meta'].
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os
from ..state import GraphState
from langchain_ollama import ChatOllama

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0)

def direct_answer(state: GraphState) -> GraphState:
    resp = _llm.invoke(f"Answer directly and concisely:\n{state['user_input']}")
    return {
        "answer": resp.content,
        "meta": {"direct_model": OLLAMA_CHAT_MODEL}
    }