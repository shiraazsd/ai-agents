# --------------------------------------------------------------------------------------
# PLANNER
# --------------------------------------------------------------------------------------
# Produces: plan (string), tasks (list[str]), route guess (to guide researcher/executor)
# Simple heuristic + optional LLM expansion (local Ollama).
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os
from ..state import MultiAgentState
from langchain_ollama import ChatOllama

MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
_llm = ChatOllama(model=MODEL, temperature=0)

def planner_node(state: MultiAgentState) -> MultiAgentState:
    q = state["user_input"]
    prompt = (
        "Decompose the user request into 2-4 concise tasks. "
        "Label if retrieval needed in case specific questions are asked. Return format must follow exact format: task | needs_rag(bool).\n"
        f"Request: {q}"
    )
    print('planner-prompt', prompt)
    resp = _llm.invoke(prompt).content
    print('planner-prompt-output', resp)
    tasks = []
    needs_rag = False
    for line in resp.splitlines():
        line=line.strip()
        if not line: continue
        tasks.append(line)
        if "true" in line.lower() or "yes" in line.lower(): needs_rag = True
    route = "rag" if needs_rag else "direct"
    print('planner-route', route)
    return {
        "plan": resp,
        "tasks": tasks,
        "route": route,  # may be refined by researcher
        "meta": {"planner_model": MODEL}
    }