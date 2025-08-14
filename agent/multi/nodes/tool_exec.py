# --------------------------------------------------------------------------------------
# OPTIONAL TOOL EXECUTOR (shell) - triggered if tasks reference a shell op
# --------------------------------------------------------------------------------------
from __future__ import annotations
import os, requests
from ..state import MultiAgentState

TOOLS_URL = os.getenv("TOOLS_URL","http://localhost:8000")

def tool_executor_node(state: MultiAgentState) -> MultiAgentState:
    # naive detection
    shell_tasks = [t for t in state.get("tasks", []) if "shell" in t.lower()]
    outputs=[]
    for t in shell_tasks:
        # extract command after a colon or 'shell '
        cmd = t.split("shell",1)[1].strip(": ").strip()
        try:
            r = requests.post(f"{TOOLS_URL}/shell", json={"command": cmd}, timeout=25)
            out = r.json().get("result","")
        except Exception as e:
            out = f"[tool error] {e}"
        outputs.append({"task": t, "command": cmd, "output": out})
    if not outputs:
        return {}
    return {
        "tool_results": outputs,
        "meta": {"tool_exec_count": len(outputs)}
    }