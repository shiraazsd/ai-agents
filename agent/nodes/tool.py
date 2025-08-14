# --------------------------------------------------------------------------------------
# TOOL NODE
# --------------------------------------------------------------------------------------
# PURPOSE:
#   Execute a restricted shell command via the external tool_server (/shell endpoint).
#
# SECURITY CONSIDERATIONS:
#   - tool_server enforces an allowlist ("ls", "pwd", "df", "echo").
#   - This node sends user command unmodified after stripping "shell " prefix.
#   - In production: ALWAYS add stronger validation / sandboxing.
#
# STATE IMPACT:
#   Adds:
#     tool_results: [{"command": <cmd>, "output": <stdout_or_error>}]
#     answer      : Same output (to allow early graph termination).
#
# EXTENSIONS:
#   - Track exit codes separately.
#   - Add multiple tool types (e.g., HTTP fetch) with structured dispatch.
# --------------------------------------------------------------------------------------
from __future__ import annotations
import requests, os
from ..state import GraphState

TOOLS_URL = os.getenv("TOOLS_URL", "http://localhost:8000")

def tool_shell(state: GraphState) -> GraphState:
    # Naive parse: If user typed "shell <command>", extract the substring after prefix.
    user = state["user_input"]
    if user.startswith("shell "):
        cmd = user[6:]
    else:
        cmd = user
    r = requests.post(f"{TOOLS_URL}/shell", json={"command": cmd}, timeout=30)
    result = r.json().get("result", "")
    return {
        "tool_results": [{"command": cmd, "output": result}],
        "answer": result
    }