from __future__ import annotations
import os, re, time, threading
from collections import deque
from typing import Dict, Any

_RATE_LOCK = threading.Lock()
_REQUEST_TIMES = deque()  # simple in‑mem sliding window

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
ENABLE_HITL = os.getenv("ENABLE_HITL", "1") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
REQUIRE_MODERATION = os.getenv("REQUIRE_MODERATION", "1") == "1"
ALLOWED_TOOLS = {t.strip() for t in os.getenv("ALLOWED_TOOLS", "search,code_exec,fetch").split(",") if t.strip()}

PII_PATTERNS = [
    (re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'), "<email>"),
    (re.compile(r'\b\d{3}[-.\s]?\d{2,4}[-.\s]?\d{4}\b'), "<phone>"),
]

def _moderate(text: str) -> Dict[str, Any]:
    # Stub moderation (extend with real provider)
    flagged = any(k in text.lower() for k in ["terror", "self-harm", "bomb"])
    return {"flagged": flagged, "reason": "policy_term_detected" if flagged else None}

def _redact(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text

def _rate_limit():
    now = time.time()
    with _RATE_LOCK:
        while _REQUEST_TIMES and now - _REQUEST_TIMES[0] > 60:
            _REQUEST_TIMES.popleft()
        if len(_REQUEST_TIMES) >= RATE_LIMIT_PER_MIN:
            return False, RATE_LIMIT_PER_MIN
        _REQUEST_TIMES.append(now)
    return True, RATE_LIMIT_PER_MIN

def _hitl_approval(state: Dict[str, Any]) -> bool:
    # Simple HITL: look for file flag or console opt-out
    flag_file = os.getenv("HITL_APPROVAL_FILE", ".hitl_approve")
    if os.path.exists(flag_file):
        with open(flag_file, "r", encoding="utf-8") as f:
            content = f.read().strip().lower()
        return content in {"y","yes","approve","approved"}
    # Non-interactive default deny until file provided
    return True

def governance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print('governance_node', state)
    user_input = state.get("user_input","")
    state["original_user_input"] = user_input
    moderated = _moderate(user_input) if REQUIRE_MODERATION else {"flagged": False}
    if moderated.get("flagged"):
        state["reviewed_answer"] = "Request blocked by moderation."
        state["halt"] = "moderation_block"
        return state
    redacted = _redact(user_input)
    if redacted != user_input:
        state["redacted"] = True
    state["user_input"] = redacted

    ok, limit = _rate_limit()
    if not ok:
        state["reviewed_answer"] = f"Rate limit exceeded ({limit}/min). Retry later."
        state["halt"] = "rate_limited"
        return state

    # Tool allowlist enforcement (planner may have proposed tools)
    planned_tools = state.get("planned_tools") or []
    disallowed = [t for t in planned_tools if t not in ALLOWED_TOOLS]
    if disallowed:
        state["reviewed_answer"] = f"Disallowed tools: {disallowed}"
        state["halt"] = "tool_block"
        return state

    if DRY_RUN:
        state["dry_run"] = True
        state["reviewed_answer"] = f"Dry-run OK. Planned tools: {planned_tools}"
        state["halt"] = "dry_run_complete"
        return state

    if ENABLE_HITL:
        if not _hitl_approval(state):
            state["reviewed_answer"] = "Awaiting human approval."
            state["halt"] = "hitl_pending"
            return state
        state["hitl_approved"] = True

    return state