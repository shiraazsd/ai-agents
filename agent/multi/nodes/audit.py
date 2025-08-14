from __future__ import annotations
from typing import Dict, Any
import hashlib, json

def _post_validate(state: Dict[str, Any]) -> Dict[str, Any]:
    answer = state.get("answer") or state.get("executor_answer") or ""
    issues = []
    if len(answer) > 8000:
        issues.append("answer_too_long")
    if "```exec" in answer:  # simplistic policy example
        issues.append("unescaped_code_block")
    return {"issues": issues, "valid": not issues}

def audit_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("halt"):
        return state
    if state.get("dry_run"):
        # Skip execution path earlier; just passthrough
        return state
    validation = _post_validate(state)
    state["audit"] = validation
    if not validation["valid"]:
        # Rollback side-effects (placeholder)
        state["rolled_back"] = True
        state["reviewed_answer"] = f"Response held for review. Issues: {validation['issues']}"
        state["halt"] = "post_validation_fail"
    else:
        # Stable hash for provenance
        payload = json.dumps({"a": state.get("answer")}, sort_keys=True).encode()
        state["content_hash"] = hashlib.sha256(payload).hexdigest()
    return state