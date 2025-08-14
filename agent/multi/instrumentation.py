from __future__ import annotations
import time, os
from typing import Callable, Dict, Any

ENABLE_TRACE = os.getenv("ENABLE_TRACE", "1") == "1"
MAX_TRACE_LEN = int(os.getenv("MAX_TRACE_LEN", "500"))

def instrument(node_name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    if not ENABLE_TRACE:
        return fn
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        before_tools = list(state.get("planned_tools") or [])
        existing_trace_len = len(state.get("trace", []))
        result = fn(state)
        dur = time.perf_counter() - start

        # Emit delta for timings
        result["timings"] = {node_name: dur}

        # Emit delta trace (list aggregator will concatenate)
        if existing_trace_len < MAX_TRACE_LEN:
            result["trace"] = [{
                "node": node_name,
                "t": time.time(),
                "dt": round(dur, 6),
                "halt": result.get("halt"),
            }]

        # Decisions (partial)
        decisions = {}
        if node_name == "planner":
            decisions["planner"] = {
                "planned_tools": result.get("planned_tools"),
                "plan": result.get("plan")
            }
        if node_name == "governance":
            decisions["governance"] = {
                "redacted": result.get("redacted"),
                "halt": result.get("halt"),
                "hitl": result.get("hitl_approved"),
                "dry_run": result.get("dry_run"),
            }
        return result
    return _wrapped