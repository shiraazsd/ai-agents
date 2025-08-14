from __future__ import annotations
import json, time, statistics, argparse, os
from typing import List, Dict, Any
from agent.multi.graph_multi import run_multi

def load_goldens(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def groundedness(answer: str, facts: List[str]) -> float:
    if not facts:
        return 1.0
    answer_l = answer.lower()
    hits = sum(1 for f in facts if f.lower() in answer_l)
    return hits / len(facts)

def tool_selection_accuracy(state: Dict[str, Any], expected: List[str]) -> Dict[str, float]:
    planned = set(state.get("planned_tools") or [])
    exp = set(expected)
    if not exp and not planned:
        return {"precision": 1, "recall": 1, "f1": 1}
    if not planned:
        return {"precision": 0, "recall": 0, "f1": 0}
    tp = len(planned & exp)
    precision = tp / len(planned) if planned else 0
    recall = tp / len(exp) if exp else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    state = run_multi(case["input"])
    latency = time.perf_counter() - start
    answer = state.get("reviewed_answer") or state.get("answer") or ""
    g = groundedness(answer, case.get("facts", []))
    tool_metrics = tool_selection_accuracy(state, case.get("expected_tools", []))
    return {
        "id": case["id"],
        "latency_s": latency,
        "groundedness": g,
        **{f"tool_{k}": v for k,v in tool_metrics.items()},
        "halt": state.get("halt"),
    }

def ablations(cases, variants):
    results = []
    for name, env in variants.items():
        print(f"[ABLATION] {name}")
        # Apply env overrides
        prev = {}
        for k,v in env.items():
            prev[k] = os.getenv(k)
            os.environ[k] = str(v)
        for c in cases:
            r = run_case(c)
            r["variant"] = name
            results.append(r)
        # restore
        for k,v in prev.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v
    return results

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for metric in ["latency_s","groundedness","tool_precision","tool_recall","tool_f1"]:
        vals = [r[metric] for r in rows if metric in r and r.get("halt") is None]
        if vals:
            out[f"{metric}_mean"] = statistics.mean(vals)
            out[f"{metric}_p95"] = sorted(vals)[int(0.95*len(vals))-1]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default="goldens.json")
    ap.add_argument("--ablations", action="store_true")
    args = ap.parse_args()

    cases = load_goldens(args.goldens)
    base_rows = [run_case(c) for c in cases]
    print("BASE:", json.dumps({"cases": base_rows, "aggregate": aggregate(base_rows)}, indent=2))

    if args.ablations:
        variants = {
            "no_governance": {"REQUIRE_MODERATION": 0, "ENABLE_HITL": 0, "DRY_RUN": 0},
            "dry_run": {"DRY_RUN": 1},
        }
        abl_rows = ablations(cases, variants)
        print("ABLATIONS:", json.dumps({"rows": abl_rows}, indent=2))

if __name__ == "__main__":
    main()