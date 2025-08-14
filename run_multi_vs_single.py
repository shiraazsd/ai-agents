# --------------------------------------------------------------------------------------
# COMPARISON HARNESS
# --------------------------------------------------------------------------------------
# Measures latency + captures outputs for single-agent vs multi-agent supervisor graph.
# Quality heuristic: simple length + presence of key terms (placeholder).
# --------------------------------------------------------------------------------------
import time, argparse, json
from agent.graph import run_agent
from agent.multi.graph_multi import run_multi

def quality_score(text: str, query: str) -> float:
    if not text:
        return 0.0
    coverage = sum(1 for w in query.lower().split() if w in text.lower())
    return 0.3*coverage + 0.7*(len(text)/max(len(query),1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    args = ap.parse_args()
    q = args.query

    t0=time.time()
    single = run_agent(q)
    single_t = time.time()-t0
    single_ans = single.get("answer","")

    t1=time.time()
    multi = run_multi(q)
    multi_t = time.time()-t1
    multi_ans = multi.get("reviewed_answer") or multi.get("draft_answer","")

    report = {
        "query": q,
        "single": {
            "latency_s": round(single_t,3),
            "answer": single_ans[:500],
            "quality": round(quality_score(single_ans,q),2)
        },
        "multi": {
            "latency_s": round(multi_t,3),
            "answer": multi_ans[:500],
            "quality": round(quality_score(multi_ans,q),2)
        },
        "delta_latency_s": round(multi_t - single_t,3),
        "quality_improvement": round(quality_score(multi_ans,q)-quality_score(single_ans,q),2)
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()