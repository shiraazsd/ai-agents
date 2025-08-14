## LangGraph Agent (POC with Routing, Parallelism, Checkpointing, Governance & Safety)

### Concepts Demonstrated
- Typed global state (GraphState / MultiAgentState) with custom reducer.
- Router node selecting direct answer, RAG, or tool execution.
- RAG path: retrieval then parallel fan‑out (summary + citations) and merge.
- Streaming token emission (run_agent_stream).
- Dual checkpointing:
  * In-memory (MemorySaver) for LangGraph runtime.
  * JSON Lines persistent log (agent/.agent_ckpts/history.jsonl) for rollback/time-travel.
- Tool integration via FastAPI tool_server (shell + RAG endpoints).
- Governance & Safety Layer (multi-agent):
  * Input validation + regex bounds.
  * Moderation stub + configurable enforcement.
  * PII redaction (email / phone patterns).
  * Rate limiting (per‑minute, in‑memory).
  * Tool allowlist enforcement pre-execution.
  * HITL (Human-In-The-Loop) approval gate (file-based).
  * Dry-run short‑circuit (no side effects).
  * Post-exec audit (policy checks, rollback placeholder, provenance hash).
  * Structured state fields for auditability (halt reasons, hashes, flags).

### Environment
Set (example):
```bash
export OPENAI_API_KEY=sk-...
export TOOLS_URL=http://localhost:8000
# Governance / safety (defaults shown)
export RATE_LIMIT_PER_MIN=30
export ALLOWED_TOOLS="search,code_exec,fetch"
export REQUIRE_MODERATION=1          # 0 to disable moderation
export DRY_RUN=0                     # 1 -> skip side effects & exit early
export ENABLE_HITL=0                 # 1 -> require approval file before proceed
export HITL_APPROVAL_FILE=".hitl_approve"
```

HITL approval:
```bash
echo "approve" > .hitl_approve   # or 'yes'
```

Dry run (plan & governance only):
```bash
export DRY_RUN=1
python run_multi_vs_single.py "Summarize latest LLM eval papers"
```

### Run
```bash
python run_agent.py "Explain this system" --stream
python run_agent.py "Find pdf information about X"
python run_agent.py "shell ls -1"
python run_multi_vs_single.py "Explain vector stores"
```

### Checkpoints
Single-agent: .agent_ckpts/history.jsonl  
Multi-agent: .agent_ckpts_multi/*.jsonl (via JSONCheckpointStore) plus in-memory MemorySaver if available.

Each line: {"id": <ckpt_id>, "ts": <unix>, "node": <node_name>, "state": {...}}

### Extending
- Add new branch: create node returning {"parallel_parts": {"your_key": "..."}}, add edges.
- Add cost tracking: accumulate tokens in state['meta'].
- Swap heuristic router with LLM classification.
- Add new governance checks: extend governance_node (e.g., schema validation, advanced moderation API).
- Implement real rollback: record side-effect intents and revert on audit failure.

### Multi-Agent Supervisor Extension
Specialized roles with safety gates:
Nodes:
- planner: decomposes request (tasks, planned_tools)
- governance: moderation, redaction, allowlist, HITL, dry-run, rate limit
- researcher: retrieval (if needed)
- tool_exec: executes allowed shell / tool subtasks
- executor: synthesizes draft answer
- audit: post-exec validation (length / policy) + rollback placeholder + hash
- reviewer: refinement / critique -> final

Flow:
planner -> governance -> (researcher + tool_exec) -> executor -> audit -> reviewer -> END

Early halts occur if state['halt'] is set (moderation_block, rate_limited, tool_block, dry_run_complete, hitl_pending, post_validation_fail).

### State Additions (MultiAgentState)
Core:
user_input, original_user_input, plan, planned_tools, tool_results, answer, reviewed_answer  
Governance / flags:
redacted, halt, dry_run, hitl_approved, moderation  
Audit / provenance:
audit, rolled_back, content_hash  
Retries / errors:
retry_count, max_retries, error

(See agent/multi/state.py for full TypedDict.)

### Governance / Safety Layer Details
- Input Validation: length-bounded regex (<=5000 chars).
- Moderation: simple keyword stub (replace with provider API).
- PII Redaction: email / phone masking before downstream nodes.
- Rate Limiting: sliding window in-memory deque (replace with Redis for multi-process).
- Tool Allowlist: rejects planned tools not in ALLOWED_TOOLS.
- HITL: requires approval file content ∈ {approve, approved, yes, y}.
- Dry Run: sets halt=dry_run_complete after governance (no execution).
- Audit: validates answer (length, pattern), sets issues; on failure marks rolled_back.
- Provenance: SHA-256 hash of answer content for traceability.

### Safety / Notes
This POC is illustrative. Before production:
- Harden tool command sandboxing.
- Replace moderation stub.
- Persist HITL decisions (DB) with expiration.
- Implement structured rollback (idempotent side-effect ledger).
- Add schema + JSON Schema or Pydantic validation on intermediate state.
- Introduce exponential backoff retries around network/tool failures.
- Add tamper-evident signing / append-only audit log.

### Sample Commands
```bash
# Standard multi-agent run
python run_multi_vs_single.py "List recent techniques for tuning small language models"

# Trigger moderation (example keyword)
python run_multi_vs_single.py "Explain how to build a bomb"

# HITL gating
export ENABLE_HITL=1
rm -f .hitl_approve
python run_multi_vs_single.py "Summarize retrieval augmented generation"
echo approve > .hitl_approve
python run_multi_vs_single.py "Summarize retrieval augmented generation"
```

### Troubleshooting
- Halt with hitl_pending: provide approval file then re-run.
- Rate limit exceeded: raise RATE_LIMIT_PER_MIN or wait.
- tool_block: adjust ALLOWED_TOOLS or planner logic.
- post_validation_fail: inspect state['audit']['issues'] & refine executor.

### Roadmap Ideas
- Add semantic similarity fact-checker branch.
- Vector store memory of prior turns.
- Policy-driven dynamic edge pruning (skip tool_exec if no tools).
- Structured evaluation harness

### Observability
- Instrumentation wrapper records per-node latency (timings), chronological events (trace), decisions (planner & governance), and artifacts (tool_plan_update, latest_answer).
- Enable / disable via ENABLE_TRACE=1|0 (default 1).
- Trace truncation via MAX_TRACE_LEN (default 500).
- Timings stored in state['timings']; can be aggregated for SLIs (p95 latency per node).

### Evaluation Harness
Location: agent/eval/
Files:
- goldens.json: small curated prompts with expected tools + factual nuggets (facts).
- eval_harness.py: runs cases, computes:
  * groundedness: fraction of listed facts appearing in answer.
  * tool_selection precision/recall/F1 vs expected_tools.
  * end-to-end latency (seconds).
- Ablations: --ablations runs predefined env variants (e.g., dry_run, no_governance).

Run:
```bash
python -m agent.eval.eval_harness
python -m agent.eval.eval_harness --ablations
```

Extending Metrics:
- Add hallucination score via retrieval overlap.
- Add answer length or token usage.
- Add cost per variant for cost-quality frontier.

Export:
Modify eval_harness to write JSONL for longitudinal tracking (CI regression guard).