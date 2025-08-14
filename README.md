## LangGraph Multi-Agent POC (Routing • RAG • Governance • Observability • Eval • Memory • MCP Tools)

### Key Capabilities
- Single & Multi-agent execution (planner → governance → researcher/tool_exec → executor → audit → reviewer).
- Typed global state (MultiAgentState) with Annotated reducers (merge lists/dicts across parallel nodes).
- Governance & Safety:
  - Input length validation.
  - Moderation stub + configurable enforcement.
  - PII redaction (email / phone patterns).
  - Rate limiting (per-minute).
  - Tool allowlist + HITL approval gate (file-based).
  - Dry-run mode (halts before side-effects).
  - Post-execution audit (policy checks, rollback placeholder, provenance hash).
- Tools & Data (MCP-first):
  - Local stdio tool allowlist (ls, cat, grep, etc.).
  - Remote MCP-style HTTP servers (e.g., web, calendar) with normalized naming: mcp:web.search.
  - Hybrid Retrieval Memory: BM25 + pseudo-embedding + cache (HybridStore).
  - Ingestion helper for long-term memory.
- RAG Enhancements:
  - Hybrid rerank (embedding similarity + keyword overlap) with query expansion.
  - Tunable chunk size/overlap & scoring blend weight.
  - Term highlighting & per-chunk scoring diagnostics.
- Observability:
  - Instrumented per-node timings, chronological trace, decisions, artifacts.
  - Structured logging (APP_LOG_LEVEL).
  - Optional prompt logging / redaction (executor).
- Evaluation Harness:
  - Goldens-based groundedness, tool-selection precision/recall/F1, latency.
  - Ablations (e.g., disable governance, dry-run).
  - JSONL history append for longitudinal tracking.
- Checkpointing & Time Travel:
  - LangGraph in-memory (MemorySaver if available).
  - Persistent JSONL (.agent_ckpts_multi/history.jsonl).
  - Rollback / logical resume from stored state (strip terminal fields).
- Rollback Harness:
  - List, show, resume checkpoints via run_multi_vs_single.py flags.
- Logging Everywhere:
  - Graph invocation, node instrumentation, tool server hybrid search, executor prompt.

---

### Environment Variables (Core Groups)

Core:
```
OPENAI_API_KEY=...
TOOLS_URL=http://localhost:8000
APP_LOG_LEVEL=INFO
```

Governance / Safety:
```
RATE_LIMIT_PER_MIN=30
ALLOWED_TOOLS="search,code_exec,fetch,ls,mcp:web.search,mcp:calendar.lookup"
REQUIRE_MODERATION=1
DRY_RUN=0
ENABLE_HITL=0
HITL_APPROVAL_FILE=.hitl_approve
```

Prompt Logging:
```
REDACT_EXECUTOR_PROMPT=0   # 1 to hide full prompt, logs only length
```

Observability:
```
ENABLE_TRACE=1
MAX_TRACE_LEN=500
```

Hybrid Retrieval / Memory:
```
VECTOR_STORE_PATH=.vector_store.jsonl
RETRIEVAL_TOP_K=5
RETRIEVAL_CACHE_TTL_S=300
RETRIEVAL_CACHE_MAX=200
```

Hybrid RAG (tool_server):
```
RAG_HYBRID=1
RAG_EMBED_CAND_MULT=4
RAG_RERANK_KEYWORD_WEIGHT=0.35
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=150
RAG_QUERY_EXPANSION=1
```

Tools (MCP / stdio):
```
STDIO_TOOLS="ls,cat,grep"
MCP_SERVERS="web:http://localhost:9101,calendar:http://localhost:9102"
```

Executor / Model:
```
OLLAMA_CHAT_MODEL=llama3.2:1b
```

---

### Running

Single agent:
```
python run_agent.py "Explain vector stores"
```

Multi agent:
```
python run_multi_vs_single.py "List recent techniques for tuning small LLMs"
```

Dry-run (plan + governance only):
```
export DRY_RUN=1
python run_multi_vs_single.py "Summarize latest LLM eval papers"
```

HITL:
```
export ENABLE_HITL=1
rm -f .hitl_approve
python run_multi_vs_single.py "Summarize retrieval augmented generation"   # halts (hitl_pending)
echo approve > .hitl_approve
python run_multi_vs_single.py "Summarize retrieval augmented generation"   # proceeds
```

Prompt redaction:
```
export REDACT_EXECUTOR_PROMPT=1
python run_multi_vs_single.py "Explain embeddings"
```

---

### Time Travel / Rollback

List recent checkpoints:
```
python run_multi_vs_single.py --list-ckpts --limit 10
```

Show checkpoint details:
```
python run_multi_vs_single.py --show-ckpt <CKPT_ID>
```

Resume (logical rollback) from a prior state:
```
python run_multi_vs_single.py --resume-ckpt <CKPT_ID>
```
Mechanism strips terminal fields (answer, reviewed_answer, audit, halt, etc.) and replays graph with remaining context (plan, tools, memory artifacts).

---

### Evaluation Harness

Goldens JSON (facts + expected_tools). Run:
```
python eval_harness.py --goldens goldens.json
python eval_harness.py --goldens goldens.json --ablations
```

Outputs aggregate metrics (mean / p95) and (optionally via enhanced version) can append JSONL for longitudinal tracking.

Metrics:
- groundedness = fraction of fact strings found in answer
- tool_precision / recall / f1 vs expected tool set
- latency_s per case

---

### Hybrid Retrieval (Tool Server)

Hybrid scoring:
```
final_score = (1 - RAG_RERANK_KEYWORD_WEIGHT)*normalized_embedding + RAG_RERANK_KEYWORD_WEIGHT*keyword_overlap
```
Query expansion adds simple synonyms (progress → status, advancement, etc.). Results logged with emb / kw / final scores plus term highlighting.

Chunking controlled by:
```
RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP
```

---

### Long-Term Memory / HybridStore

Pseudo-embedding + BM25 approximation:
- Adds docs to JSONL store
- Deterministic hash-based embeddings (no external model)
- Sliding TTL cache for queries (query → ranked docs)

Ingestion example:
```
python -m agent.memory.ingest_seed ./docs
```

State fields:
```
memory_query, memory_docs, memory_used, retrieval_cache_hit, retrieval_latency_s
```

---

### Tools (MCP + Local)

Planner emits:
- stdio: 'ls'
- mcp:web.search
- mcp:calendar.lookup

Execution normalizes specs:
```
{"type":"stdio","tool":"ls","args":["-1"]}
{"type":"mcp","server":"web","method":"search","payload":{"q":"langgraph"}}
```

Set ALLOWED_TOOLS & STDIO_TOOLS to control governance allowlist.

---

### Observability & Logging

Instrumentation (per node):
- trace: chronological node events
- timings: cumulative seconds per node
- artifacts: e.g. tool_plan_update, latest_answer
- decisions: planner & governance summaries
- metrics: extensible runtime metrics

Logging:
- Multi graph & executor use standard logging (APP_LOG_LEVEL).
- Executor logs full or redacted prompt.
- Hybrid search logs scoring path.

---

### State (Excerpt)

MultiAgentState includes (partial):
```
user_input, original_user_input, route,
plan, planned_tools, tool_results, used_tools, tool_errors,
memory_query, memory_docs, memory_used, retrieval_cache_hit, retrieval_latency_s,
draft_answer, answer, reviewed_answer,
moderation, redacted, halt, dry_run, hitl_approved,
audit, rolled_back, content_hash,
trace, timings, artifacts, decisions, metrics, meta
```

---

### Checkpoints

Stored at:
```
.agent_ckpts_multi/history.jsonl
```
Each line:
```
{"id": "...", "ts": <unix>, "node":"<node_name>", "state": {...}}
```

Use rollback harness (see Time Travel).

---

### Governance Halt Codes

Possible values in state['halt']:
```
moderation_block
rate_limited
tool_block
dry_run_complete
hitl_pending
post_validation_fail
```

---

### Security / Hardening (Next Steps)
- Replace moderation stub with provider API.
- Real rollback for side-effects: maintain ledger & idempotent compensation.
- Sandbox stdio tools or shift to constrained microservice.
- Stronger PII detection (NLP / regex mix).
- Add JSON Schema validation per node’s output.
- Use real embeddings (e.g., sentence-transformers) & FAISS for retrieval with encryption at rest.
- OpenTelemetry exporter for distributed tracing.
- Auth / ACL around tool server.

---

### Sample Combined Flow

1. planner: generate plan + tool list
2. governance: moderate, redact, gate, dry-run / HITL
3. researcher & tool_exec (parallel): hybrid retrieval + tool invocation
4. executor: build draft answer (logs prompt)
5. audit: validate & hash content
6. reviewer: refine / finalize (answer or reviewed_answer)
7. checkpoint persisted

---

### Troubleshooting

| Issue | Action |
|-------|--------|
| hitl_pending | Write approval file (echo approve > .hitl_approve) & re-run |
| Rate limit exceeded | Increase RATE_LIMIT_PER_MIN or wait 60s |
| tool_block | Add missing tools to ALLOWED_TOOLS |
| post_validation_fail | Inspect state['audit']['issues'] |
| Poor RAG results | Tune RAG_RERANK_KEYWORD_WEIGHT, CHUNK_SIZE/OVERLAP, add more corpus docs |
| Prompt not logged | Set REDACT_EXECUTOR_PROMPT=0 and APP_LOG_LEVEL=INFO |

---

### Git / Repo Setup (Quick)

```
git init
git add .
git commit -m "chore: initial commit"
# create repo (gh CLI) or manually then:
git remote add origin git@github.com:<org>/<repo>.git
git branch -M main
git push -u origin main
```

Add to .gitignore (already recommended):
```
.agent_ckpts/
.agent_ckpts_multi/
.vector_store.jsonl
.hitl_approve
.env
```

---

### Disclaimer
Prototype code; not production-hardened. Apply stronger security, validation, and monitoring before real-world use.