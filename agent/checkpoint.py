# --------------------------------------------------------------------------------------
# CHECKPOINTING / TIME TRAVEL
# --------------------------------------------------------------------------------------
# PURPOSE:
#   Lightweight JSONL-based persistence layer for graph state snapshots. While LangGraph
#   offers built‑in checkpointers (e.g., MemorySaver, Redis, SQL), this custom store:
#     - Persists every node-completed state to disk (append-only log).
#     - Enables rollback (truncate after a given checkpoint id).
#     - Enables "time travel" (read historic state without mutating the log).
#
# DESIGN PRINCIPLES:
#   - Append-only for durability & auditability.
#   - Human-inspectable (JSON Lines).
#   - Minimal dependencies (just stdlib).
#
# LIMITATIONS:
#   - Entire state snapshot stored each line (simple, but can grow large).
#   - No compaction / GC (you can add a periodic "squash" if needed).
#   - Not concurrency-safe (single-process POC).
#
# EXTENSIONS (ideas):
#   - Add compression (gzip) for large histories.
#   - Shard by session id (if multi-user).
#   - Add indexing (by node name, timestamp range).
# --------------------------------------------------------------------------------------
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from .state import GraphState

class JSONCheckpointStore:
    """
    Simple JSONL store.

    FILE FORMAT (history.jsonl):
      {"id": "...", "ts": <float epoch>, "node": "<node_name>", "state": {...}}
      {"id": "...", "ts": <float epoch>, "node": "<node_name>", "state": {...}}
      ...

    Each record is a COMPLETE state snapshot after a node finishes (post-reducer).
    """

    def __init__(self, path: str = ".agent_ckpts"):
        self.base = Path(path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.file = self.base / "history.jsonl"

    # ------------------------------------------------------------------
    # append: persist a new snapshot generated post-node execution.
    # Returns generated checkpoint id (UUID4 hex short).
    # ------------------------------------------------------------------
    def append(self, state: GraphState, node: str):
        rec = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "node": node,
            "state": state
        }
        with self.file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec["id"]

    # ------------------------------------------------------------------
    # load_all: read full history (for small POC it's fine).
    # For large logs you'd stream / iterate instead.
    # ------------------------------------------------------------------
    def load_all(self) -> List[Dict[str, Any]]:
        if not self.file.exists():
            return []
        return [
            json.loads(l)
            for l in self.file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    # ------------------------------------------------------------------
    # latest_state: convenience to pick final snapshot (tail).
    # Returns None if no snapshots.
    # ------------------------------------------------------------------
    def latest_state(self) -> Optional[GraphState]:
        records = self.load_all()
        if not records:
            return None
        return records[-1]["state"]

    # ------------------------------------------------------------------
    # rollback: destructive operation.
    #   - Truncates log AFTER the specified checkpoint id.
    #   - Returns state at that checkpoint (or None if not found).
    #
    # STRATEGY:
    #   Iterate records until checkpoint found; rewrite file with that prefix.
    # ------------------------------------------------------------------
    def rollback(self, checkpoint_id: str) -> Optional[GraphState]:
        records = self.load_all()
        keep = []
        rolling_state: Optional[GraphState] = None
        found = False
        for r in records:
            keep.append(r)
            if r["id"] == checkpoint_id:
                found = True
                rolling_state = r["state"]
                break
        if not found:
            return None  # checkpoint id not present
        with self.file.open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        return rolling_state

    # ------------------------------------------------------------------
    # time_travel: non-destructive read of a historical index.
    #   index=0 is earliest snapshot.
    #   Clamps out-of-range indices.
    # ------------------------------------------------------------------
    def time_travel(self, index: int) -> Optional[GraphState]:
        records = self.load_all()
        if not records:
            return None
        index = max(0, min(index, len(records)-1))
        return records[index]["state"]