"""Runs a skill's evals.yaml through ADK's evaluator against the marketplace agent.

evals.yaml (all keys except `prompt` optional):

    runs: 1
    criteria: {response_match_score: 0.5}     # overrides/extends defaults
    cases:
      - prompt: How many business days are in the first week of 2026?
        tools: [business_days]                # expected calls, in order; args ignored
        response: There are 4 business days.  # judged for meaning, not wording

Usage: python -m skillkit.evals <plugin/skill>   (one skill per process; see `mkt eval`)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

from . import catalog

JUDGE = {"judge_model": os.getenv("EVAL_JUDGE_MODEL", "gemini-flash-latest"), "num_samples": 1}


def load_spec(skill: catalog.Skill) -> dict:
    spec = yaml.safe_load(skill.evals_file.read_text()) or {}
    return {"cases": spec} if isinstance(spec, list) else spec


def default_criteria(cases: list[dict]) -> dict:
    criteria = {}
    if any("tools" in c for c in cases):
        criteria["tool_trajectory_avg_score"] = {"threshold": 1.0, "match_type": "IN_ORDER", "ignore_args": True}
    if any("response" in c for c in cases):
        criteria["final_response_match_v2"] = {"threshold": 0.5, "judge_model_options": JUDGE}
    return criteria


def to_eval_set(skill: catalog.Skill, cases: list[dict]):
    from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation
    from google.adk.evaluation.eval_set import EvalSet
    from google.genai import types

    def content(text, role):
        return types.Content(role=role, parts=[types.Part(text=text)]) if text else None

    return EvalSet(
        eval_set_id=skill.id.replace("/", "__"),
        eval_cases=[
            EvalCase(
                eval_id=c.get("id", f"case-{i}"),
                conversation=[
                    Invocation(
                        user_content=content(c["prompt"], "user"),
                        final_response=content(c.get("response"), "model"),
                        intermediate_data=IntermediateData(
                            tool_uses=[types.FunctionCall(name=t, args={}) for t in c.get("tools", [])]
                        ),
                    )
                ],
            )
            for i, c in enumerate(cases, 1)
        ],
    )


def run(skill_id: str, root: Path) -> None:
    from google.adk.evaluation.agent_evaluator import AgentEvaluator
    from google.adk.evaluation.eval_config import EvalConfig

    (skill,) = catalog.skills(root, {skill_id})
    spec = load_spec(skill)
    cases = spec["cases"]
    config = EvalConfig(criteria={**default_criteria(cases), **spec.get("criteria", {})})
    asyncio.run(
        AgentEvaluator.evaluate_eval_set(
            agent_module="marketplace_agent",
            eval_set=to_eval_set(skill, cases),
            eval_config=config,
            num_runs=spec.get("runs", 1),
        )
    )


if __name__ == "__main__":
    sid = sys.argv[1]
    root = Path.cwd()
    (s,) = catalog.skills(root, {sid})
    os.environ["SKILLKIT_SKILLS"] = sid
    if (s.path / "fixtures").is_dir():
        os.environ["SKILLKIT_HTTP_FIXTURES"] = str(s.path / "fixtures")
    sys.path.insert(0, str(root))
    run(sid, root)
