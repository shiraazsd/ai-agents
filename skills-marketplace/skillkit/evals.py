"""Runs evals.yaml files through ADK's evaluator against an ADK agent.

evals.yaml (all keys except `prompt` optional):

    runs: 1
    fixtures: fixtures                        # dir of recorded HTTP responses (default: ./fixtures)
    criteria: {response_match_score: 0.5}     # overrides/extends defaults
    cases:
      - prompt: How many business days are in the first week of 2026?
        tools: [business_days]                # expected calls, in order; args ignored
        response: There are 4 business days.  # judged for meaning, not wording

Usage (one eval file per process, because the agent reads its env at import):

    python -m skillkit.evals <plugin/skill>          # a skill's evals.yaml
    python -m skillkit.evals --file evals/agent.yaml  # any evals file (agent-level flows)

Env: SKILLKIT_ROOT (skills root, default cwd), SKILLKIT_AGENT_MODULE (default marketplace_agent).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

from . import catalog

JUDGE = {"judge_model": os.getenv("EVAL_JUDGE_MODEL", "gemini-flash-latest"), "num_samples": 1}


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text()) or {}
    return {"cases": spec} if isinstance(spec, list) else spec


def default_criteria(cases: list[dict]) -> dict:
    criteria = {}
    if any("tools" in c for c in cases):
        criteria["tool_trajectory_avg_score"] = {"threshold": 1.0, "match_type": "IN_ORDER", "ignore_args": True}
    if any("response" in c for c in cases):
        criteria["final_response_match_v2"] = {"threshold": 0.5, "judge_model_options": JUDGE}
    return criteria


def to_eval_set(eval_set_id: str, cases: list[dict]):
    from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation
    from google.adk.evaluation.eval_set import EvalSet
    from google.genai import types

    def content(text, role):
        return types.Content(role=role, parts=[types.Part(text=text)]) if text else None

    return EvalSet(
        eval_set_id=eval_set_id,
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


def run(eval_file: Path, eval_set_id: str) -> None:
    from google.adk.evaluation.agent_evaluator import AgentEvaluator
    from google.adk.evaluation.eval_config import EvalConfig

    spec = load_spec(eval_file)
    cases = spec["cases"]
    asyncio.run(
        AgentEvaluator.evaluate_eval_set(
            agent_module=os.getenv("SKILLKIT_AGENT_MODULE", "marketplace_agent"),
            eval_set=to_eval_set(eval_set_id, cases),
            eval_config=EvalConfig(criteria={**default_criteria(cases), **spec.get("criteria", {})}),
            num_runs=spec.get("runs", 1),
        )
    )


def main(argv: list[str]) -> None:
    root = Path(os.getenv("SKILLKIT_ROOT", Path.cwd()))
    if argv[0] == "--file":
        eval_file = Path(argv[1]).resolve()
        eval_set_id = eval_file.stem.replace(".", "_")
    else:
        (skill,) = catalog.skills(root, {argv[0]})
        os.environ["SKILLKIT_SKILLS"] = skill.id  # isolate, unless the agent pins its own selection
        eval_file, eval_set_id = skill.evals_file, skill.id.replace("/", "__")
    fixtures = eval_file.parent / load_spec(eval_file).get("fixtures", "fixtures")
    if fixtures.is_dir():
        os.environ["SKILLKIT_HTTP_FIXTURES"] = str(fixtures)
    sys.path.insert(0, str(Path.cwd()))
    run(eval_file, eval_set_id)


if __name__ == "__main__":
    main(sys.argv[1:])
