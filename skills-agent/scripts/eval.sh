#!/usr/bin/env bash
# Runs the bundle's evals for this agent's plugins + evals/agent.yaml against support_agent.
# Needs GOOGLE_API_KEY (or Vertex env) and fetched skills.
set -euo pipefail
cd "$(dirname "$0")/.."
SKILLS="${SKILLS_DIR:-support_agent/_skills}"
export SKILLKIT_ROOT="$SKILLS" SKILLKIT_AGENT_MODULE=support_agent PYTHONPATH="$PWD:$SKILLS"
PLUGINS=$(python -c 'import support_agent.agent as a; print(" ".join(sorted(a.PLUGINS)))')
exec python -m skillkit.cli eval $PLUGINS --file evals/agent.yaml "$@"
