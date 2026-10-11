# Support Agent

A Google ADK agent that serves a pinned selection of ADK-qualified skills (`metadata: {adk: true}`) from the
[skills marketplace](https://github.com/shiraazsd/skills-marketplace). Skills are owned and
evaluated there; this repo decides **which** skills, **which version**, and **how** the agent runs.

```
skills.lock               # marketplace release this agent runs (version + sha256)
support_agent/agent.py    # model, instruction, PLUGINS = which marketplace plugins to serve
support_agent/_skills/    # fetched bundle (gitignored): skills + skillkit runtime
evals/agent.yaml          # agent-level evals (flows across skills) + fixtures/
scripts/skills.py         # fetch | bump
scripts/eval.sh           # bundle evals for PLUGINS + agent evals, against this agent
```

## Develop

```bash
pip install -r requirements.txt
python scripts/skills.py fetch                 # needs GH_TOKEN or `gh auth login` (marketplace is private)
pytest -q                                      # offline
GOOGLE_API_KEY=… scripts/eval.sh               # real evals
GOOGLE_API_KEY=… adk web .                     # chat with it

SKILLS_DIR=../skills-marketplace adk web .     # try unreleased skills from a local checkout
```

## How skills reach production

1. A skill PR merges in the marketplace → release `skills-v0.<n>.0` is published.
2. `bump-skills` (every 6h, or run it manually) opens a PR here updating `skills.lock`.
3. CI on that PR runs every pinned skill's evals **with all of this agent's skills loaded**
   (catches routing conflicts) plus `evals/agent.yaml`.
4. Merge → `deploy` ships to Cloud Run. Roll back by reverting the lock bump.

## Setup checklist

- Secret `SKILLS_TOKEN`: fine-grained token (or GitHub App token) with *contents: read* on the marketplace
  and *contents + pull requests: write* here. Used to fetch bundles and to open bump PRs that trigger CI.
- Secret `GOOGLE_API_KEY` for evals (free: <https://aistudio.google.com/apikey>).
- First pin: run the `bump-skills` workflow (or `python scripts/skills.py bump`) once the marketplace has a release.
- Deploy vars: `GCP_PROJECT`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_RUNTIME_SA`, optional `SKILLKIT_AUTH`.
- Branch protection: require `test` + `evals`.

Tool auth (Kerberos locally, OIDC on Cloud Run) is handled by skillkit; see the marketplace's `docs/auth.md`.
