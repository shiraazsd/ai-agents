# Skills Marketplace

One folder of [Agent Skills](https://agentskills.io) (`SKILL.md` + optional Python tools), usable from
**Claude Code**, **GitHub Copilot**, and a deployed **Google ADK** agent, with evals gating every PR.

```
common/                      # org-wide skills          (owned by platform, see CODEOWNERS)
teams/<team>/                # one plugin per team      (owned by that team)
  .claude-plugin/plugin.json
  skills/<skill>/
    SKILL.md                 # frontmatter (name, description) + instructions
    scripts/tools.py         # optional: plain typed functions = tools
    evals.yaml               # required: prompts + expected tools/answer
    fixtures/                # optional: recorded HTTP responses for offline evals
skillkit/                    # shared runtime: auth, tool CLI, ADK loader, `mkt` CLI
marketplace_agent/           # the ADK agent (serves every skill)
```

## Use the skills

```bash
pip install "skillkit[kerberos] @ git+https://github.com/shiraazsd/skills-marketplace"   # once; tools import it
```

| Assistant | Install |
|---|---|
| Claude Code | `/plugin marketplace add shiraazsd/skills-marketplace` then `/plugin install common@skills-marketplace` |
| Copilot / any SKILL.md client | clone, then `mkt install copilot` (→ `~/.copilot/skills`) or `mkt install copilot --project .` (→ `.github/skills`) |
| Claude Code without plugins | `mkt install claude [--only common,platform]` |
| ADK (local) | `GOOGLE_API_KEY=… adk web .` and pick `marketplace_agent` |

## Contribute

```bash
pip install -e ".[dev,kerberos]"
mkt new-team payments --owner @my-org/payments   # once per team (adds CODEOWNERS line)
mkt new-skill payments refund-lookup             # scaffold SKILL.md, tools.py, evals.yaml
mkt validate && pytest -q                        # lint + offline tests
mkt eval payments/refund-lookup                  # real ADK eval (needs GOOGLE_API_KEY)
```

Rules (enforced by `mkt validate`): folder name = skill `name`; description says *when* to use it;
skill and tool names are unique across the marketplace; every plugin has a CODEOWNERS owner;
`marketplace.json` is regenerated with `mkt sync`.

A tool is just a function. The same file works as an ADK tool and as a shell command:

```python
from skillkit import run, session

def service_status(service: str) -> dict:
    """Get health and on-call for an internal service."""
    return session(audience=API).get(f"{API}/services/{service}").json()

if __name__ == "__main__":
    run()          # python scripts/tools.py service_status --service payments-api
```

Set `metadata: {adk: false}` in the frontmatter for dev-only skills (e.g. coding conventions).

## Pipeline

| When | What |
|---|---|
| PR | `validate` (lint, CODEOWNERS coverage, offline tests) → `evals` for changed skills (all skills if `skillkit/` or the agent changed) |
| Merge to `main` | Claude Code/Copilot users get it on next `/plugin marketplace update` / `git pull`; `release` publishes `skills-v0.<n>.0` (tarball + sha256) for agent repos; `deploy` ships the in-repo agent (only if configured) |

Evals use ADK's evaluator: tool calls must happen in order (args ignored), answers are LLM-judged for meaning.
Override per skill with `criteria:` in `evals.yaml` (any [ADK metric](https://google.github.io/adk-docs/evaluate/)).

## Setup checklist (repo admin)

1. Branch protection on `main`: require PR, **require Code Owner review**, require `validate` + `evals` checks.
2. Replace `@shiraazsd` in `.github/CODEOWNERS` with real teams.
3. Secret `GOOGLE_API_KEY` (free: <https://aistudio.google.com/apikey>) for evals.
4. Deploy (optional): repo vars `GCP_PROJECT`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_RUNTIME_SA`
   ([Workload Identity Federation](https://github.com/google-github-actions/auth#workload-identity-federation-through-a-service-account)). The job is skipped until `GCP_PROJECT` is set.

## Agents in other repos

`marketplace_agent/` here is the reference agent skill evals run against. Production agents can live
in their own repos (template: `skills-agent`): each pins a bundle in `skills.lock`, picks the plugins it
serves, re-runs those skills' evals against itself, and upgrades via an automated PR. In that setup,
don't set the `GCP_*` variables here; each agent repo owns its own deploy.

```bash
mkt pack v0.7.0                         # what the release workflow publishes
mkt eval common --file my/agent.yaml    # plugin names and extra eval files are accepted
```

Auth details: [docs/auth.md](docs/auth.md).
