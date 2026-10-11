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
deploy/                   # Bedrock AgentCore Runtime: Dockerfile (arm64), main.py entrypoint,
                          #   agentcore.yaml (runtime), bootstrap.yaml (one-time: ECR + GitHub deploy role)
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
4. Merge → `deploy` builds the arm64 image, pushes it to ECR and updates the AgentCore runtime.
   Roll back by reverting the lock bump (or redeploying an earlier image tag).

## Deploy (Amazon Bedrock AgentCore Runtime)

One-time, in the target AWS account (AgentCore must be available in the region):

```bash
aws cloudformation deploy --template-file deploy/bootstrap.yaml --stack-name support-agent-bootstrap \
  --capabilities CAPABILITY_NAMED_IAM --parameter-overrides GitHubRepo=<owner>/skills-agent
aws ssm put-parameter --name /support-agent/google-api-key --type SecureString --value '<gemini key>'
```

Set repo variables from the bootstrap outputs: `AWS_DEPLOY_ROLE_ARN`, `ECR_REPOSITORY_URI`, `AWS_REGION`
(optional: `AGENT_MODEL`, `SKILLKIT_AUTH`). Every merge to `main` then deploys.

Call it (IAM/SigV4 by default; session ids must be ≥ 33 chars and keep conversation context):

```bash
aws bedrock-agentcore invoke-agent-runtime --agent-runtime-arn <AgentRuntimeArn output> \
  --runtime-session-id "$(uuidgen)-$(uuidgen)" --payload '{"prompt": "Is payments-api up?"}' out.json
```

Options in `deploy/agentcore.yaml` (set once with `aws cloudformation deploy … --parameter-overrides`; later deploys keep them):

| Need | Parameter |
|---|---|
| Claude etc. on Bedrock instead of Gemini | `AgentModel=bedrock/<model id>` and `GoogleApiKeyParam=""` |
| Users call with SSO (Entra ID / Okta / Cognito) tokens | `JwtDiscoveryUrl`, `JwtAllowedAudience` |
| Skills reach internal networks / Kerberos KDC | `SubnetIds`, `SecurityGroupIds` (VPC mode) |
| Kerberos to backends | `SkillkitAuth=kerberos`, `KeytabSecretArn`, `Krb5ConfParam`, and `KERBEROS=true` build arg |
| Cost | `IdleSessionTimeout` (default 300 s): memory is billed while a session is alive |

The container follows the AgentCore Runtime contract (`POST /invocations`, `GET /ping`, port 8080,
linux/arm64) via the `bedrock-agentcore` SDK; each runtime session gets its own microVM, so ADK's
in-memory sessions hold multi-turn context without a database.

## Setup checklist

- Secret `SKILLS_TOKEN`: fine-grained token (or GitHub App token) with *contents: read* on the marketplace
  and *contents + pull requests: write* here. Used to fetch bundles and to open bump PRs that trigger CI.
- Secret `GOOGLE_API_KEY` for evals (free: <https://aistudio.google.com/apikey>).
- First pin: run the `bump-skills` workflow (or `python scripts/skills.py bump`) once the marketplace has a release.
- Deploy vars: `AWS_DEPLOY_ROLE_ARN`, `ECR_REPOSITORY_URI`, `AWS_REGION` (see Deploy above).
- Branch protection: require `test` + `evals`.

Tool auth (Kerberos locally; SigV4, or Kerberos via keytab, on AgentCore) is handled by skillkit; see the marketplace's `docs/auth.md`.
