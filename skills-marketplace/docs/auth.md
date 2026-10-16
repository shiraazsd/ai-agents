# Auth

Skills never handle credentials. They call `skillkit.session(audience=...)` and the
runtime decides how to authenticate (`SKILLKIT_AUTH`, default `auto`):

| Where the tool runs | Mode | How |
|---|---|---|
| Developer laptop (Claude Code, Copilot) | `kerberos` | SPNEGO from your ticket cache — just `kinit`. Needs `skillkit[kerberos]`. |
| ADK on Bedrock AgentCore Runtime | `sigv4` (template default) | Requests signed with the runtime's IAM role (`SKILLKIT_SIGV4_SERVICE`, default `execute-api`). Backend: API Gateway with IAM auth, or any SigV4-verifying service. Needs `skillkit[aws]`. |
| AgentCore, backend is Kerberos-only | `kerberos` + keytab | At startup the agent reads a service-principal keytab from Secrets Manager (`KeytabSecretArn`) and `krb5.conf` from SSM (`Krb5ConfParam`), and sets `KRB5_CLIENT_KTNAME` / `KRB5_CONFIG`; GSSAPI acquires tickets itself. Run the runtime in VPC mode (`SubnetIds`) with reach to the KDC and backend, and build the image with `KERBEROS=true`. |
| ADK on Cloud Run / GKE | `oidc` (auto on Cloud Run) | Google-signed ID token for `audience`. Kept for GCP deployments. |
| Evals / CI | fixtures | `fixtures/<host>/<path>.json` served in place of the network. No credentials in CI. |

Set it explicitly per deployment (the agent repo's `SKILLKIT_AUTH` repo variable → AgentCore runtime env).

## Adding a provider

```python
from skillkit.auth import provider

@provider("vault-token")
def _vault(s, audience):
    s.headers["X-Vault-Token"] = read_token()
```

Then `SKILLKIT_AUTH=vault-token`. Put org-wide providers in `skillkit/auth.py` (platform-owned).

## Per-user (delegated) auth

The modes above are *service* identity. If a tool must act as the end user (OAuth to SaaS APIs),
use **AgentCore Identity** credential providers on AWS (the runtime role already allows fetching
workload access tokens), or ADK's built-in tool auth (`AuthConfig` / OAuth2 flows). Add it as a
skillkit provider once you have a concrete backend.
