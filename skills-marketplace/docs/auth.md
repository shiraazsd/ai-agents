# Auth

Skills never handle credentials. They call `skillkit.session(audience=...)` and the
runtime decides how to authenticate (`SKILLKIT_AUTH`, default `auto`):

| Where the tool runs | Mode | How |
|---|---|---|
| Developer laptop (Claude Code, Copilot) | `kerberos` | SPNEGO from your ticket cache — just `kinit`. Needs `skillkit[kerberos]`. |
| ADK on Cloud Run / Agent Engine / GKE | `oidc` (auto on Cloud Run; set explicitly elsewhere) | Google-signed ID token for `audience`, minted from the runtime service account. Backend validates it (IAP, Cloud Run IAM, or your own JWT check). |
| ADK, but backend is Kerberos-only | `kerberos` + keytab | Mount a service-principal keytab from Secret Manager, set `KRB5_CLIENT_KTNAME=/secrets/svc.keytab` and `KRB5_CONFIG`; GSSAPI acquires tickets itself. Needs network reach to the KDC (VPC connector / on-prem runtime). |
| Evals / CI | fixtures | `fixtures/<host>/<path>.json` served in place of the network. No credentials in CI. |

Set it explicitly in deploy with repo variable `SKILLKIT_AUTH` (default `oidc`).

## Adding a provider

```python
from skillkit.auth import provider

@provider("vault-token")
def _vault(s, audience):
    s.headers["X-Vault-Token"] = read_token()
```

Then `SKILLKIT_AUTH=vault-token`. Put org-wide providers in `skillkit/auth.py` (platform-owned).

## Per-user (delegated) auth in ADK

The modes above are *service* identity. If a tool must act as the end user, use ADK's
built-in tool auth (`AuthConfig` / OAuth2 flows) for that tool — it prompts the user for consent
through the ADK client. Add it as a skillkit provider once you have a concrete backend.
