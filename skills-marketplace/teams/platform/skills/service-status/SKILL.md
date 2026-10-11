---
name: service-status
description: Look up the live health, version and on-call owner of an internal service. Use when someone asks whether a service is up, degraded, or who owns it.
metadata:
  adk: true
---

# Service status

Call `service_status(service)` with the service's short name (e.g. `payments-api`).
Report status, version and on-call in one or two sentences; if degraded, include the incident link.

Auth is handled by skillkit: Kerberos on your laptop (`kinit` first), OIDC when running in ADK.

From a shell: `python scripts/tools.py service_status --service payments-api`
