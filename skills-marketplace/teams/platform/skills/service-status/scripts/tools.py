"""Internal status API client."""

import os

from skillkit import run, session

STATUS_API = os.getenv("STATUS_API_URL", "https://status.internal.example.com")


def service_status(service: str) -> dict:
    """Get health, version and on-call owner for an internal service by short name."""
    r = session(audience=STATUS_API).get(f"{STATUS_API}/services/{service}", timeout=10)
    if r.status_code == 404:
        return {"service": service, "error": "unknown service"}
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    run()
