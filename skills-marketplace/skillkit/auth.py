"""Pluggable outbound auth for skill tools.

Skill code only ever does:

    from skillkit import session
    r = session(audience="https://svc.internal").get("https://svc.internal/api")

and the provider is picked by the runtime, not the skill:

    SKILLKIT_AUTH=kerberos  SPNEGO with the current ticket cache (developer `kinit`),
                            or a keytab via KRB5_CLIENT_KTNAME (service account).
    SKILLKIT_AUTH=oidc      Google-signed ID token for `audience` (Cloud Run / Agent Engine / GKE).
    SKILLKIT_AUTH=none      No auth.
    SKILLKIT_AUTH=auto      (default) oidc on Cloud Run, kerberos elsewhere.

SKILLKIT_HTTP_FIXTURES=<dir> replaces the network with recorded responses (used by evals).
Register more providers with @provider("name").
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

_PROVIDERS: dict[str, Callable[[requests.Session, Optional[str]], None]] = {}


def provider(name: str):
    def register(fn):
        _PROVIDERS[name] = fn
        return fn

    return register


@provider("none")
def _none(s: requests.Session, audience: Optional[str]) -> None:
    pass


@provider("kerberos")
def _kerberos(s: requests.Session, audience: Optional[str]) -> None:
    try:
        from requests_gssapi import OPTIONAL, HTTPSPNEGOAuth
    except ImportError as e:
        raise RuntimeError("Kerberos auth needs: pip install 'skillkit[kerberos]'") from e
    s.auth = HTTPSPNEGOAuth(mutual_authentication=OPTIONAL)


class _OidcAuth(requests.auth.AuthBase):
    def __init__(self, audience: str):
        self.audience = audience

    def __call__(self, r):
        import google.auth.transport.requests
        from google.oauth2 import id_token

        token = id_token.fetch_id_token(google.auth.transport.requests.Request(), self.audience)
        r.headers["Authorization"] = f"Bearer {token}"
        return r


@provider("oidc")
def _oidc(s: requests.Session, audience: Optional[str]) -> None:
    if not audience:
        raise ValueError("oidc auth needs an audience, e.g. session(audience='https://svc')")
    s.auth = _OidcAuth(audience)


def _auto() -> str:
    return "oidc" if os.getenv("K_SERVICE") else "kerberos"  # K_SERVICE is set by Cloud Run


class _FixtureAdapter(requests.adapters.BaseAdapter):
    """Serves <dir>/<host>/<path>.json; 404 when missing."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def send(self, request, **kwargs):
        u = urlparse(request.url)
        f = self.root / u.hostname / ((u.path.strip("/") or "index") + ".json")
        resp = requests.Response()
        resp.url, resp.request = request.url, request
        resp.status_code = 200 if f.is_file() else 404
        resp._content = f.read_bytes() if f.is_file() else json.dumps({"error": f"no fixture {f}"}).encode()
        resp.headers["Content-Type"] = "application/json"
        return resp

    def close(self):
        pass


def session(audience: Optional[str] = None, mode: Optional[str] = None) -> requests.Session:
    """A requests.Session authenticated for the current runtime."""
    s = requests.Session()
    if fixtures := os.getenv("SKILLKIT_HTTP_FIXTURES"):
        adapter = _FixtureAdapter(Path(fixtures))
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s
    mode = mode or os.getenv("SKILLKIT_AUTH", "auto")
    if mode == "auto":
        mode = _auto()
    if mode not in _PROVIDERS:
        raise ValueError(f"unknown SKILLKIT_AUTH={mode!r}; known: {sorted(_PROVIDERS)}")
    _PROVIDERS[mode](s, audience)
    return s
