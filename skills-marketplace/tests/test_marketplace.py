"""Offline checks: catalog, CLI wrapper, auth fixtures and ADK wiring. No model calls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from skillkit import catalog, session
from skillkit.cli import problems

ROOT = Path(__file__).resolve().parents[1]


def test_marketplace_is_valid():
    assert problems() == []


def test_tools_run_as_cli():
    script = ROOT / "common/skills/business-days/scripts/tools.py"
    out = subprocess.run([sys.executable, script, "business_days", "--start", "2026-01-05", "--end", "2026-01-19"], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["business_days"] == 10


def test_http_fixtures(monkeypatch):
    skill = ROOT / "teams/platform/skills/service-status"
    monkeypatch.setenv("SKILLKIT_HTTP_FIXTURES", str(skill / "fixtures"))
    r = session().get("https://status.internal.example.com/services/payments-api")
    assert r.json()["status"] == "degraded"
    assert session().get("https://status.internal.example.com/services/nope").status_code == 404


def test_unknown_auth_mode(monkeypatch):
    monkeypatch.setenv("SKILLKIT_AUTH", "magic")
    with pytest.raises(ValueError):
        session()


def test_adk_toolset_loads_every_skill():
    adk = pytest.importorskip("skillkit.adk")
    ts = adk.toolset(ROOT)
    assert set(ts._skills) == {s.name for s in catalog.skills(ROOT) if s.adk_enabled}
    assert {"business_days", "add_business_days", "service_status"} <= set(ts._provided_tools_by_name)
