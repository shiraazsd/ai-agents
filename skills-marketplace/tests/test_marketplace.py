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
    assert set(ts._skills) == {"business-days", "service-status"}  # commit-messages is dev-only
    assert {"business_days", "add_business_days", "service_status"} <= set(ts._provided_tools_by_name)


def test_adk_is_opt_in(tmp_path):
    adk_ids = {s.id for s in catalog.skills(ROOT) if s.adk_enabled}
    assert adk_ids == {"common/business-days", "platform/service-status"}
    assert "common/commit-messages" in {s.id for s in catalog.skills(ROOT)}  # still a Claude Code / Copilot skill


def test_dev_only_skills_never_ship_to_adk(tmp_path, monkeypatch):
    import tarfile

    from skillkit import cli

    cli.cmd_pack(type("A", (), {"version": "v0.0.0", "out": str(tmp_path)}))
    names = tarfile.open(tmp_path / "skills-v0.0.0.tar.gz").getnames()
    assert any(n.startswith("common/skills/business-days") for n in names)
    assert not any("commit-messages" in n for n in names)

    cli.cmd_bundle(type("A", (), {"out": str(tmp_path / "dist")}))
    assert not (tmp_path / "dist/marketplace_agent/common/skills/commit-messages").exists()
