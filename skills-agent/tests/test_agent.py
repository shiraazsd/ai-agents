"""Offline: bundle fetch/verify, and the agent wiring driven by a scripted model (no API key)."""

import asyncio
import hashlib
import io
import tarfile

import pytest
import yaml

import scripts.skills as skills_cli


def _tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return buf.getvalue()


def _use_lock(tmp_path, monkeypatch, sha):
    lock = tmp_path / "skills.lock"
    lock.write_text(yaml.safe_dump({"repo": "o/r", "version": "v9", "sha256": sha}))
    monkeypatch.setattr(skills_cli, "LOCK", lock)
    monkeypatch.setattr(skills_cli, "DEST", tmp_path / "_skills")


def test_fetch_verifies_checksum(tmp_path, monkeypatch):
    data = _tarball({"manifest.json": "{}"})
    (tmp_path / "b.tgz").write_bytes(data)
    args = type("A", (), {"tarball": str(tmp_path / "b.tgz")})

    _use_lock(tmp_path, monkeypatch, "0" * 64)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        skills_cli.cmd_fetch(args)

    _use_lock(tmp_path, monkeypatch, hashlib.sha256(data).hexdigest())
    skills_cli.cmd_fetch(args)
    assert (tmp_path / "_skills" / "manifest.json").is_file()


def test_agent_runs_skill_tools(monkeypatch):
    pytest.importorskip("google.adk")
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    steps = [
        types.Part(function_call=types.FunctionCall(name="load_skill", args={"skill_name": "service-status"})),
        types.Part(function_call=types.FunctionCall(name="service_status", args={"service": "payments-api"})),
        types.Part(text="payments-api is degraded."),
    ]

    class Scripted(BaseLlm):
        model: str = "scripted"
        offered: list = []

        async def generate_content_async(self, req, stream=False):
            self.offered.append(set(req.tools_dict))
            yield LlmResponse(content=types.Content(role="model", parts=[steps[len(self.offered) - 1]]))

    monkeypatch.setenv("SKILLKIT_HTTP_FIXTURES", "evals/fixtures")
    from support_agent import agent

    llm = Scripted()
    agent.root_agent.model = llm

    async def run():
        r = InMemoryRunner(agent=agent.root_agent, app_name="t")
        s = await r.session_service.create_session(app_name="t", user_id="u")
        msg = types.Content(role="user", parts=[types.Part(text="Is payments-api up?")])
        return [p for e in [e async for e in r.run_async(user_id="u", session_id=s.id, new_message=msg)] if e.content for p in e.content.parts]

    parts = asyncio.run(run())
    result = next(p.function_response.response for p in parts if p.function_response and p.function_response.name == "service_status")
    assert result["status"] == "degraded"
    assert "service_status" not in llm.offered[0] and "service_status" in llm.offered[1]
