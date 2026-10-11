"""Drives the real ADK agent with a scripted model (no API key): load_skill -> tool -> answer."""

import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

STEPS = [
    types.Part(function_call=types.FunctionCall(name="load_skill", args={"skill_name": "service-status"})),
    types.Part(function_call=types.FunctionCall(name="service_status", args={"service": "payments-api"})),
    types.Part(text="payments-api is degraded."),
]


class Scripted(BaseLlm):
    model: str = "scripted"
    offered: list = []

    async def generate_content_async(self, req, stream=False):
        self.offered.append(set(req.tools_dict))
        yield LlmResponse(content=types.Content(role="model", parts=[STEPS[len(self.offered) - 1]]))


def test_skill_tool_runs_after_load(monkeypatch):
    monkeypatch.setenv("SKILLKIT_HTTP_FIXTURES", "teams/platform/skills/service-status/fixtures")
    from marketplace_agent import agent

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
    assert "service_status" not in llm.offered[0]  # hidden until the skill is loaded
    assert "service_status" in llm.offered[1]
