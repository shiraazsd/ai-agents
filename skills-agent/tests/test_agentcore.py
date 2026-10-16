"""AgentCore Runtime contract, offline: /ping, /invocations (SSE) with a session id, scripted model."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("bedrock_agentcore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


CALLS = []


class Scripted(BaseLlm):
    """Turn 1: load the skill, turn 2: call its tool, turn 3: answer; later turns echo the turn count."""

    model: str = "scripted"

    async def generate_content_async(self, req, stream=False):
        CALLS.append(1)
        steps = [
            types.Part(function_call=types.FunctionCall(name="load_skill", args={"skill_name": "service-status"})),
            types.Part(function_call=types.FunctionCall(name="service_status", args={"service": "payments-api"})),
            types.Part(text="payments-api is degraded."),
        ]
        user_turns = sum(1 for c in req.contents if c.role == "user" and any(p.text for p in c.parts or []))
        part = steps[len(CALLS) - 1] if len(CALLS) <= 3 else types.Part(text=f"turn {user_turns}")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _events(resp):
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SKILLKIT_HTTP_FIXTURES", "evals/fixtures")
    import main

    main.root_agent.model = Scripted()
    CALLS.clear()
    return TestClient(main.app)


def test_ping(client):
    assert client.get("/ping").json()["status"] == "Healthy"


def test_invocation_streams_tool_use_and_answer(client):
    sid = "s" * 40  # AgentCore session ids are >= 33 chars
    headers = {"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sid}
    events = _events(client.post("/invocations", json={"prompt": "Is payments-api up?"}, headers=headers))
    assert [e for e in events if e["type"] != "text"] == [
        {"type": "tool_call", "name": "load_skill"},
        {"type": "tool_result", "name": "load_skill"},
        {"type": "tool_call", "name": "service_status"},
        {"type": "tool_result", "name": "service_status"},
    ]
    assert events[-1] == {"type": "text", "text": "payments-api is degraded."}

    # Same session id -> same ADK session: the model now sees two user turns.
    follow_up = _events(client.post("/invocations", json={"prompt": "And now?"}, headers=headers))
    assert follow_up[-1] == {"type": "text", "text": "turn 2"}


def test_bad_payload(client):
    assert _events(client.post("/invocations", json={"nope": 1}))[0]["type"] == "error"
