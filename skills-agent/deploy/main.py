"""Bedrock AgentCore Runtime entrypoint for support_agent.

AgentCore calls POST /invocations with the caller's JSON payload and a session id header.
Each runtime session runs in its own microVM, so ADK's in-memory session service keeps
multi-turn context for as long as AgentCore keeps that session alive.

Payload: {"prompt": "...", "user_id": "optional"}
Stream:  {"type": "tool_call", "name": ...} / {"type": "tool_result", "name": ...} / {"type": "text", "text": ...}
"""

import aws_bootstrap

aws_bootstrap.load_secrets()

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from support_agent.agent import root_agent  # noqa: E402

APP_NAME = "support_agent"
app = BedrockAgentCoreApp()
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


@app.entrypoint
async def invoke(payload, context):
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        yield {"type": "error", "error": 'payload must be {"prompt": "<text>"}'}
        return
    user_id = str(payload.get("user_id", "user"))
    session_id = context.session_id or "local"
    sessions = runner.session_service
    if not await sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id):
        await sessions.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        for part in (event.content.parts if event.content else None) or []:
            if part.function_call:
                yield {"type": "tool_call", "name": part.function_call.name}
            elif part.function_response:
                yield {"type": "tool_result", "name": part.function_response.name}
            elif part.text and not part.thought:
                yield {"type": "text", "text": part.text}


if __name__ == "__main__":
    app.run()
