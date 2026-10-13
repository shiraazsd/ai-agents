"""Chat with the deployed AgentCore runtime.

  python scripts/ask.py "How many business days between 2026-01-05 and 2026-01-19?"
  python scripts/ask.py                      # interactive: one session, so follow-ups keep context

Needs AWS credentials allowed to call bedrock-agentcore:InvokeAgentRuntime (e.g. `aws sso login`).
The runtime ARN is read from the support-agent stack, or pass --arn.
"""

import argparse
import json
import sys
import uuid

import boto3


def runtime_arn(stack: str) -> str:
    outputs = boto3.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return next(o["OutputValue"] for o in outputs if o["OutputKey"] == "AgentRuntimeArn")


def ask(client, arn: str, session: str, prompt: str, verbose: bool) -> None:
    resp = client.invoke_agent_runtime(agentRuntimeArn=arn, runtimeSessionId=session, payload=json.dumps({"prompt": prompt}).encode())
    for line in resp["response"].iter_lines():
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "text":
            print(event["text"], end="", flush=True)
        elif verbose and event.get("type") in ("tool_call", "tool_result"):
            print(f"\n  [{event['type']}: {event['name']}]", flush=True)
        elif "error" in event:
            print(f"\n  [error: {event['error']}]", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--arn")
    ap.add_argument("--stack", default="support-agent")
    ap.add_argument("--quiet", action="store_true", help="hide tool calls")
    a = ap.parse_args()
    arn, client = a.arn or runtime_arn(a.stack), boto3.client("bedrock-agentcore")
    session = f"{uuid.uuid4()}-{uuid.uuid4()}"  # AgentCore session ids are >= 33 chars
    if a.prompt:
        return ask(client, arn, session, a.prompt, not a.quiet)
    print(f"session {session} (Ctrl-D to quit)")
    for prompt in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("you> "), None):
        if prompt.strip():
            ask(client, arn, session, prompt.strip(), not a.quiet)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
