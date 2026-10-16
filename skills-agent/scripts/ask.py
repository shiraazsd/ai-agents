"""Interactive chat (REPL) with the deployed AgentCore runtime.

  python scripts/ask.py                 # chat; follow-ups share a session
  python scripts/ask.py "question"      # one-shot, for scripts

In the REPL: /new (fresh session) · /tools (show/hide tool calls) · /session · /help · /exit (or Ctrl-D).
Needs AWS credentials allowed to call bedrock-agentcore:InvokeAgentRuntime (works as-is in CloudShell).
The runtime ARN is read from the support-agent stack, or pass --arn.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
DIM, BOLD, RED, CYAN, RESET = ("\033[2m", "\033[1m", "\033[31m", "\033[36m", "\033[0m") if COLOR else ("",) * 5
CLEAR = "\r\033[K" if COLOR else ""
HISTORY = Path.home() / ".support_agent_history"
WRAPPER_TAGS = re.compile(r"</?(response|answer)>")  # small models sometimes wrap answers in these
HINTS = {
    "ExpiredTokenException": "sign in again / refresh your credentials",
    "AccessDeniedException": "your identity can't invoke this runtime, or the budget block is active",
    "ThrottlingException": "slow down and retry",
}


def runtime_arn(stack: str) -> str:
    outputs = boto3.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return next(o["OutputValue"] for o in outputs if o["OutputKey"] == "AgentRuntimeArn")


def new_session() -> str:
    return f"{uuid.uuid4()}-{uuid.uuid4()}"  # AgentCore session ids are >= 33 chars


class Chat:
    def __init__(self, arn: str, show_tools: bool):
        self.client, self.arn = boto3.client("bedrock-agentcore"), arn
        self.session, self.show_tools = new_session(), show_tools

    def ask(self, prompt: str) -> None:
        started = time.time()
        print(f"{DIM}thinking…{RESET}", end="", flush=True) if COLOR else None
        try:
            resp = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.arn, runtimeSessionId=self.session, payload=json.dumps({"prompt": prompt}).encode()
            )
            for line in resp["response"].iter_lines():
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                kind = event.get("type")
                text = WRAPPER_TAGS.sub("", event.get("text", "")).strip() if kind == "text" else ""
                if text:
                    print(f"{CLEAR}{text}")
                elif kind == "tool_call" and self.show_tools:
                    print(f"{CLEAR}{DIM}  → {event['name']}{RESET}")
                elif "error" in event:
                    print(f"{CLEAR}{RED}agent error: {event['error']}{RESET}")
        except ClientError as e:
            err = e.response.get("Error", {})
            hint = HINTS.get(err.get("Code"))
            print(f"{CLEAR}{RED}{err.get('Code')}: {err.get('Message')}{f' ({hint})' if hint else ''}{RESET}")
        except BotoCoreError as e:
            print(f"{CLEAR}{RED}{e}{RESET}")
        print(f"{CLEAR}{DIM}({time.time() - started:.1f}s){RESET}")

    def repl(self) -> None:
        try:
            import readline  # arrow keys + history where available

            if HISTORY.exists():
                readline.read_history_file(HISTORY)
        except (ImportError, OSError):
            readline = None
        print(f"{BOLD}Support agent{RESET} {DIM}— /help for commands, Ctrl-D to quit{RESET}")
        try:
            while True:
                try:
                    line = input(f"{CYAN}you>{RESET} ").strip()
                except KeyboardInterrupt:
                    print()
                    continue
                if not line:
                    continue
                if line.startswith("/") or line in ("exit", "quit"):
                    if not self.command(line):
                        break
                    continue
                self.ask(line)
        except EOFError:
            print()
        finally:
            if readline:
                try:
                    readline.write_history_file(HISTORY)
                except OSError:
                    pass

    def command(self, line: str) -> bool:
        cmd = (line.lstrip("/").split() or ["help"])[0].lower()
        if cmd in ("exit", "quit", "q"):
            return False
        if cmd == "new":
            self.session = new_session()
            print(f"{DIM}new session: the agent has forgotten this conversation{RESET}")
        elif cmd == "tools":
            self.show_tools = not self.show_tools
            print(f"{DIM}tool calls {'shown' if self.show_tools else 'hidden'}{RESET}")
        elif cmd == "session":
            print(f"{DIM}{self.session}{RESET}")
        else:
            print(f"{DIM}/new      start a fresh conversation\n/tools    show/hide tool calls\n/session  print the session id\n/exit     quit (or Ctrl-D){RESET}")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="ask one question and exit")
    ap.add_argument("--arn")
    ap.add_argument("--stack", default="support-agent")
    ap.add_argument("--quiet", action="store_true", help="hide tool calls")
    a = ap.parse_args()
    try:
        chat = Chat(a.arn or runtime_arn(a.stack), show_tools=not a.quiet)
    except (ClientError, BotoCoreError, StopIteration) as e:
        sys.exit(f"can't find the runtime (stack {a.stack!r}); pass --arn or check your AWS region/credentials: {e}")
    if a.prompt:
        chat.ask(a.prompt)
    else:
        chat.repl()


if __name__ == "__main__":
    main()
