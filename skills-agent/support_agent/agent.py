"""Support agent: serves a pinned selection of marketplace skills.

Skills (and the skillkit runtime that ships with them) come from the bundle
pinned in skills.lock, fetched into ./_skills by `python scripts/skills.py fetch`.
Set SKILLS_DIR=../skills-marketplace to develop against a local checkout instead.
"""

import os
import sys
from pathlib import Path

PLUGINS = {"common", "platform"}  # which marketplace plugins this agent serves

SKILLS_ROOT = Path(os.getenv("SKILLS_DIR") or Path(__file__).resolve().parent / "_skills").resolve()
if not (SKILLS_ROOT / "skillkit").is_dir():
    raise RuntimeError(f"no skills at {SKILLS_ROOT}: run `python scripts/skills.py fetch` (or set SKILLS_DIR)")
sys.path.insert(0, str(SKILLS_ROOT))

from google.adk.agents import Agent  # noqa: E402

from skillkit.adk import model, toolset  # noqa: E402

root_agent = Agent(
    name="support_agent",
    model=model(),  # AGENT_MODEL: gemini-* (native) or e.g. bedrock/<id> via LiteLLM
    instruction=(
        "You are the internal support assistant. Load the matching skill before acting, "
        "follow its instructions, and use its tools rather than guessing values."
    ),
    tools=[toolset(SKILLS_ROOT, only=PLUGINS)],
)
