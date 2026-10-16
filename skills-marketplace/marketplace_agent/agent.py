"""ADK agent that serves every marketplace skill.

Locally: repo root is the skills root. In a deployed bundle (`mkt build`),
skills and skillkit are copied next to this file.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLED = (HERE / "common").is_dir()
ROOT = HERE if BUNDLED else HERE.parent
sys.path.insert(0, str(ROOT))  # makes `skillkit` importable from bundled copies

from google.adk.agents import Agent  # noqa: E402

from skillkit.adk import model, toolset  # noqa: E402

root_agent = Agent(
    name="marketplace_agent",
    model=model(),
    instruction=(
        "You help users by applying the available skills. Load the matching skill before acting, "
        "follow its instructions, and use its tools rather than guessing values."
    ),
    tools=[toolset(Path(os.getenv("SKILLKIT_ROOT", ROOT)))],
)
