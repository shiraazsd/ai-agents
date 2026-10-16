"""skillkit: shared runtime for marketplace skills.

Skill tools import from here so the same code runs under Claude Code / Copilot
(as a CLI script) and inside the Google ADK agent (as a FunctionTool).
"""

from .auth import session
from .tools import run

__all__ = ["session", "run"]
