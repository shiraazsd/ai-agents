"""Expose marketplace skills to a Google ADK agent via ADK's native SkillToolset.

Each SKILL.md is loaded as an ADK Skill (progressive disclosure: list -> load ->
resources). Functions in a skill's scripts/tools.py become FunctionTools that are
only declared to the model once that skill is loaded.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from google.adk.skills import load_skill_from_dir
from google.adk.tools import FunctionTool
from google.adk.tools.skill_toolset import SkillDiscoveryMode, SkillToolset

from . import catalog
from .tools import public_functions


def _load_module(skill: catalog.Skill):
    spec = importlib.util.spec_from_file_location(f"skills.{skill.plugin}.{skill.name}".replace("-", "_"), skill.tools_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toolset(root: Path, only: set[str] | None = None) -> SkillToolset:
    """Build a SkillToolset from every ADK-enabled skill under `root`.

    SKILLKIT_SKILLS="common,payments/refund-lookup" narrows the set (used by evals).
    """
    if only is None and os.getenv("SKILLKIT_SKILLS"):
        only = set(os.environ["SKILLKIT_SKILLS"].split(","))
    adk_skills, tools = [], []
    for s in catalog.skills(root, only):
        if not s.adk_enabled:
            continue
        skill = load_skill_from_dir(s.path)
        if s.tools_file.is_file():
            fns = [FunctionTool(fn) for fn in public_functions(_load_module(s))]
            skill.frontmatter.metadata["adk_additional_tools"] = [t.name for t in fns]
            tools += fns
        adk_skills.append(skill)
    return SkillToolset(adk_skills, additional_tools=tools, discovery_mode=SkillDiscoveryMode.EAGER)
