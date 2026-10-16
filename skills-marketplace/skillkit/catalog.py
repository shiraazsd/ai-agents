"""Discovers plugins (common/, teams/<team>/) and their skills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PLUGIN_GLOBS = ("common", "teams/*")


@dataclass(frozen=True)
class Skill:
    plugin: str
    name: str
    path: Path
    meta: dict
    body: str

    @property
    def id(self) -> str:
        return f"{self.plugin}/{self.name}"

    @property
    def tools_file(self) -> Path:
        return self.path / "scripts" / "tools.py"

    @property
    def evals_file(self) -> Path:
        return self.path / "evals.yaml"

    @property
    def adk_enabled(self) -> bool:
        """Opt-in: only skills with `metadata: {adk: true}` are served by ADK agents (and evaluated/released)."""
        return str((self.meta.get("metadata") or {}).get("adk", "")).lower() == "true"


def parse_skill_md(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm) or {}, body.strip()


def plugins(root: Path) -> list[Path]:
    found = [p for g in PLUGIN_GLOBS for p in sorted(root.glob(g)) if (p / ".claude-plugin" / "plugin.json").is_file()]
    return found


def plugin_manifest(plugin_dir: Path) -> dict:
    return json.loads((plugin_dir / ".claude-plugin" / "plugin.json").read_text())


def skills(root: Path, only: set[str] | None = None) -> list[Skill]:
    """All skills; `only` filters by skill id ("plugin/skill") or plugin name."""
    out = []
    for plugin_dir in plugins(root):
        plugin = plugin_manifest(plugin_dir)["name"]
        for md in sorted(plugin_dir.glob("skills/*/SKILL.md")):
            meta, body = parse_skill_md(md)
            s = Skill(plugin, md.parent.name, md.parent, meta, body)
            if not only or s.id in only or plugin in only:
                out.append(s)
    return out
