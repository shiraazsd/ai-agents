"""mkt — marketplace tooling. Run `mkt -h`.

  mkt new-team <team> --owner @org/team    scaffold teams/<team> + CODEOWNERS entry
  mkt new-skill <plugin> <skill> [--adk]   scaffold a skill (--adk: also served by ADK agents)
  mkt validate                             lint everything (CI gate)
  mkt sync                                 regenerate .claude-plugin/marketplace.json
  mkt changed <base-ref>                   skill ids touched since base
  mkt eval [ids...] [--changed BASE] [--file F]   run ADK evals (one process per eval set)
  mkt install claude|copilot [--only ...]  link skills into your local AI assistant
  mkt bundle [out]                         build the deployable ADK agent folder (single-repo mode)
  mkt pack <version>                       versioned skills tarball for agents in other repos
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from . import catalog

ROOT = Path(os.getenv("SKILLKIT_ROOT", Path.cwd()))
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SHARED_PATHS = ("skillkit/", "marketplace_agent/", "pyproject.toml")  # changes here re-eval everything


# ---------------------------------------------------------------- marketplace.json
def marketplace() -> dict:
    current = json.loads(MARKETPLACE.read_text()) if MARKETPLACE.exists() else {}
    entries = []
    for p in catalog.plugins(ROOT):
        m = catalog.plugin_manifest(p)
        entries.append({"name": m["name"], "source": f"./{p.relative_to(ROOT).as_posix()}", "description": m.get("description", "")})
    return {
        "name": current.get("name", ROOT.name),
        "owner": current.get("owner", {"name": "platform-team"}),
        "plugins": entries,
    }


def cmd_sync(_):
    MARKETPLACE.parent.mkdir(exist_ok=True)
    MARKETPLACE.write_text(json.dumps(marketplace(), indent=2) + "\n")
    print(f"wrote {MARKETPLACE.relative_to(ROOT)}")


# ---------------------------------------------------------------- validate
def owned_paths() -> set[str]:
    if not CODEOWNERS.exists():
        return set()
    return {line.split()[0].strip("/") for line in CODEOWNERS.read_text().splitlines() if line.strip() and not line.startswith("#")}


def problems() -> list[str]:
    errs, owners, seen_skills, seen_tools = [], owned_paths(), {}, {}
    for p in catalog.plugins(ROOT):
        rel = p.relative_to(ROOT).as_posix()
        if catalog.plugin_manifest(p).get("name") != p.name:
            errs.append(f"{rel}: plugin.json name must be '{p.name}'")
        if rel not in owners:
            errs.append(f"{rel}: no CODEOWNERS entry (add '/{rel}/ @org/team')")
    for s in catalog.skills(ROOT):
        where = s.path.relative_to(ROOT).as_posix()
        name, desc = s.meta.get("name"), str(s.meta.get("description", ""))
        if name != s.name or not NAME.match(s.name or ""):
            errs.append(f"{where}: frontmatter name must equal folder name and be kebab-case")
        if not 20 <= len(desc) <= 1024:
            errs.append(f"{where}: description must be 20-1024 chars and say *when* to use the skill")
        if not s.body:
            errs.append(f"{where}: SKILL.md body is empty")
        if s.name in seen_skills:
            errs.append(f"{where}: skill name clashes with {seen_skills[s.name]} (names are global in ADK)")
        seen_skills[s.name] = where
        adk_flag = (s.meta.get("metadata") or {}).get("adk")
        if adk_flag is not None and str(adk_flag).lower() not in ("true", "false"):
            errs.append(f"{where}: metadata.adk must be true or false")
        if s.adk_enabled and not s.evals_file.is_file():
            errs.append(f"{where}: ADK skills need evals.yaml")
        if s.adk_enabled and s.tools_file.is_file():  # ADK registers all tools in one namespace
            for tool in re.findall(r"^def ([a-z][a-z0-9_]*)\(", s.tools_file.read_text(), re.M):
                if tool in seen_tools:
                    errs.append(f"{where}: tool '{tool}' clashes with {seen_tools[tool]}")
                seen_tools[tool] = where
    if MARKETPLACE.exists() and json.loads(MARKETPLACE.read_text()) != marketplace():
        errs.append(".claude-plugin/marketplace.json is stale — run `mkt sync`")
    return errs


def cmd_validate(_):
    errs = problems()
    for e in errs:
        print(f"::error::{e}" if os.getenv("GITHUB_ACTIONS") else f"✗ {e}")
    if errs:
        sys.exit(1)
    print(f"✓ {len(catalog.skills(ROOT))} skills in {len(catalog.plugins(ROOT))} plugins look good")


# ---------------------------------------------------------------- changed / eval
def changed(base: str) -> list[str]:
    files = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    all_skills = catalog.skills(ROOT)
    if any(f.startswith(SHARED_PATHS) for f in files):
        return [s.id for s in all_skills]
    return [s.id for s in all_skills if any(f.startswith(s.path.relative_to(ROOT).as_posix() + "/") for f in files)]


def cmd_changed(a):
    print("\n".join(changed(a.base)))


def cmd_eval(a):
    if a.ids:
        wanted = set(a.ids)
        unknown = [i for i in a.ids if not catalog.skills(ROOT, {i})]
        if unknown:
            sys.exit(f"unknown skill or plugin: {', '.join(unknown)}")
        not_adk = [i for i in a.ids if not any(s.adk_enabled for s in catalog.skills(ROOT, {i}))]
        if not_adk:
            sys.exit(f"not ADK skills (add `metadata: {{adk: true}}`): {', '.join(not_adk)}")
    elif a.changed:
        wanted = set(changed(a.changed))
    else:
        wanted = set() if a.file else None  # no args at all = every skill
    skill_ids = [s.id for s in catalog.skills(ROOT, wanted) if s.adk_enabled] if wanted != set() else []
    targets = [[i] for i in skill_ids] + [["--file", f] for f in a.file]
    if not targets:
        return print("nothing to evaluate")
    if not any(os.getenv(v) for v in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")):
        sys.exit("set GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey) or GOOGLE_GENAI_USE_VERTEXAI=1")
    env = {**os.environ, "SKILLKIT_ROOT": str(ROOT), "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.getenv("PYTHONPATH")]))}
    failed = [" ".join(t) for t in targets if subprocess.run([sys.executable, "-m", "skillkit.evals", *t], env=env).returncode]
    print(f"\n{len(targets) - len(failed)}/{len(targets)} eval sets passed")
    if failed:
        sys.exit("failed: " + ", ".join(failed))


# ---------------------------------------------------------------- local install
TARGETS = {"claude": ".claude/skills", "copilot": ".copilot/skills"}
PROJECT_TARGETS = {"claude": ".claude/skills", "copilot": ".github/skills"}


def cmd_install(a):
    base = Path(a.project) / PROJECT_TARGETS[a.target] if a.project else Path.home() / TARGETS[a.target]
    base.mkdir(parents=True, exist_ok=True)
    for s in catalog.skills(ROOT, set(a.only.split(",")) if a.only else None):
        link = base / s.name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            print(f"skip {link} (exists, not ours)")
            continue
        link.symlink_to(s.path.resolve(), target_is_directory=True)
        print(f"linked {s.id} -> {link}")


# ---------------------------------------------------------------- scaffolding
SKILL_MD = """---
name: {name}
description: TODO — what this does and when to use it (the model reads this to decide).
{adk}---

# {title}

TODO: concise instructions. Tools (if any) live in `scripts/tools.py`; from a
shell run `python scripts/tools.py <tool> --arg value`.
"""
TOOLS_PY = '''"""Tools for {name}. Each public function = one tool (type hints + docstring required)."""

from skillkit import run


def example(text: str) -> dict:
    """Echo text back. Replace me."""
    return {{"echo": text}}


if __name__ == "__main__":
    run()
'''
ADK_META = "metadata:\n  adk: true   # served by ADK agents; requires evals.yaml\n"
EVALS_YAML = """cases:
  - prompt: TODO a realistic user request
    tools: [example]
    response: TODO the gist of a good answer
"""


def cmd_new_team(a):
    team = ROOT / "teams" / a.team
    (team / ".claude-plugin").mkdir(parents=True)
    (team / "skills").mkdir()
    (team / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": a.team, "description": f"Skills owned by {a.team}"}, indent=2) + "\n")
    with CODEOWNERS.open("a") as f:
        f.write(f"/teams/{a.team}/ {a.owner}\n")
    cmd_sync(a)
    print(f"created teams/{a.team} owned by {a.owner}")


def cmd_new_skill(a):
    (plugin_dir,) = [p for p in catalog.plugins(ROOT) if p.name == a.plugin]
    d = plugin_dir / "skills" / a.skill
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD.format(name=a.skill, title=a.skill.replace("-", " ").title(), adk=ADK_META if a.adk else ""))
    (d / "scripts" / "tools.py").write_text(TOOLS_PY.format(name=a.skill))
    if a.adk:
        (d / "evals.yaml").write_text(EVALS_YAML)
    print(f"created {d.relative_to(ROOT)}" + (" (ADK-qualified)" if a.adk else " (Claude Code / Copilot only; add --adk to qualify)"))


# ---------------------------------------------------------------- deploy bundle / release
def adk_payload() -> list[Path]:
    """What ADK agents get: plugin manifests + ADK-qualified skill folders (never dev-only skills)."""
    skills = [s for s in catalog.skills(ROOT) if s.adk_enabled]
    plugins = sorted({s.path.parent.parent for s in skills})
    return [p / ".claude-plugin" for p in plugins] + [s.path for s in skills]


def _no_pycache(ti):
    return None if "__pycache__" in ti.name.split("/") else ti


def cmd_bundle(a):
    out = Path(a.out) / "marketplace_agent"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "marketplace_agent", out, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "skillkit", out / "skillkit", ignore=shutil.ignore_patterns("__pycache__"))
    for src in adk_payload():
        shutil.copytree(src, out / src.relative_to(ROOT), ignore=shutil.ignore_patterns("__pycache__", "evals.yaml", "fixtures"))
    (out / "requirements.txt").write_text("google-adk>=2.11\npyyaml>=6\nrequests>=2.31\ngoogle-auth>=2.30\n")
    print(f"bundle ready: {out}  (run locally: adk web {out.parent})")


def cmd_pack(a):
    """Versioned tarball for agents in other repos: skillkit + ADK skills (with evals/fixtures) + manifest."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / f"skills-{a.version}.tar.gz"
    manifest = json.dumps({
        "version": a.version,
        "commit": os.getenv("GITHUB_SHA", ""),
        "skills": [s.id for s in catalog.skills(ROOT) if s.adk_enabled],
    }, indent=2).encode()
    with tarfile.open(tar_path, "w:gz") as tar:
        for src in [ROOT / "skillkit", *adk_payload()]:
            tar.add(src, arcname=src.relative_to(ROOT).as_posix(), filter=_no_pycache)
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mtime = len(manifest), int(__import__("time").time())
        tar.addfile(info, io.BytesIO(manifest))
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / f"{tar_path.name}.sha256").write_text(f"{digest}  {tar_path.name}\n")
    print(f"{tar_path}  sha256={digest}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mkt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    p = sub.add_parser("changed"); p.add_argument("base"); p.set_defaults(fn=cmd_changed)
    p = sub.add_parser("eval"); p.add_argument("ids", nargs="*", help="skill ids or plugin names"); p.add_argument("--changed", metavar="BASE"); p.add_argument("--file", action="append", default=[], help="extra evals.yaml (agent-level)"); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("install"); p.add_argument("target", choices=TARGETS); p.add_argument("--only", help="plugins or plugin/skill ids, comma-separated"); p.add_argument("--project", help="install into this repo instead of your home dir"); p.set_defaults(fn=cmd_install)
    p = sub.add_parser("new-team"); p.add_argument("team"); p.add_argument("--owner", required=True); p.set_defaults(fn=cmd_new_team)
    p = sub.add_parser("new-skill"); p.add_argument("plugin"); p.add_argument("skill"); p.add_argument("--adk", action="store_true", help="qualify for ADK agents (adds metadata + evals.yaml)"); p.set_defaults(fn=cmd_new_skill)
    p = sub.add_parser("bundle"); p.add_argument("out", nargs="?", default="dist"); p.set_defaults(fn=cmd_bundle)
    p = sub.add_parser("pack"); p.add_argument("version"); p.add_argument("--out", default="dist"); p.set_defaults(fn=cmd_pack)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
