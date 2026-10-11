"""Manage the pinned marketplace skills bundle.

  python scripts/skills.py fetch [--tarball FILE]   download (or use FILE), verify sha256, unpack to support_agent/_skills
  python scripts/skills.py bump [--version vX]      point skills.lock at the latest (or given) release

GitHub auth for the private marketplace: GH_TOKEN / GITHUB_TOKEN, else `gh auth token`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "skills.lock"
DEST = ROOT / "support_agent" / "_skills"
API = "https://api.github.com"


def _token() -> str | None:
    tok = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not tok and shutil.which("gh"):
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip() or None
    return tok


def _gh(url: str, **headers) -> requests.Response:
    tok = _token()
    h = {"Accept": "application/vnd.github+json", **headers, **({"Authorization": f"Bearer {tok}"} if tok else {})}
    r = requests.get(url, headers=h, timeout=60)
    r.raise_for_status()
    return r


def _download(repo: str, version: str) -> bytes:
    release = _gh(f"{API}/repos/{repo}/releases/tags/skills-{version}").json()
    (asset,) = [a for a in release["assets"] if a["name"] == f"skills-{version}.tar.gz"]
    return _gh(asset["url"], Accept="application/octet-stream").content


def read_lock() -> dict:
    return yaml.safe_load(LOCK.read_text())


def write_lock(lock: dict) -> None:
    header = "".join(line for line in LOCK.read_text().splitlines(keepends=True) if line.startswith("#"))
    LOCK.write_text(header + f"repo: {lock['repo']}\nversion: {lock['version']}\nsha256: {lock['sha256']}\n")


def cmd_fetch(a) -> None:
    lock = read_lock()
    data = Path(a.tarball).read_bytes() if a.tarball else _download(lock["repo"], lock["version"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != lock["sha256"]:
        sys.exit(f"sha256 mismatch for {lock['version']}: lock={lock['sha256']} got={digest}")
    shutil.rmtree(DEST, ignore_errors=True)
    DEST.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(DEST, filter="data")
    print(f"skills {lock['version']} -> {DEST}")


def cmd_bump(a) -> None:
    lock = read_lock()
    version = a.version
    if not version:
        releases = _gh(f"{API}/repos/{lock['repo']}/releases?per_page=30").json()
        tags = [r["tag_name"] for r in releases if r["tag_name"].startswith("skills-") and not r["draft"]]
        if not tags:
            sys.exit("no skills-* releases found")
        version = tags[0].removeprefix("skills-")
    if version == lock["version"] and not a.version:
        return print(f"already on {version}")
    digest = hashlib.sha256(_download(lock["repo"], version)).hexdigest()
    print(f"{lock['version']} -> {version}")
    write_lock({**lock, "version": version, "sha256": digest})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch"); p.add_argument("--tarball"); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("bump"); p.add_argument("--version"); p.set_defaults(fn=cmd_bump)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
