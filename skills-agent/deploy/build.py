"""Build and push the agent image in AWS CodeBuild (no local Docker needed).

  python scripts/skills.py fetch                  # skills must be in support_agent/_skills
  python deploy/build.py [--tag TAG] [--kerberos]  # prints the pushed image URI

Uses the support-agent-build and support-agent-bootstrap stacks; AWS credentials from the environment.
"""

import argparse
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [".dockerignore", "deploy", "support_agent"]
SKIP = {"__pycache__", "evals.yaml", "fixtures"}


def outputs(cf, stack: str) -> dict:
    return {o["OutputKey"]: o["OutputValue"] for o in cf.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])}


def context_zip() -> bytes:
    if not (ROOT / "support_agent/_skills/skillkit").is_dir():
        sys.exit("no skills bundle: run `python scripts/skills.py fetch` first")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        for top in INCLUDE:
            for p in [ROOT / top] if (ROOT / top).is_file() else sorted((ROOT / top).rglob("*")):
                if p.is_file() and not SKIP & set(p.relative_to(ROOT).parts):
                    z.write(p, p.relative_to(ROOT).as_posix())
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--kerberos", action="store_true")
    a = ap.parse_args()
    tag = a.tag or subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or str(int(time.time()))

    cf = boto3.client("cloudformation")
    build, boot = outputs(cf, "support-agent-build"), outputs(cf, "support-agent-bootstrap")
    image = f"{boot['ImageRepoUri']}:{tag}"
    boto3.client("s3").put_object(Bucket=build["SourceBucket"], Key="context.zip", Body=context_zip())

    cb = boto3.client("codebuild")
    env = [{"name": "IMAGE_URI", "value": image, "type": "PLAINTEXT"},
           {"name": "KERBEROS", "value": str(a.kerberos).lower(), "type": "PLAINTEXT"}]
    bid = cb.start_build(projectName=build["ProjectName"], environmentVariablesOverride=env)["build"]["id"]
    print(f"build {bid} -> {image}", flush=True)
    while True:
        b = cb.batch_get_builds(ids=[bid])["builds"][0]
        if b["buildComplete"]:
            break
        print(f"  {b.get('currentPhase')}…", flush=True)
        time.sleep(20)
    if b["buildStatus"] != "SUCCEEDED":
        logs = b.get("logs", {})
        sys.exit(f"build {b['buildStatus']}: see CloudWatch {logs.get('groupName')}/{logs.get('streamName')}")
    print(image)


if __name__ == "__main__":
    main()
