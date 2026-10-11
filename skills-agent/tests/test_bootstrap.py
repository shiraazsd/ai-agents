"""Cold-start secret loading, against stubbed AWS clients."""

import sys
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import Stubber  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import aws_bootstrap  # noqa: E402


def test_loads_api_key_and_keytab(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/support-agent/google-api-key")
    monkeypatch.setenv("KEYTAB_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123456789012:secret:kt")
    monkeypatch.delenv("KRB5_CONF_PARAM", raising=False)

    ssm, sm = boto3.client("ssm"), boto3.client("secretsmanager")
    with Stubber(ssm) as s1, Stubber(sm) as s2:
        s1.add_response("get_parameter", {"Parameter": {"Value": "k3y"}}, {"Name": "/support-agent/google-api-key", "WithDecryption": True})
        s2.add_response("get_secret_value", {"SecretBinary": b"\x05\x02keytab"}, {"SecretId": "arn:aws:secretsmanager:us-east-1:123456789012:secret:kt"})
        monkeypatch.setattr(boto3, "client", lambda name, **_: {"ssm": ssm, "secretsmanager": sm}[name])
        aws_bootstrap.load_secrets()

    import os

    assert os.environ["GOOGLE_API_KEY"] == "k3y"
    assert Path(os.environ["KRB5_CLIENT_KTNAME"]).read_bytes() == b"\x05\x02keytab"
