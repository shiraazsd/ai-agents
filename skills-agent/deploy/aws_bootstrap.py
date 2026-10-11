"""Cold-start secrets for Lambda: pulled from SSM / Secrets Manager into env before the agent loads.

  GOOGLE_API_KEY_PARAM  SSM SecureString holding a Gemini API key      -> GOOGLE_API_KEY
  KEYTAB_SECRET_ARN     Secrets Manager secret (binary or base64) keytab -> KRB5_CLIENT_KTNAME
  KRB5_CONF_PARAM       SSM String/SecureString with krb5.conf contents  -> KRB5_CONFIG
"""

import base64
import os
from pathlib import Path


def _ssm(name: str) -> str:
    import boto3

    return boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]


def load_secrets() -> None:
    if (name := os.getenv("GOOGLE_API_KEY_PARAM")) and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = _ssm(name)
    if name := os.getenv("KRB5_CONF_PARAM"):
        conf = Path("/tmp/krb5.conf")
        conf.write_text(_ssm(name))
        os.environ["KRB5_CONFIG"] = str(conf)
    if arn := os.getenv("KEYTAB_SECRET_ARN"):
        import boto3

        secret = boto3.client("secretsmanager").get_secret_value(SecretId=arn)
        keytab = Path("/tmp/service.keytab")
        keytab.write_bytes(secret.get("SecretBinary") or base64.b64decode(secret["SecretString"]))
        keytab.chmod(0o600)
        os.environ["KRB5_CLIENT_KTNAME"] = str(keytab)
