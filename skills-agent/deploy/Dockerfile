# AgentCore Runtime requires linux/arm64. Built from the repo root after
# `python scripts/skills.py fetch` (see .github/workflows/deploy.yml).
FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.12-slim

# KERBEROS=true adds GSSAPI so skills can use Kerberos (keytab) auth to internal backends.
ARG KERBEROS=false
RUN if [ "$KERBEROS" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends gcc libkrb5-dev && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$KERBEROS" = "true" ]; then pip install --no-cache-dir requests-gssapi; fi

COPY support_agent ./support_agent
COPY deploy/main.py deploy/aws_bootstrap.py ./

RUN useradd -m -u 1000 agent
USER agent
# DOCKER_CONTAINER makes BedrockAgentCoreApp bind 0.0.0.0:8080 (/invocations, /ping).
ENV PYTHONUNBUFFERED=1 DOCKER_CONTAINER=1
EXPOSE 8080
CMD ["python", "main.py"]
