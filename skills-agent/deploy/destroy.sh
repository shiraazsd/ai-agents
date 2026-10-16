#!/usr/bin/env bash
# Removes everything the support-agent deploy created in this account/region:
# the four CloudFormation stacks, their ECR images and S3 build sources, and AgentCore runtime logs.
# Run in AWS CloudShell (uses your console identity):  bash destroy.sh
set -euo pipefail
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
STACKS="support-agent support-agent-budget support-agent-build support-agent-bootstrap"

echo "Account $(aws sts get-caller-identity --query Account --output text), region $AWS_DEFAULT_REGION"
for s in $STACKS; do
  st=$(aws cloudformation describe-stacks --stack-name "$s" --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "not found")
  echo "  stack $s: $st"
done
echo "  ECR images: $(aws ecr list-images --repository-name support-agent --query 'length(imageIds)' --output text 2>/dev/null || echo 0)"
read -rp "Delete all of the above? Type DELETE: " answer
[ "$answer" = "DELETE" ] || { echo "aborted, nothing deleted"; exit 1; }

delete_stack() {
  if aws cloudformation describe-stacks --stack-name "$1" >/dev/null 2>&1; then
    echo "deleting $1 ..."
    aws cloudformation delete-stack --stack-name "$1"
    aws cloudformation wait stack-delete-complete --stack-name "$1" \
      || { echo "  $1 failed to delete; see: aws cloudformation describe-stack-events --stack-name $1"; exit 1; }
    echo "  $1 deleted"
  fi
}

delete_stack support-agent            # AgentCore runtime + its role
delete_stack support-agent-budget     # budget, deny-all policy, budget action

bucket=$(aws cloudformation describe-stacks --stack-name support-agent-build \
  --query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text 2>/dev/null || true)
if [ -n "$bucket" ] && [ "$bucket" != "None" ]; then aws s3 rm "s3://$bucket" --recursive; fi
delete_stack support-agent-build      # CodeBuild project, role, bucket, logs

if aws ecr describe-repositories --repository-names support-agent >/dev/null 2>&1; then
  ids=$(aws ecr list-images --repository-name support-agent --query imageIds --output json)
  if [ "$ids" != "[]" ]; then aws ecr batch-delete-image --repository-name support-agent --image-ids "$ids" >/dev/null; fi
fi
delete_stack support-agent-bootstrap  # ECR repo, GitHub OIDC provider, deploy role

for g in $(aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/support_agent \
             --query "logGroups[].logGroupName" --output text); do
  aws logs delete-log-group --log-group-name "$g" && echo "deleted log group $g"
done

echo
echo "Check (all should be empty):"
echo "  stacks:   $(aws cloudformation list-stacks --query "StackSummaries[?starts_with(StackName,'support-agent') && StackStatus!='DELETE_COMPLETE'].StackName" --output text)"
echo "  runtimes: $(aws bedrock-agentcore-control list-agent-runtimes --query 'agentRuntimes[].agentRuntimeName' --output text)"
echo "  budgets:  $(aws budgets describe-budgets --account-id "$(aws sts get-caller-identity --query Account --output text)" --query 'Budgets[].BudgetName' --output text 2>/dev/null)"
echo "Done."
