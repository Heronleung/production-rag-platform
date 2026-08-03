#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CLUSTER_NAME=${CLUSTER_NAME:-rag}
INGRESS_NGINX_VERSION=${INGRESS_NGINX_VERSION:-controller-v1.12.1}

for command in docker kind kubectl helm; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done

if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME" --config "$ROOT_DIR/deploy/kind/cluster.yaml"
fi

INGRESS_SOURCE="https:""//raw.githubusercontent.com"
kubectl apply -f "${INGRESS_SOURCE}/kubernetes/ingress-nginx/${INGRESS_NGINX_VERSION}/deploy/static/provider/kind/deploy.yaml"

# The admission Jobs and controller pod are created asynchronously. Waiting on
# a pod selector immediately after apply can fail with "no matching resources"
# before the Deployment has created its first pod. Wait on named resources
# instead, then verify that the admission Service has a ready endpoint before
# Helm creates any Ingress objects.
kubectl wait --namespace ingress-nginx \
  --for=condition=complete job/ingress-nginx-admission-create \
  --timeout=180s
kubectl wait --namespace ingress-nginx \
  --for=condition=complete job/ingress-nginx-admission-patch \
  --timeout=180s
kubectl rollout status --namespace ingress-nginx \
  deployment/ingress-nginx-controller \
  --timeout=180s

for attempt in $(seq 1 30); do
  admission_endpoint=$(kubectl --namespace ingress-nginx get endpoints \
    ingress-nginx-controller-admission \
    --output=jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)
  if [[ -n "$admission_endpoint" ]]; then
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    echo "ingress-nginx admission endpoint did not become ready" >&2
    exit 1
  fi
  sleep 2
done

echo "kind cluster '$CLUSTER_NAME' is ready"
echo "add '127.0.0.1 rag.local' to /etc/hosts and browse http://rag.local:8080"
