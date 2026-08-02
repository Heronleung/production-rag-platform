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
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s

echo "kind cluster '$CLUSTER_NAME' is ready"
echo "add '127.0.0.1 rag.local' to /etc/hosts and browse http://rag.local:8080"
