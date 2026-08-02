#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CLUSTER_NAME=${CLUSTER_NAME:-rag}
IMAGE_TAG=${IMAGE_TAG:-dev}

cd "$ROOT_DIR"
docker build -f Dockerfile.api -t "rag-api:${IMAGE_TAG}" .
docker build -f web/Dockerfile -t "rag-web:${IMAGE_TAG}" web
kind load docker-image --name "$CLUSTER_NAME" "rag-api:${IMAGE_TAG}" "rag-web:${IMAGE_TAG}"

helm upgrade --install rag deploy/helm/rag-platform \
  --namespace rag --create-namespace \
  -f deploy/helm/rag-platform/values-kind.yaml \
  --set-string api.image.tag="$IMAGE_TAG" \
  --set-string web.image.tag="$IMAGE_TAG" \
  --wait --timeout 5m

kubectl -n rag get pods,services,ingress
