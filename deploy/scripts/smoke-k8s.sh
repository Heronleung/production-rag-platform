#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://rag.local:8080}
RESOLVE=${RESOLVE:-rag.local:8080:127.0.0.1}
SMOKE_FILE=${SMOKE_FILE:-}

curl --fail --silent --show-error --resolve "$RESOLVE" "$BASE_URL/healthz"
echo
curl --fail --silent --show-error --resolve "$RESOLVE" "$BASE_URL/api/ready"
echo

if [[ -n "$SMOKE_FILE" ]]; then
  curl --fail --silent --show-error --resolve "$RESOLVE" \
    -F "file=@${SMOKE_FILE}" "$BASE_URL/api/ingest"
  echo
  curl --fail --no-buffer --show-error --resolve "$RESOLVE" \
    -H 'Content-Type: application/json' \
    -d '{"query":"What does this document say about vector indexes?","top_k":5}' \
    "$BASE_URL/api/query"
fi
