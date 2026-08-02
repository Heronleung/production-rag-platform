# Containers and Kubernetes (Phase 6)

Phase 6 packages the FastAPI and Next.js application layer into non-root images and deploys it with Helm. The first target is a local **kind** cluster. Milvus and Ollama stay on the host initially; every endpoint remains configurable for later managed or in-cluster services.

## Decisions

| Concern | Initial choice |
| --- | --- |
| Cluster | kind |
| Images | build locally and `kind load docker-image` |
| Milvus / Ollama | host services, configured through chart values |
| Ingress | ingress-nginx |
| Hostname / TLS | `http://rag.local:8080`, no TLS |
| Secrets | existing Kubernetes Secret referenced through `api.existingSecret` |

No secret value is stored in the chart. For OpenAI or authenticated Milvus, create a Secret separately:

```bash
kubectl -n rag create secret generic rag-provider-secrets \
  --from-literal=OPENAI_API_KEY='...' \
  --from-literal=MILVUS_TOKEN='...'

helm upgrade --install rag deploy/helm/rag-platform \
  --namespace rag --create-namespace \
  --set api.existingSecret=rag-provider-secrets
```

## Local containers before Kubernetes

Start the dependency services first:

```bash
docker compose -f deploy/compose/milvus.yml up -d
docker compose -f deploy/compose/ollama.yml up -d
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull qwen2.5:1.5b
```

Build and run the app layer:

```bash
docker compose -f deploy/compose/app.yml up --build -d
curl -f http://localhost:8000/healthz
curl -f http://localhost:3000/healthz
curl -f http://localhost:3000/api/ready
```

## kind deployment

Requirements: Docker, kind, kubectl, and Helm.

```bash
bash deploy/scripts/kind-up.sh
bash deploy/scripts/build-and-load.sh
```

Add this hosts entry once:

```text
127.0.0.1 rag.local
```

Then open <http://rag.local:8080>.

The local chart values use `host.docker.internal` for Milvus and Ollama. Docker Desktop provides this hostname. On native Linux, set explicit reachable gateway URLs instead:

```bash
helm upgrade --install rag deploy/helm/rag-platform \
  --namespace rag --create-namespace \
  -f deploy/helm/rag-platform/values-kind.yaml \
  --set-string api.env.OLLAMA_BASE_URL=http://172.17.0.1:11434 \
  --set-string api.env.MILVUS_URI=http://172.17.0.1:19530
```

Confirm the gateway for your Docker setup before using it; do not assume `172.17.0.1` on every host.

## Health semantics

- API `/healthz`: process-only liveness; never checks dependencies.
- API `/readyz`: Milvus/embedder readiness; returns 503 when unavailable.
- Web `/healthz`: frontend-only liveness; does not restart the web Pod when the API is down.
- Web `/api/ready`: user-facing composite readiness proxied to the API.

## SSE and uploads

The Ingress disables response and request buffering, uses 180-second proxy timeouts, and permits a 26MB request body. The backend remains authoritative with its 25MB upload limit. Verify streaming through the Ingress rather than only through port-forwarding:

```bash
SMOKE_FILE=/path/to/OCR_test.pdf bash deploy/scripts/smoke-k8s.sh
```

## Security defaults

Both containers run as non-root, drop Linux capabilities, deny privilege escalation, and use read-only root filesystems with writable `emptyDir` mounts only for `/tmp` and the Next.js cache. The API is ClusterIP-only; only the web Service is exposed by Ingress.

NetworkPolicy is disabled until the external dependency CIDRs are known. Enabling it without `networkPolicy.allowedExternalCIDRs` intentionally blocks API access to host Milvus/Ollama. Configure the reachable CIDRs before enabling it.

## Rollback

Use immutable image tags outside local development. After a failed release:

```bash
helm history rag -n rag
helm rollback rag <REVISION> -n rag --wait
kubectl get deployments -n rag
```
