# Runtime LLM provider selection

The UI can choose a chat provider per query without changing the embedding model or rebuilding the Milvus collection.

## Data flow

1. Ollama creates the question embedding with the configured embedding model.
2. Milvus retrieves the relevant stored chunks.
3. FastAPI sends the numbered context to the chat provider selected in the UI.
4. Citations and answer tokens return through the existing SSE contract.

Only answer generation changes. Stored vectors remain compatible with the collection schema.

## Local Ollama

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
ollama serve
```

The default chat model is `qwen2.5:3b`. The settings interface accepts any model name available from `GET /api/tags`.

## DeepSeek

Set the key only in the FastAPI environment:

```bash
export DEEPSEEK_API_KEY='replace-locally'
export DEEPSEEK_LLM_MODEL=deepseek-chat
```

Supported defaults are `deepseek-chat` and `deepseek-reasoner`.

## OpenAI

Set the key only in the FastAPI environment:

```bash
export OPENAI_API_KEY='replace-locally'
export OPENAI_LLM_MODEL=gpt-4o-mini
```

Do not use `NEXT_PUBLIC_` for either provider key. The browser sends only the provider and model names.

## Connection check

```bash
curl -sS -X POST http://127.0.0.1:8000/llm/check \
  -H 'Content-Type: application/json' \
  -d '{"provider":"ollama","model":"qwen2.5:3b"}'
```

A successful response identifies the provider and model. Authentication, missing-model, rate-limit, and upstream failures are sanitized so API keys and Authorization headers are not returned.

## Query override

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"Which index types does Milvus support?",
    "llm_provider":"deepseek",
    "llm_model":"deepseek-chat",
    "stream":true
  }'
```

Omitting `llm_provider` and `llm_model` preserves the server defaults.

## Docker Compose

Compose reads keys from the local shell or an untracked `.env` file:

```bash
export DEEPSEEK_API_KEY='replace-locally'
docker compose -f deploy/compose/milvus.yml up -d
docker compose -f deploy/compose/app.yml up -d --build
```

## Kubernetes

Create a Secret separately and reference it through `api.existingSecret`:

```bash
kubectl -n rag create secret generic rag-llm-provider-keys \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
  --from-literal=DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install rag deploy/helm/rag-platform \
  --namespace rag \
  -f deploy/helm/rag-platform/values-kind.yaml \
  --set api.existingSecret=rag-llm-provider-keys
```

The keys must not appear under `api.env`; that map is rendered into a ConfigMap.
