# Model providers: local Ollama vs OpenAI API

The project ships two interchangeable providers for both embeddings and chat.
They live side by side and are selected by configuration, never by code changes.

```
api/embeddings/
├── base.py             # Embedder ABC + offline HashEmbedder for tests
├── ollama_embedder.py  # local models, no API key
├── openai_embedder.py  # hosted API
└── __init__.py         # get_embedder() factory

api/llm.py                 # OllamaChat + OpenAIChat + get_llm() factory
```

## Switching provider

```dotenv
# .env
EMBEDDING_PROVIDER=ollama   # or: openai
LLM_PROVIDER=ollama         # or: openai
```

Nothing else changes. `get_embedder()` and `get_llm()` read these two values and
return the right implementation.

## Local setup (default, no API key)

```bash
# Option A: native install
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Option B: container
docker compose -f deploy/compose/ollama.yml up -d
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull llama3.1:8b

# Verify before touching Milvus
uv run python scripts/check_embedder.py
```

## Hosted setup

```bash
uv sync --extra openai
# .env: EMBEDDING_PROVIDER=openai, LLM_PROVIDER=openai, OPENAI_API_KEY=sk-...
uv run python scripts/check_embedder.py --provider openai
```

## Choosing an embedding model

| Provider | Model | Dim | Notes |
| --- | --- | --- | --- |
| ollama | `nomic-embed-text` | 768 | Default. 274 MB, fast on CPU, 8192-token context. |
| ollama | `mxbai-embed-large` | 1024 | Stronger retrieval quality, roughly 670 MB. |
| ollama | `bge-m3` | 1024 | Multilingual, good for mixed Chinese and English corpora. |
| ollama | `all-minilm` | 384 | Smallest and fastest, noticeably weaker. |
| openai | `text-embedding-3-small` | 1536 | Cheap hosted baseline. |
| openai | `text-embedding-3-large` | 3072 | Highest quality, highest cost and storage. |

## The rule that must not be broken

**One collection, one embedding model.** Vectors from different models are not
comparable, and the Milvus schema fixes the dimension at creation time. After
changing the embedding model you must re-embed and rebuild:

```bash
uv run python scripts/check_embedder.py            # confirm the new dimension
uv run python scripts/migrate_chroma_to_milvus.py --recreate
```

A sensible convention is to put the model in the collection name, for example
`MILVUS_COLLECTION=rag_chunks_nomic768`, so two models can coexist while you
compare them.

## Practical trade-offs

| Dimension | Ollama (local) | OpenAI (hosted) |
| --- | --- | --- |
| Cost | zero marginal cost | per token, and evaluation runs are token-heavy |
| Privacy | documents never leave the machine | documents are sent to a third party |
| Throughput | bounded by local CPU or GPU | bounded by rate limits |
| Reproducibility | model pinned locally, fully stable | provider may deprecate a model |
| Setup | needs a daemon and disk space | needs only a key |

For this project the recommended split is: **develop and evaluate on Ollama**,
because Phase 4 fires thousands of evaluation calls, then optionally re-run the
final benchmark on OpenAI to include a hosted-baseline column in the README.
That comparison table is itself a good interview artefact.
