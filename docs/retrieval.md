# Retrieval Quality (Phase 3)

Phase 2 shipped a working `/query` endpoint backed by plain cosine top-k
retrieval. Phase 3 adds two orthogonal improvements that can be enabled
independently via `POST /query` flags.

---

## Why plain top-k is not enough

| Problem | Symptom | Phase 3 fix |
| --- | --- | --- |
| Redundant chunks | LLM context is full of paraphrases; answer misses different facts | MMR |
| Phrasing mismatch | Corpus uses "vector index" but user asks "ANN algorithm" | Multi-query |

---

## Maximal Marginal Relevance (MMR)

MMR (Carbonell & Goldstein, 1998) greedily selects a diverse subset from an
initial over-fetch of candidates.

```
score(d) = λ · sim(d, query) − (1−λ) · max_{s ∈ S} sim(d, s)
```

- **λ = 1.0** — pure relevance (degenerates to top-k).
- **λ = 0.0** — pure diversity.
- **λ = 0.5** — default; balanced.

No new dependencies: similarity is computed with the stored chunk vectors
(requested from Milvus via `return_vectors=True`) using the same cosine
calculation already in `InMemoryStore`.

### When to use it

MMR is most useful when the corpus has many chunks from a single long document.
Without it, all top-k slots may be filled by adjacent chunks saying the same
thing.

### Curl example

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How does Milvus handle index building?",
    "top_k": 5,
    "use_mmr": true,
    "mmr_lambda": 0.6
  }'
```

`mmr_lambda: 0.6` slightly favours relevance while still penalising redundancy.
`mmr_fetch_k` defaults to `top_k * 4` (20 candidates fetched, 5 selected).

---

## Multi-query retrieval

A single embedding can miss chunks that express the same idea in different
terminology. Multi-query asks the LLM to rephrase the question, embeds each
variant independently, and merges the result sets.

### Pipeline

1. LLM generates `multi_query_count` alternative phrasings (JSON array).
2. Original + all variants are each embedded and searched.
3. Results are merged, deduplicated by `source::chunk_index`, keeping the
   highest score seen for each chunk across all queries.
4. Merged set is sorted by score and truncated to `top_k`.
5. If LLM output is unparseable, the function falls back to the original
   question only — the endpoint never errors because of expansion failure.

### When to use it

Useful when queries use vocabulary that differs from the ingested documents.
It costs one LLM `complete()` call and `(1 + count)` embed calls per request,
so it adds latency proportional to model speed.

### Curl example

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "approximate nearest neighbour search",
    "top_k": 5,
    "multi_query": true,
    "multi_query_count": 3
  }'
```

---

## Combining both

MMR and multi-query compose cleanly: multi-query expands recall first, then MMR
reduces redundancy in the merged candidate set.

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How does Milvus persist data?",
    "top_k": 5,
    "use_mmr": true,
    "mmr_lambda": 0.5,
    "multi_query": true,
    "multi_query_count": 3
  }'
```

With both enabled: `multi_query_count + 1` searches are executed, results are
merged, then MMR selects the final `top_k` from the merged set. The over-fetch
budget (`mmr_fetch_k`) applies to each individual search before merging.

---

## Backward compatibility

All new `QueryRequest` fields default to off:

| Field | Default | Effect when off |
| --- | --- | --- |
| `use_mmr` | `false` | Plain top-k, no vector return |
| `mmr_lambda` | `0.5` | N/A |
| `mmr_fetch_k` | `null` (`top_k * 4`) | N/A |
| `multi_query` | `false` | Single query, no LLM expansion call |
| `multi_query_count` | `3` | N/A |

Existing clients that do not send these fields get identical Phase 2 behaviour.

---

## What Phase 3 does not do: cross-encoder reranking

A cross-encoder scores each `(query, chunk)` pair jointly and is strictly more
accurate than bi-encoder cosine similarity for reranking. However,
`sentence-transformers` is a heavy dependency (~1 GB model download) and adds
start-up time. Reranking is deferred to **Phase 4**, where the RAGAS evaluation
pipeline will let us measure the improvement before committing the cost.
