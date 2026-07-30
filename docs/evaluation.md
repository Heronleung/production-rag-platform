# RAG evaluation and regression gates (Phase 4)

Phase 3 proved that MMR and multi-query execute and return different evidence.
That is not the same as proving that either strategy improves quality. Phase 4
adds a repeatable evaluation loop over the production retrieval and answer
pipeline.

## Current scope

The first Phase 4 increment is dependency-free and deterministic:

- a validated JSONL golden dataset;
- four comparable strategies: `vanilla`, `mmr`, `multi_query`, `combined`;
- hit rate, reciprocal rank, source recall and key precision;
- mean, p50 and p95 end-to-end latency;
- one JSON report schema for all strategies;
- a baseline comparator that exits non-zero on regression.

Ragas is deliberately a second increment. Ragas 0.4 introduced a new experiment
and LLM-factory architecture, and adding it changes a large dependency graph.
It must be added on a networked machine together with the regenerated `uv.lock`.
Never edit only `pyproject.toml`: that recreates the stale-lock problem that
previously left console scripts missing from the virtual environment.

## Golden dataset

`evaluation/datasets/golden.jsonl` contains ten samples. Each row has:

```json
{
  "id": "index-types",
  "question": "What index types does the vector database support?",
  "reference_answer": "The vector database supports ...",
  "reference_sources": ["OCR_test.pdf"],
  "reference_chunk_keys": ["OCR_test.pdf::0"],
  "tags": ["verified", "retrieval", "milvus"]
}
```

Only `index-types` has a chunk key verified from a live citation so far. The
other rows are tagged `needs-chunk-review`. Metrics that require a missing
annotation are reported as `null`, not zero. This prevents incomplete labelling
from creating false failures or inflated scores.

Before treating hit rate or MRR as representative, inspect each query's live
citations and add all acceptable `source::chunk_index` keys. A single-source
corpus makes source recall easy to satisfy, so chunk-key annotation is the more
useful signal.

## Deterministic metrics

| Metric | Ground truth | Meaning |
| --- | --- | --- |
| `hit_rate_at_k` | chunk keys | 1 when any expected chunk appears in top-k |
| `mrr_at_k` | chunk keys | reciprocal rank of the first expected chunk |
| `key_precision_at_k` | chunk keys | fraction of retrieved keys explicitly marked relevant |
| `source_recall_at_k` | sources | fraction of expected sources represented |

The aggregate ignores `null` samples for that metric. It never silently treats
an unlabelled question as a miss.

## Run an evaluation

Milvus and Ollama must be healthy first:

```bash
curl -s http://localhost:8000/readyz | python3 -m json.tool
```

Then run each strategy against exactly the same dataset:

```bash
uv run python scripts/evaluate.py \
  --strategy vanilla \
  --output evaluation/reports/vanilla.json

uv run python scripts/evaluate.py \
  --strategy mmr \
  --output evaluation/reports/mmr.json

uv run python scripts/evaluate.py \
  --strategy multi_query \
  --output evaluation/reports/multi-query.json

uv run python scripts/evaluate.py \
  --strategy combined \
  --output evaluation/reports/combined.json
```

Each report records the dataset SHA-256, retrieval strategy, top-k, embedding
model, chat model, per-sample answer/evidence/scores/errors, and aggregate
metrics. Generated reports are ignored by Git. Commit only a reviewed baseline
under `evaluation/baselines/`.

## Regression gate

After selecting a reviewed baseline:

```bash
mkdir -p evaluation/baselines
cp evaluation/reports/vanilla.json evaluation/baselines/local.json

uv run python scripts/evaluate.py \
  --strategy vanilla \
  --output evaluation/reports/current.json \
  --baseline evaluation/baselines/local.json \
  --fail-on-regression
```

Default allowed drops:

| Metric | Allowed drop |
| --- | ---: |
| hit rate | 0.00 |
| MRR | 0.02 |
| source recall | 0.02 |
| key precision | 0.02 |

Latency is reported but not hard-gated on a developer laptop because model
loading and host contention make it noisy. Phase 7 can gate latency in a stable
runner environment.

## Tests

```bash
uv run pytest tests/test_evaluation.py tests/test_retrieval.py tests/test_api.py
```

`tests/test_evaluation.py` is fully offline. It covers malformed and duplicate
dataset rows, metric mathematics, absent annotations, aggregation, regression
within tolerance and regression beyond tolerance.

## Ragas second increment

The LLM-judged metrics planned for the reviewed reports are:

- Faithfulness: whether answer claims are supported by retrieved contexts;
- Response relevancy: whether the answer addresses the question;
- Context precision: whether retrieved contexts are useful for the reference;
- Context recall: whether the retrieved contexts cover the reference claims.

On the networked development machine, add the pinned optional dependency and
commit both files produced by uv:

```bash
uv add --optional eval 'ragas==0.4.3'
git add pyproject.toml uv.lock
```

Do not commit that dependency update until the adapter has run successfully
with the intended evaluator provider. LLM-judged scores are initially report-only;
set a regression threshold only after repeated runs establish their variance.
