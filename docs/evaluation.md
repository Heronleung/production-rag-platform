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

All ten top-ranked chunk keys were manually reviewed from the first live vanilla
report and are now marked `verified`. The review also aligned reference answers
with the actual corpus. This matters because an earlier draft described the
repository's intended implementation in several places while `OCR_test.pdf`
describes a different hypothetical production design—for example:

- the corpus uses a dual embedding strategy and all-MiniLM-L6-v2 for general text;
- Kubernetes deployment uses ArgoCD and Helm GitOps;
- secret management uses HashiCorp Vault and the Secrets Store CSI Driver.

Using repository-state answers against that corpus would invalidate LLM-judged
faithfulness, relevancy and correctness scores.

## Deterministic metrics

| Metric | Ground truth | Meaning | Gate status |
| --- | --- | --- | --- |
| `hit_rate_at_k` | verified top chunks | 1 when an expected chunk appears in top-k | hard gate |
| `mrr_at_k` | verified top chunks | reciprocal rank of the first expected chunk | hard gate |
| `source_recall_at_k` | sources | fraction of expected sources represented | hard gate |
| `key_precision_at_k` | all acceptable keys | fraction of retrieved keys explicitly marked relevant | report only |

Only the top-ranked chunk for each question has been reviewed. Consequently,
`key_precision_at_k = 0.2` currently means one verified key among five retrieved
keys; it is a lower bound, not evidence that the other four chunks are wrong.
It remains report-only until every acceptable top-k chunk is annotated.

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

Key precision and latency are reported but not hard-gated. Key precision needs
complete top-k annotation. Laptop latency is noisy because model loading and
host contention vary; Phase 7 can gate it in a stable runner environment.

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
