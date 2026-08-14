import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_QUERY_SETTINGS, toQueryPayload } from "../lib/query.ts";

test("keeps vanilla retrieval and local Ollama as defaults", () => {
  const payload = toQueryPayload("  What is Milvus?  ", DEFAULT_QUERY_SETTINGS);
  assert.equal(payload.query, "What is Milvus?");
  assert.equal(payload.top_k, 5);
  assert.equal(payload.use_mmr, false);
  assert.equal(payload.multi_query, false);
  assert.equal(payload.llm_provider, "ollama");
  assert.equal(payload.llm_model, "qwen2.5:3b");
  assert.equal("source_filter" in payload, false);
});

test("maps advanced retrieval and provider controls to the backend schema", () => {
  const payload = toQueryPayload("question", {
    ...DEFAULT_QUERY_SETTINGS,
    sourceFilter: " OCR_test.pdf ",
    useMmr: true,
    mmrLambda: 0.7,
    multiQuery: true,
    multiQueryCount: 4,
    llmProvider: "deepseek",
    llmModel: "deepseek-chat",
  });
  assert.equal(payload.source_filter, "OCR_test.pdf");
  assert.equal(payload.mmr_lambda, 0.7);
  assert.equal(payload.multi_query_count, 4);
  assert.equal(payload.llm_provider, "deepseek");
  assert.equal(payload.llm_model, "deepseek-chat");
});
