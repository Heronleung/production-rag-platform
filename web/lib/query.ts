import type { LlmProvider, QueryPayload, QuerySettings } from "./types";

export const DEFAULT_MODELS: Record<LlmProvider, string> = {
  ollama: "qwen2.5:3b",
  deepseek: "deepseek-chat",
  openai: "gpt-4o-mini",
};

export const DEFAULT_QUERY_SETTINGS: QuerySettings = {
  topK: 5,
  sourceFilter: "",
  temperature: 0,
  useMmr: false,
  mmrLambda: 0.5,
  multiQuery: false,
  multiQueryCount: 3,
  llmProvider: "ollama",
  llmModel: DEFAULT_MODELS.ollama,
};

export function toQueryPayload(question: string, settings: QuerySettings): QueryPayload {
  const sourceFilter = settings.sourceFilter.trim();
  return {
    query: question.trim(),
    top_k: settings.topK,
    ...(sourceFilter ? { source_filter: sourceFilter } : {}),
    temperature: settings.temperature,
    stream: true,
    use_mmr: settings.useMmr,
    mmr_lambda: settings.mmrLambda,
    multi_query: settings.multiQuery,
    multi_query_count: settings.multiQueryCount,
    llm_provider: settings.llmProvider,
    llm_model: settings.llmModel.trim() || DEFAULT_MODELS[settings.llmProvider],
  };
}
