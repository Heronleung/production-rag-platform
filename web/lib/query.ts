import type { QueryPayload, QuerySettings } from "./types";

export const DEFAULT_QUERY_SETTINGS: QuerySettings = {
  topK: 5,
  sourceFilter: "",
  temperature: 0,
  useMmr: false,
  mmrLambda: 0.5,
  multiQuery: false,
  multiQueryCount: 3,
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
  };
}
