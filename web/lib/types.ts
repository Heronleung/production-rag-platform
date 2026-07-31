export type Citation = {
  source: string;
  chunk_index: number;
  score: number;
  text: string;
};

export type QuerySettings = {
  topK: number;
  sourceFilter: string;
  temperature: number;
  useMmr: boolean;
  mmrLambda: number;
  multiQuery: boolean;
  multiQueryCount: number;
};

export type QueryPayload = {
  query: string;
  top_k: number;
  source_filter?: string;
  temperature: number;
  stream: true;
  use_mmr: boolean;
  mmr_lambda: number;
  multi_query: boolean;
  multi_query_count: number;
};

export type ReadinessDependency = {
  name: string;
  ok: boolean;
  detail: string;
};

export type ReadinessResponse = {
  status: string;
  dependencies: ReadinessDependency[];
};

export type IngestResponse = {
  source: string;
  chunks_written: number;
  embedding_model: string;
  elapsed_seconds: number;
};
