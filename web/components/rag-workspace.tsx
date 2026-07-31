"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import { DEFAULT_QUERY_SETTINGS, toQueryPayload } from "@/lib/query";
import { consumeSse, type SseEvent } from "@/lib/sse";
import type {
  Citation,
  IngestResponse,
  QuerySettings,
  ReadinessResponse,
} from "@/lib/types";

type QueryStatus = "idle" | "connecting" | "streaming" | "complete" | "error" | "cancelled";
type ReadyState = { kind: "checking" | "ready" | "degraded" | "offline"; detail: string };

type ApiError = { detail?: unknown; request_id?: string | null };

function detailText(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail) return JSON.stringify(detail);
  return fallback;
}

async function responseError(response: Response): Promise<{ message: string; requestId: string }> {
  let payload: ApiError = {};
  try {
    payload = (await response.json()) as ApiError;
  } catch {
    // A proxy or load balancer may return a non-JSON error page.
  }
  return {
    message: detailText(payload.detail, `${response.status} ${response.statusText}`),
    requestId: payload.request_id || response.headers.get("x-request-id") || "",
  };
}

function parseEventJson<T>(event: SseEvent): T {
  try {
    return JSON.parse(event.data) as T;
  } catch {
    throw new Error(`Malformed ${event.event} event from the backend.`);
  }
}

function StatusDot({ state }: { state: ReadyState }) {
  return (
    <span className={`status-pill status-${state.kind}`} title={state.detail}>
      <span aria-hidden="true" className="status-dot" />
      {state.kind === "checking" ? "Checking stack" : state.kind}
    </span>
  );
}

export function RagWorkspace() {
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [settings, setSettings] = useState<QuerySettings>(DEFAULT_QUERY_SETTINGS);
  const [status, setStatus] = useState<QueryStatus>("idle");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [model, setModel] = useState("");
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [queryError, setQueryError] = useState("");
  const [requestId, setRequestId] = useState("");
  const [ready, setReady] = useState<ReadyState>({ kind: "checking", detail: "Checking /readyz" });
  const abortRef = useRef<AbortController | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [ingestError, setIngestError] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const checkReady = useCallback(async () => {
    setReady({ kind: "checking", detail: "Checking /readyz" });
    try {
      const response = await fetch("/api/ready", { cache: "no-store" });
      const payload = (await response.json()) as ReadinessResponse & ApiError;
      if (response.ok && payload.status === "ready") {
        setReady({ kind: "ready", detail: "Embedder and vector store are healthy." });
        return;
      }
      const failed = payload.dependencies?.filter((item) => !item.ok) ?? [];
      setReady({
        kind: response.status === 502 ? "offline" : "degraded",
        detail: failed.length
          ? failed.map((item) => `${item.name}: ${item.detail || "not ready"}`).join(" · ")
          : detailText(payload.detail, "Backend dependencies are not ready."),
      });
    } catch (error) {
      setReady({ kind: "offline", detail: error instanceof Error ? error.message : "Network error" });
    }
  }, []);

  useEffect(() => {
    void checkReady();
    return () => abortRef.current?.abort();
  }, [checkReady]);

  const runQuery = useCallback(
    async (rawQuestion: string) => {
      const trimmed = rawQuestion.trim();
      if (!trimmed || status === "connecting" || status === "streaming") return;

      const controller = new AbortController();
      abortRef.current = controller;
      setLastQuestion(trimmed);
      setQuestion("");
      setAnswer("");
      setCitations([]);
      setModel("");
      setElapsed(null);
      setQueryError("");
      setRequestId("");
      setStatus("connecting");

      let terminalEvent = false;
      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(toQueryPayload(trimmed, settings)),
          signal: controller.signal,
        });
        setRequestId(response.headers.get("x-request-id") || "");
        if (!response.ok) {
          const failure = await responseError(response);
          setRequestId(failure.requestId);
          throw new Error(failure.message);
        }

        await consumeSse(response, (event) => {
          if (event.event === "citations") {
            const payload = parseEventJson<{ citations: Citation[]; llm_model: string }>(event);
            setCitations(payload.citations);
            setModel(payload.llm_model);
            setStatus("streaming");
          } else if (event.event === "token") {
            const payload = parseEventJson<{ text: string }>(event);
            setAnswer((current) => current + payload.text);
            setStatus("streaming");
          } else if (event.event === "done") {
            const payload = parseEventJson<{ elapsed_seconds: number }>(event);
            terminalEvent = true;
            setElapsed(payload.elapsed_seconds);
            setStatus("complete");
          } else if (event.event === "error") {
            terminalEvent = true;
            const payload = parseEventJson<{ detail?: string }>(event);
            throw new Error(payload.detail || "The model stream failed.");
          }
        });

        if (!terminalEvent) throw new Error("The response stream ended before a done event.");
      } catch (error) {
        if (controller.signal.aborted) {
          setStatus("cancelled");
        } else {
          setQueryError(error instanceof Error ? error.message : "Query failed.");
          setStatus("error");
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [settings, status],
  );

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runQuery(question);
  }

  async function submitIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIngestError("");
    setIngestResult(null);
    if (!file) {
      setIngestError("Choose a PDF, Markdown, or text file first.");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setIngestError("The backend upload limit is 25MB.");
      return;
    }
    if (!/\.(pdf|md|txt)$/i.test(file.name)) {
      setIngestError("Supported file extensions: .pdf, .md, .txt.");
      return;
    }
    if (chunkOverlap >= chunkSize) {
      setIngestError("Chunk overlap must be smaller than chunk size.");
      return;
    }

    setIngesting(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(
        `/api/ingest?chunk_size=${chunkSize}&chunk_overlap=${chunkOverlap}`,
        { method: "POST", body: form },
      );
      if (!response.ok) {
        const failure = await responseError(response);
        throw new Error(`${failure.message}${failure.requestId ? ` · request ${failure.requestId}` : ""}`);
      }
      setIngestResult((await response.json()) as IngestResponse);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await checkReady();
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : "Ingestion failed.");
    } finally {
      setIngesting(false);
    }
  }

  const busy = status === "connecting" || status === "streaming";

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">R</div>
          <div>
            <p className="eyebrow">Production RAG Platform</p>
            <h1>Evidence workbench</h1>
          </div>
        </div>
        <button className="status-button" type="button" onClick={() => void checkReady()}>
          <StatusDot state={ready} />
          <span className="refresh-label">Refresh</span>
        </button>
      </header>

      {ready.kind !== "ready" && (
        <section className={`stack-banner banner-${ready.kind}`} role="status">
          <strong>{ready.kind === "checking" ? "Checking the stack…" : "The RAG stack is not ready."}</strong>
          <span>{ready.detail}</span>
        </section>
      )}

      <div className="workspace-grid">
        <section className="chat-panel" aria-label="RAG conversation">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Grounded answer</p>
              <h2>Ask the corpus</h2>
            </div>
            {model && <span className="model-chip">{model}</span>}
          </div>

          <div className="conversation" aria-live="polite">
            {status === "idle" ? (
              <div className="empty-state">
                <span className="empty-orbit" aria-hidden="true">✦</span>
                <h3>Start with a question that needs evidence.</h3>
                <p>Citations arrive before the first answer token, so you can inspect retrieval while generation runs.</p>
                <div className="suggestions">
                  {["Which index types does Milvus support?", "How are secrets managed?", "What prevents hallucinations?"].map((item) => (
                    <button key={item} type="button" onClick={() => setQuestion(item)}>{item}</button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="turn">
                <div className="question-card">
                  <span>You</span>
                  <p>{lastQuestion}</p>
                </div>
                <div className="answer-card">
                  <div className="answer-label">
                    <span>RAG</span>
                    <span className={`run-state run-${status}`}>{status}</span>
                  </div>
                  {answer ? <p className="answer-text">{answer}</p> : busy ? <div className="thinking"><i /><i /><i /> Retrieving evidence</div> : null}
                  {queryError && <div className="inline-error" role="alert">{queryError}</div>}
                  {status === "cancelled" && <div className="inline-note">Generation stopped. The partial answer is preserved.</div>}
                  {(elapsed !== null || requestId) && (
                    <div className="answer-meta">
                      {elapsed !== null && <span>{elapsed.toFixed(2)}s</span>}
                      {requestId && <span title="Use this id to find the request in backend logs">request {requestId}</span>}
                    </div>
                  )}
                  {(status === "error" || status === "cancelled") && lastQuestion && (
                    <button className="text-action" type="button" onClick={() => void runQuery(lastQuestion)}>Retry query</button>
                  )}
                </div>

                {citations.length > 0 && (
                  <section className="citations" aria-label="Retrieved citations">
                    <div className="citation-heading">
                      <h3>Retrieved evidence</h3>
                      <span>{citations.length} passages</span>
                    </div>
                    <div className="citation-list">
                      {citations.map((citation, index) => (
                        <details className="citation-card" key={`${citation.source}-${citation.chunk_index}-${index}`}>
                          <summary>
                            <span className="citation-number">{index + 1}</span>
                            <span className="citation-source">{citation.source}<small>chunk {citation.chunk_index}</small></span>
                            <span className="score">{citation.score.toFixed(3)}</span>
                          </summary>
                          <p>{citation.text}</p>
                        </details>
                      ))}
                    </div>
                  </section>
                )}
              </div>
            )}
          </div>

          <form className="composer" onSubmit={submitQuestion}>
            <label className="sr-only" htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask a grounded question…"
              rows={3}
              disabled={busy}
            />
            <div className="composer-actions">
              <span>Enter to send · Shift+Enter for a new line</span>
              {busy ? (
                <button className="stop-button" type="button" onClick={() => abortRef.current?.abort()}>Stop</button>
              ) : (
                <button className="send-button" type="submit" disabled={!question.trim()}>Ask corpus <span aria-hidden="true">↗</span></button>
              )}
            </div>
          </form>
        </section>

        <aside className="control-panel" aria-label="RAG controls">
          <section className="control-section">
            <div className="panel-heading compact">
              <div><p className="eyebrow">Retrieval</p><h2>Search controls</h2></div>
              <span className="baseline-chip">vanilla default</span>
            </div>
            <label>Top K <output>{settings.topK}</output>
              <input type="range" min="1" max="12" value={settings.topK} onChange={(event) => setSettings({ ...settings, topK: Number(event.target.value) })} />
            </label>
            <label>Source filter
              <input type="text" placeholder="Exact filename (optional)" value={settings.sourceFilter} onChange={(event) => setSettings({ ...settings, sourceFilter: event.target.value })} />
            </label>
            <label>Temperature <output>{settings.temperature.toFixed(1)}</output>
              <input type="range" min="0" max="1" step="0.1" value={settings.temperature} onChange={(event) => setSettings({ ...settings, temperature: Number(event.target.value) })} />
            </label>

            <details className="advanced-controls">
              <summary>Advanced retrieval</summary>
              <label className="toggle-row"><span><strong>MMR</strong><small>Reduce redundant passages</small></span>
                <input type="checkbox" checked={settings.useMmr} onChange={(event) => setSettings({ ...settings, useMmr: event.target.checked })} />
              </label>
              {settings.useMmr && <label>MMR lambda <output>{settings.mmrLambda.toFixed(1)}</output>
                <input type="range" min="0" max="1" step="0.1" value={settings.mmrLambda} onChange={(event) => setSettings({ ...settings, mmrLambda: Number(event.target.value) })} />
              </label>}
              <label className="toggle-row"><span><strong>Multi-query</strong><small>Generate alternate phrasings</small></span>
                <input type="checkbox" checked={settings.multiQuery} onChange={(event) => setSettings({ ...settings, multiQuery: event.target.checked })} />
              </label>
              {settings.multiQuery && <label>Query variants <output>{settings.multiQueryCount}</output>
                <input type="range" min="1" max="6" value={settings.multiQueryCount} onChange={(event) => setSettings({ ...settings, multiQueryCount: Number(event.target.value) })} />
              </label>}
            </details>
          </section>

          <section className="control-section ingest-section">
            <div className="panel-heading compact"><div><p className="eyebrow">Knowledge</p><h2>Ingest a document</h2></div></div>
            <form onSubmit={submitIngest}>
              <label className="file-drop">
                <input ref={fileInputRef} type="file" accept=".pdf,.md,.txt" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
                <span aria-hidden="true">＋</span>
                <strong>{file ? file.name : "Choose PDF, Markdown, or text"}</strong>
                <small>{file ? `${(file.size / 1024).toFixed(1)} KB` : "Up to 25MB"}</small>
              </label>
              <div className="chunk-grid">
                <label>Chunk size<input type="number" min="100" max="8000" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} /></label>
                <label>Overlap<input type="number" min="0" max="2000" value={chunkOverlap} onChange={(event) => setChunkOverlap(Number(event.target.value))} /></label>
              </div>
              <button className="ingest-button" type="submit" disabled={ingesting}>{ingesting ? "Embedding…" : "Ingest document"}</button>
            </form>
            {ingestError && <div className="inline-error" role="alert">{ingestError}</div>}
            {ingestResult && (
              <div className="ingest-success" role="status">
                <strong>{ingestResult.source} is searchable</strong>
                <span>{ingestResult.chunks_written} chunks · {ingestResult.elapsed_seconds.toFixed(2)}s</span>
                <small>{ingestResult.embedding_model}</small>
              </div>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}
