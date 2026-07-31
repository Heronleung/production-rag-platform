# Frontend architecture (Phase 5)

The Phase 5 UI lives in `web/` and uses Next.js App Router with TypeScript.

## Why a same-origin proxy

A browser cannot use `EventSource` for this API because `/query` is a POST with a JSON body. The client therefore uses `fetch()` and incrementally reads the response stream. Next.js route handlers proxy to FastAPI so the browser stays on one origin and the backend URL remains server-only.

The query proxy returns `upstream.body` directly. It must not call `text()` or `json()` for a successful SSE response, because that would buffer the entire generation and defeat token streaming.

## Client state machine

`idle → connecting → streaming → complete`

Recoverable exits are `error` and `cancelled`. Only one active request is allowed. Stop aborts the browser fetch; the Next.js handler forwards the request signal to the FastAPI fetch. Retry starts a fresh request with the previous question and current controls.

## SSE parser

`web/lib/sse.ts` is independent of React and tested with network boundaries between every character. It supports LF/CRLF, comments, ids, multiple `data:` lines, and a final event without a blank terminator.

## Error model

- Non-2xx HTTP responses are parsed as FastAPI's `{detail, request_id}` shape.
- A failure after stream headers arrives as `event: error`.
- A stream that ends without `done` is treated as truncated.
- Request ids are shown with the answer so backend JSON logs can be correlated.

## Phase boundary

Phase 5 does not change retrieval ranking, the golden dataset, evaluation thresholds, or backend model configuration. Production image/build, Kubernetes networking, and CORS policy are Phase 6 concerns.
