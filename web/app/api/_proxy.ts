const DEFAULT_BACKEND = "http://127.0.0.1:8000";

export function backendUrl(path: string): string {
  const base = (process.env.RAG_API_URL || DEFAULT_BACKEND).replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export function passthrough(upstream: Response): Response {
  const headers = new Headers();
  for (const name of ["content-type", "cache-control", "x-request-id", "x-accel-buffering"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("Cache-Control", "no-store");
  return new Response(upstream.body, { status: upstream.status, headers });
}

export function proxyFailure(error: unknown): Response {
  console.error("RAG backend proxy failed", error);
  return Response.json(
    { detail: "The RAG backend is unavailable.", request_id: null },
    { status: 502, headers: { "Cache-Control": "no-store" } },
  );
}

export function forwardedRequestId(request: Request): Record<string, string> {
  const value = request.headers.get("x-request-id");
  return value ? { "X-Request-ID": value } : {};
}
