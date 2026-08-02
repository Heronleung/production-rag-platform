export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export function GET(): Response {
  return Response.json(
    { status: "ok", service: "rag-web" },
    { headers: { "Cache-Control": "no-store" } },
  );
}
