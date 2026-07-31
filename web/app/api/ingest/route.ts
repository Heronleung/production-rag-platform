import {
  backendUrl,
  forwardedRequestId,
  passthrough,
  proxyFailure,
} from "../_proxy";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  try {
    const query = new URL(request.url).search;
    const upstream = await fetch(backendUrl(`/ingest${query}`), {
      method: "POST",
      headers: forwardedRequestId(request),
      body: await request.formData(),
      cache: "no-store",
      signal: request.signal,
    });
    return passthrough(upstream);
  } catch (error) {
    return proxyFailure(error);
  }
}
