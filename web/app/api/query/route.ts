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
    const upstream = await fetch(backendUrl("/query"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...forwardedRequestId(request),
      },
      body: await request.text(),
      cache: "no-store",
      signal: request.signal,
    });
    return passthrough(upstream);
  } catch (error) {
    return proxyFailure(error);
  }
}
