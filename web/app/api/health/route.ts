import { backendUrl, passthrough, proxyFailure } from "../_proxy";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  try {
    return passthrough(await fetch(backendUrl("/healthz"), { cache: "no-store" }));
  } catch (error) {
    return proxyFailure(error);
  }
}
