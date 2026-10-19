import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const backendEndpoint = `${BACKEND_URL.replace(/\/$/, "")}/health`;
    const backendRes = await fetch(backendEndpoint, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(5000),
      cache: "no-store",
    });

    if (!backendRes.ok) {
      return NextResponse.json({ status: "degraded", model_ready: false }, { status: 503 });
    }

    const data = await backendRes.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: "unavailable", model_ready: false, backend_url: BACKEND_URL },
      { status: 503 }
    );
  }
}
