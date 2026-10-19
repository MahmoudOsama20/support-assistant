import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const requestId = req.headers.get("x-request-id");

    const headers: Record<string, string> = {
      "Content-Type": "application/json; charset=utf-8",
      Accept: "application/json",
    };
    if (requestId) {
      headers["X-Request-ID"] = requestId;
    }

    const backendEndpoint = `${BACKEND_URL.replace(/\/$/, "")}/chat`;
    const backendRes = await fetch(backendEndpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      // 70s timeout for agent inference
      signal: AbortSignal.timeout(70000),
    });

    const data = await backendRes.json();
    const resHeaders: Record<string, string> = {};
    const backendRid = backendRes.headers.get("x-request-id");
    if (backendRid) {
      resHeaders["X-Request-ID"] = backendRid;
    }

    return NextResponse.json(data, {
      status: backendRes.status,
      headers: resHeaders,
    });
  } catch (error: unknown) {
    const err = error as { name?: string; message?: string };
    if (err.name === "TimeoutError") {
      return NextResponse.json(
        {
          type: "error",
          status: "error",
          code: "timeout",
          message: "The backend request timed out. Please try again.",
          retryable: true,
        },
        { status: 504 }
      );
    }

    return NextResponse.json(
      {
        type: "error",
        status: "error",
        code: "internal_error",
        message: "Failed to connect to Nile Wallet backend service. Is FastAPI running on " + BACKEND_URL + "?",
        retryable: true,
      },
      { status: 503 }
    );
  }
}
