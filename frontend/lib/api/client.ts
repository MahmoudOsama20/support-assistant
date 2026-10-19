import { ChatRequest, ChatResponse, HealthResponse } from "./types";

const DEFAULT_TIMEOUT_MS = 65000;

export class ApiError extends Error {
  code: string;
  statusCode?: number;
  retryable: boolean;
  requestId?: string;

  constructor(message: string, code = "api_error", statusCode?: number, retryable = false, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.statusCode = statusCode;
    this.retryable = retryable;
    this.requestId = requestId;
  }
}

/**
 * Determines the API base URL:
 * - If running in browser and NEXT_PUBLIC_API_BASE_URL is set, uses that, OR defaults to relative /api proxy
 * - Next.js proxy routes (/api/chat, /api/health) avoid CORS issues and protect secrets
 */
export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    // When using Next.js proxy route, use same-origin relative path /api
    return process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
  }
  return process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
}

export async function sendChatMessage(
  request: ChatRequest,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<ChatResponse> {
  const baseUrl = getApiBaseUrl();
  const url = baseUrl.endsWith("/") ? `${baseUrl}chat` : `${baseUrl}/chat`;
  const timeoutMs = options?.timeoutMs || DEFAULT_TIMEOUT_MS;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Link caller signal if provided
  if (options?.signal) {
    options.signal.addEventListener("abort", () => controller.abort());
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        Accept: "application/json",
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    const responseRequestId = response.headers.get("x-request-id") || undefined;
    let payload: unknown;

    try {
      payload = await response.json();
    } catch {
      throw new ApiError(
        "Invalid response received from server.",
        "invalid_response",
        response.status,
        response.status >= 500,
        responseRequestId
      );
    }

    // The backend returns ErrorResponse JSON even on non-200 with matching schema
    if (typeof payload === "object" && payload !== null && "type" in payload) {
      return payload as ChatResponse;
    }

    if (!response.ok) {
      const errObj = payload as Record<string, unknown> | null;
      const msg = typeof errObj?.message === "string" ? errObj.message : `Server returned error (${response.status})`;
      const code = typeof errObj?.code === "string" ? errObj.code : "server_error";
      throw new ApiError(msg, code, response.status, response.status >= 500, responseRequestId);
    }

    throw new ApiError("Malformed response shape from backend", "invalid_response", response.status, false, responseRequestId);
  } catch (error: unknown) {
    if (error instanceof ApiError) {
      throw error;
    }

    const err = error as { name?: string; message?: string };
    if (err.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.", "timeout", 504, true);
    }

    // Network error or backend unreachable
    throw new ApiError(
      "Unable to connect to the backend service. Please check if the FastAPI server is running.",
      "backend_unavailable",
      0,
      true
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl();
  const url = baseUrl.endsWith("/") ? `${baseUrl}health` : `${baseUrl}/health`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return { status: "degraded", model_ready: false };
    }

    const data = (await response.json()) as HealthResponse;
    return data;
  } catch {
    return { status: "unavailable", model_ready: false };
  }
}
