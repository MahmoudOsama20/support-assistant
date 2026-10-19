export type Language = "ar" | "en";
export type Tool = "rag" | "sql";
export type RefusalReason = "no_evidence" | "out_of_scope" | "unsafe_request" | "policy";
export type ErrorCode =
  | "timeout"
  | "llm_unavailable"
  | "llm_error"
  | "internal_error"
  | "model_error"
  | "invalid_response";

export interface Citation {
  doc_id: string;
  chunk_id: string;
  title: string;
  heading?: string | null;
  status?: string | null;
  snippet: string;
}

export interface Metadata {
  flags: Record<string, boolean>;
  timings_ms: Record<string, number>;
  degraded: boolean;
}

export interface BaseResponse {
  request_id: string;
  language: Language;
  intent: string | null;
  intent_confidence: number | null;
  route: string;
  route_confidence: number;
  metadata: Metadata;
}

export interface RagAnswer extends BaseResponse {
  type: "rag_answer";
  status: "success";
  tool: "rag";
  answer: string;
  citations: Citation[];
}

export interface SqlAnswer extends BaseResponse {
  type: "sql_answer";
  status: "success";
  tool: "sql";
  answer: string;
  sql: string;
  columns: string[];
  rows: (string | number | boolean | null)[][];
  row_count: number;
  truncated: boolean;
}

export interface Clarification extends BaseResponse {
  type: "clarification";
  status: "clarify";
  message: string;
  tool?: Tool | null;
}

export interface Refusal extends BaseResponse {
  type: "refusal";
  status: "refused";
  reason: RefusalReason;
  message: string;
  tool?: Tool | null;
  sql?: string | null;
  row_count?: number | null;
}

export interface ErrorResponse extends BaseResponse {
  type: "error";
  status: "error";
  code: ErrorCode;
  message: string;
  retryable: boolean;
}

export type ChatResponse =
  | RagAnswer
  | SqlAnswer
  | Clarification
  | Refusal
  | ErrorResponse;

export interface ChatRequest {
  query: string;
}

export interface HealthResponse {
  status: "ok" | "starting" | string;
  model_ready: boolean;
}

export interface ApiClientConfig {
  baseUrl?: string;
  timeoutMs?: number;
}
