import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { sendChatMessage, checkBackendHealth, ApiError } from "../lib/api/client";
import { RagAnswer, SqlAnswer, Clarification, Refusal } from "../lib/api/types";

describe("API Client", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("handles successful RAG answer response", async () => {
    const mockRag: RagAnswer = {
      type: "rag_answer",
      status: "success",
      tool: "rag",
      request_id: "req_test_123",
      language: "en",
      intent: "qa_currency",
      intent_confidence: 0.95,
      route: "kb_question",
      route_confidence: 0.99,
      answer: "The daily transfer limit for Basic accounts is 5,000 EGP.",
      citations: [
        {
          doc_id: "kb_003",
          chunk_id: "kb_003#c1",
          title: "Transfer limits",
          heading: "Daily limits",
          status: "active",
          snippet: "Basic accounts can transfer up to 5,000 EGP per day.",
        },
      ],
      metadata: {
        flags: { too_short: false },
        timings_ms: { total: 120, llm: 90 },
        degraded: false,
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ "x-request-id": "req_test_123" }),
      json: async () => mockRag,
    } as unknown as Response);

    const res = await sendChatMessage({ query: "What is the daily transfer limit?" });
    expect(res.type).toBe("rag_answer");
    if (res.type === "rag_answer") {
      expect(res.answer).toContain("5,000 EGP");
      expect(res.citations).toHaveLength(1);
      expect(res.citations[0].doc_id).toBe("kb_003");
    }
  });

  it("handles SQL answer response with columns and rows", async () => {
    const mockSql: SqlAnswer = {
      type: "sql_answer",
      status: "success",
      tool: "sql",
      request_id: "req_sql_456",
      language: "ar",
      intent: null,
      intent_confidence: null,
      route: "sql_query",
      route_confidence: 0.98,
      answer: "رصيد منى عادل هو 84,250.50 جنيه مصري.",
      sql: "SELECT balance FROM accounts WHERE customer_id = 1",
      columns: ["full_name", "balance", "currency"],
      rows: [["Mona Adel", 84250.5, "EGP"]],
      row_count: 1,
      truncated: false,
      metadata: { flags: {}, timings_ms: { total: 45 }, degraded: false },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => mockSql,
    } as unknown as Response);

    const res = await sendChatMessage({ query: "اعرض رصيد منى عادل" });
    expect(res.type).toBe("sql_answer");
    if (res.type === "sql_answer") {
      expect(res.columns).toEqual(["full_name", "balance", "currency"]);
      expect(res.rows[0][0]).toBe("Mona Adel");
      expect(res.row_count).toBe(1);
    }
  });

  it("handles clarification response", async () => {
    const mockClarify: Clarification = {
      type: "clarification",
      status: "clarify",
      tool: null,
      request_id: "req_clarify_789",
      language: "en",
      intent: null,
      intent_confidence: null,
      route: "none",
      route_confidence: 0,
      message: "Could you please clarify what transfer you are asking about?",
      metadata: { flags: { too_short: true }, timings_ms: { total: 10 }, degraded: false },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => mockClarify,
    } as unknown as Response);

    const res = await sendChatMessage({ query: "transfer" });
    expect(res.type).toBe("clarification");
    if (res.type === "clarification") {
      expect(res.message).toContain("clarify");
    }
  });

  it("handles refusal response", async () => {
    const mockRefusal: Refusal = {
      type: "refusal",
      status: "refused",
      reason: "out_of_scope",
      request_id: "req_refuse_001",
      language: "en",
      intent: "general_weather",
      intent_confidence: 0.9,
      route: "none",
      route_confidence: 0,
      message: "I can only answer questions related to Nile Wallet services.",
      metadata: { flags: {}, timings_ms: { total: 15 }, degraded: false },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: async () => mockRefusal,
    } as unknown as Response);

    const res = await sendChatMessage({ query: "What is the weather in Cairo?" });
    expect(res.type).toBe("refusal");
    if (res.type === "refusal") {
      expect(res.reason).toBe("out_of_scope");
    }
  });

  it("handles backend unreachable network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(sendChatMessage({ query: "hello" })).rejects.toThrow(ApiError);
  });

  it("checks backend health status correctly", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok", model_ready: true }),
    } as unknown as Response);

    const health = await checkBackendHealth();
    expect(health.status).toBe("ok");
    expect(health.model_ready).toBe(true);
  });
});
