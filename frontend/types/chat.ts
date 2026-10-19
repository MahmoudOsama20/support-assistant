import { ChatResponse } from "@/lib/api/types";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  response?: ChatResponse;
  status?: "pending" | "success" | "error";
  error?: string;
  latencyMs?: number;
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}
