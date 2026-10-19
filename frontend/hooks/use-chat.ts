"use client";

import { useState, useCallback, useEffect } from "react";
import { ChatMessage, ConversationSession } from "@/types/chat";
import { sendChatMessage, ApiError } from "@/lib/api/client";
import { ChatResponse } from "@/lib/api/types";
import { toast } from "sonner";

const STORAGE_KEY = "nile_support_sessions_v1";
const ACTIVE_SESSION_KEY = "nile_support_active_id_v1";

function createNewSessionId(): string {
  return "session_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now();
}

function createMessageId(): string {
  return "msg_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now();
}

export function useChat() {
  const [sessions, setSessions] = useState<ConversationSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isMounted, setIsMounted] = useState<boolean>(false);

  useEffect(() => {
    setIsMounted(true);
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as ConversationSession[];
        setSessions(parsed);
        const storedActive = localStorage.getItem(ACTIVE_SESSION_KEY);
        if (storedActive && parsed.some((s) => s.id === storedActive)) {
          setActiveSessionId(storedActive);
        } else if (parsed.length > 0) {
          setActiveSessionId(parsed[0].id);
        }
      }
    } catch {}
  }, []);

  // Save sessions to localStorage on change
  const persistSessions = (newSessions: ConversationSession[]) => {
    setSessions(newSessions);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newSessions));
    } catch {
      // Ignore storage quota
    }
  };

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;
  const messages = activeSession ? activeSession.messages : [];

  const startNewChat = useCallback(() => {
    const newId = createNewSessionId();
    const newSession: ConversationSession = {
      id: newId,
      title: "New Conversation",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    };
    persistSessions([newSession, ...sessions]);
    setActiveSessionId(newId);
    try {
      localStorage.setItem(ACTIVE_SESSION_KEY, newId);
    } catch {
      // Ignore
    }
  }, [sessions]);

  const selectSession = useCallback((id: string) => {
    setActiveSessionId(id);
    try {
      localStorage.setItem(ACTIVE_SESSION_KEY, id);
    } catch {
      // Ignore
    }
  }, []);

  const deleteSession = useCallback(
    (id: string) => {
      const filtered = sessions.filter((s) => s.id !== id);
      persistSessions(filtered);
      if (activeSessionId === id) {
        const nextId = filtered.length > 0 ? filtered[0].id : null;
        setActiveSessionId(nextId);
        if (nextId) localStorage.setItem(ACTIVE_SESSION_KEY, nextId);
        else localStorage.removeItem(ACTIVE_SESSION_KEY);
      }
    },
    [sessions, activeSessionId]
  );

  const clearAllSessions = useCallback(() => {
    persistSessions([]);
    setActiveSessionId(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(ACTIVE_SESSION_KEY);
    } catch {
      // Ignore
    }
  }, []);

  const sendMessage = useCallback(
    async (queryText: string) => {
      const cleanQuery = queryText.trim();
      if (!cleanQuery || isLoading) return;

      let currentId = activeSessionId;
      let sessionList = [...sessions];

      // If no active session, create one
      if (!currentId || !sessionList.some((s) => s.id === currentId)) {
        currentId = createNewSessionId();
        const initialSession: ConversationSession = {
          id: currentId,
          title: cleanQuery.slice(0, 36) + (cleanQuery.length > 36 ? "..." : ""),
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          messages: [],
        };
        sessionList = [initialSession, ...sessionList];
        setActiveSessionId(currentId);
      }

      const userMsg: ChatMessage = {
        id: createMessageId(),
        role: "user",
        content: cleanQuery,
        timestamp: new Date().toISOString(),
      };

      const pendingAssistantMsg: ChatMessage = {
        id: createMessageId(),
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        status: "pending",
      };

      // Append messages
      sessionList = sessionList.map((s) => {
        if (s.id === currentId) {
          const updatedTitle =
            s.messages.length === 0
              ? cleanQuery.slice(0, 36) + (cleanQuery.length > 36 ? "..." : "")
              : s.title;
          return {
            ...s,
            title: updatedTitle,
            updatedAt: new Date().toISOString(),
            messages: [...s.messages, userMsg, pendingAssistantMsg],
          };
        }
        return s;
      });

      persistSessions(sessionList);
      setIsLoading(true);

      const t0 = performance.now();

      try {
        const response: ChatResponse = await sendChatMessage({ query: cleanQuery });
        const latencyMs = performance.now() - t0;

        // Update assistant message with live backend result
        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === currentId) {
              const updatedMessages = s.messages.map((m) => {
                if (m.id === pendingAssistantMsg.id) {
                  return {
                    ...m,
                    status: "success" as const,
                    response,
                    latencyMs,
                  };
                }
                return m;
              });
              return { ...s, messages: updatedMessages, updatedAt: new Date().toISOString() };
            }
            return s;
          });
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch {}
          return next;
        });
      } catch (error: unknown) {
        const latencyMs = performance.now() - t0;
        let errorMessage = "An error occurred while contacting the assistant.";

        if (error instanceof ApiError) {
          errorMessage = error.message;
        } else if (error instanceof Error) {
          errorMessage = error.message;
        }

        toast.error(errorMessage);

        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === currentId) {
              const updatedMessages = s.messages.map((m) => {
                if (m.id === pendingAssistantMsg.id) {
                  return {
                    ...m,
                    status: "error" as const,
                    error: errorMessage,
                    latencyMs,
                  };
                }
                return m;
              });
              return { ...s, messages: updatedMessages, updatedAt: new Date().toISOString() };
            }
            return s;
          });
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch {}
          return next;
        });
      } finally {
        setIsLoading(false);
      }
    },
    [activeSessionId, sessions, isLoading]
  );

  const retryMessage = useCallback(
    (queryText: string) => {
      sendMessage(queryText);
    },
    [sendMessage]
  );

  return {
    sessions,
    activeSession,
    activeSessionId,
    messages,
    isLoading,
    isMounted,
    sendMessage,
    startNewChat,
    selectSession,
    deleteSession,
    clearAllSessions,
    retryMessage,
  };
}
