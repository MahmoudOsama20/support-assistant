"use client";

import React, { useEffect, useRef } from "react";
import { ChatMessage } from "@/types/chat";
import { UserMessage } from "./user-message";
import { AssistantMessage } from "./assistant-message";

interface MessageListProps {
  messages: ChatMessage[];
  onRetry: (query: string) => void;
  showDevPanel: boolean;
}

export function MessageList({ messages, onRetry, showDevPanel }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
      <div className="mx-auto max-w-4xl space-y-2">
        {messages.map((message) => {
          if (message.role === "user") {
            return <UserMessage key={message.id} message={message} />;
          }
          return (
            <AssistantMessage
              key={message.id}
              message={message}
              onRetry={onRetry}
              showDevPanel={showDevPanel}
            />
          );
        })}
        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
}
