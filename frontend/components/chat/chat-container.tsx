"use client";

import React, { useState } from "react";
import { ChatMessage } from "@/types/chat";
import { MessageList } from "./message-list";
import { WelcomeScreen } from "./welcome-screen";
import { MessageComposer } from "./message-composer";

interface ChatContainerProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (query: string) => void;
  onRetry: (query: string) => void;
  showDevPanel: boolean;
}

export function ChatContainer({
  messages,
  isLoading,
  onSendMessage,
  onRetry,
  showDevPanel,
}: ChatContainerProps) {
  const [selectedExample, setSelectedExample] = useState<string>("");

  const handleSelectPrompt = (prompt: string) => {
    setSelectedExample(prompt);
    // Directly send example prompt
    onSendMessage(prompt);
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-slate-50/50 dark:bg-slate-950/50">
      {messages.length === 0 ? (
        <div className="flex-1 overflow-y-auto flex items-center justify-center">
          <WelcomeScreen onSelectPrompt={handleSelectPrompt} />
        </div>
      ) : (
        <MessageList
          messages={messages}
          onRetry={onRetry}
          showDevPanel={showDevPanel}
        />
      )}

      <MessageComposer
        onSendMessage={onSendMessage}
        isLoading={isLoading}
        initialValue={selectedExample}
      />
    </div>
  );
}
