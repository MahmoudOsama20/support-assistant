"use client";

import React, { useState, useEffect } from "react";
import { useChat } from "@/hooks/use-chat";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { ChatContainer } from "@/components/chat/chat-container";

export default function Home() {
  const {
    sessions,
    activeSession,
    messages,
    isLoading,
    isMounted,
    sendMessage,
    startNewChat,
    selectSession,
    deleteSession,
    clearAllSessions,
    retryMessage,
  } = useChat();

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isDevPanelOpen, setIsDevPanelOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("nile_theme");
      if (stored === "dark") {
        setIsDarkMode(true);
      } else if (stored === "light") {
        setIsDarkMode(false);
      } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        setIsDarkMode(true);
      }
    } catch {}
  }, []);

  // Synchronize theme class with documentElement
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  const handleToggleTheme = () => {
    setIsDarkMode((prev) => {
      const next = !prev;
      if (next) {
        document.documentElement.classList.add("dark");
        localStorage.setItem("nile_theme", "dark");
      } else {
        document.documentElement.classList.remove("dark");
        localStorage.setItem("nile_theme", "light");
      }
      return next;
    });
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-slate-50 dark:bg-slate-950">
      {/* Sidebar Navigation */}
      <Sidebar
        sessions={isMounted ? sessions : []}
        activeSessionId={activeSession?.id || null}
        onSelectSession={selectSession}
        onNewChat={startNewChat}
        onDeleteSession={deleteSession}
        onClearAll={clearAllSessions}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        isDarkMode={isDarkMode}
        onToggleTheme={handleToggleTheme}
      />

      {/* Main Conversation Column */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          onOpenSidebar={() => setIsSidebarOpen(true)}
          onNewChat={startNewChat}
          isDevPanelOpen={isDevPanelOpen}
          onToggleDevPanel={() => setIsDevPanelOpen((prev) => !prev)}
          activeSessionTitle={activeSession?.title}
        />

        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          onSendMessage={sendMessage}
          onRetry={retryMessage}
          showDevPanel={isDevPanelOpen}
        />
      </div>
    </div>
  );
}
