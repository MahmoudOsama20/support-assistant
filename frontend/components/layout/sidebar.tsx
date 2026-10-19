"use client";

import React from "react";
import { ConversationSession } from "@/types/chat";
import { useLanguage } from "@/lib/i18n/context";
import { ConnectionBadge } from "./connection-badge";
import { Button } from "@/components/ui/button";
import {
  Plus,
  MessageSquare,
  Trash2,
  Globe,
  Moon,
  Sun,
  X,
  ShieldCheck,
  Layers,
} from "lucide-react";

interface SidebarProps {
  sessions: ConversationSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onClearAll: () => void;
  isOpen: boolean;
  onClose: () => void;
  isDarkMode: boolean;
  onToggleTheme: () => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearAll,
  isOpen,
  onClose,
  isDarkMode,
  onToggleTheme,
}: SidebarProps) {
  const { t, language, toggleLanguage } = useLanguage();

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-sm lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 z-40 flex w-72 flex-col border-r border-slate-200/80 bg-slate-50/90 backdrop-blur-md transition-transform duration-200 dark:border-slate-800 dark:bg-slate-900/95 lg:static lg:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0 rtl:translate-x-full rtl:lg:translate-x-0"
        }`}
      >
        {/* Header / Brand */}
        <div className="flex h-16 items-center justify-between border-b border-slate-200/70 px-4 dark:border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white shadow-sm shadow-emerald-500/20">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <h1 className="font-semibold text-sm leading-tight text-slate-900 dark:text-white">
                {t.appTitle}
              </h1>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                {t.appSubtitle}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800 lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <Button
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className="w-full justify-start gap-2 shadow-sm"
            size="md"
          >
            <Plus className="h-4 w-4" />
            <span>{t.newChat}</span>
          </Button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto px-3 py-1 space-y-1">
          <div className="flex items-center justify-between px-2 py-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
            <span>{t.chatHistory}</span>
            {sessions.length > 0 && (
              <button
                onClick={onClearAll}
                className="text-[11px] font-normal text-slate-400 hover:text-rose-500 transition-colors"
                title={t.clearHistory}
              >
                {t.clearHistory}
              </button>
            )}
          </div>

          {sessions.length === 0 ? (
            <div className="px-3 py-8 text-center text-xs text-slate-400 dark:text-slate-500">
              <MessageSquare className="mx-auto mb-2 h-7 w-7 stroke-1 opacity-50" />
              <p>{t.noHistory}</p>
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  className={`group relative flex items-center justify-between rounded-lg px-3 py-2 text-xs transition-colors cursor-pointer ${
                    isActive
                      ? "bg-emerald-50 font-medium text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300"
                      : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800/60"
                  }`}
                  onClick={() => {
                    onSelectSession(session.id);
                    onClose();
                  }}
                >
                  <div className="flex items-center gap-2 truncate pr-2">
                    <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-60" />
                    <span className="truncate">{session.title}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-rose-500 transition-opacity"
                    aria-label="Delete chat"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Footer / Controls */}
        <div className="border-t border-slate-200/80 p-3 space-y-2 dark:border-slate-800">
          <div className="flex items-center justify-between px-1">
            <ConnectionBadge />
            <div className="flex items-center gap-1">
              {/* Language Switcher */}
              <button
                onClick={toggleLanguage}
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
                title={language === "en" ? "تبديل إلى العربية" : "Switch to English"}
              >
                <Globe className="h-3.5 w-3.5" />
                <span>{language === "en" ? "العربية" : "English"}</span>
              </button>

              {/* Theme Toggle */}
              <button
                onClick={onToggleTheme}
                className="rounded-lg border border-slate-200 p-1.5 text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
                aria-label="Toggle theme"
              >
                {isDarkMode ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-200/60 dark:border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 px-1">
            <div className="flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>E5 + BGE-M3 + Groq</span>
            </div>
            <a
              href="https://github.com/MahmoudOsama20/support-assistant"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
              title="View GitHub Repository"
            >
              <svg className="h-3.5 w-3.5 fill-current" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </aside>
    </>
  );
}
