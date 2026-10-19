"use client";

import React, { useState, useRef, useEffect } from "react";
import { useLanguage } from "@/lib/i18n/context";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";

interface MessageComposerProps {
  onSendMessage: (query: string) => void;
  isLoading: boolean;
  initialValue?: string;
}

export function MessageComposer({ onSendMessage, isLoading, initialValue = "" }: MessageComposerProps) {
  const { t } = useLanguage();
  const [query, setQuery] = useState(initialValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (initialValue) {
      const timer = setTimeout(() => {
        setQuery(initialValue);
        textareaRef.current?.focus();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [initialValue]);

  // Auto-resize textarea height
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const nextHeight = Math.min(textarea.scrollHeight, 180);
    textarea.style.height = `${nextHeight}px`;
  }, [query]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isLoading) return;

    onSendMessage(trimmed);
    setQuery("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const charCount = query.length;
  const isTooLong = charCount > 2000;
  const canSubmit = charCount > 0 && !isTooLong && !isLoading;

  return (
    <div className="w-full border-t border-slate-200/80 bg-white/90 p-4 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/90">
      <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
        <div className="relative flex flex-col rounded-2xl border border-slate-200/90 bg-white shadow-xs focus-within:border-emerald-500/60 focus-within:ring-2 focus-within:ring-emerald-500/20 dark:border-slate-800 dark:bg-slate-950/70 transition-all">
          <textarea
            ref={textareaRef}
            rows={1}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t.inputPlaceholder}
            disabled={isLoading}
            className="w-full resize-none bg-transparent px-4 py-3.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none dark:text-slate-100 dark:placeholder:text-slate-500 leading-relaxed max-h-44 overflow-y-auto"
          />

          <div className="flex items-center justify-between px-3 py-2 border-t border-slate-100 dark:border-slate-900 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <span className={`text-[11px] font-mono ${isTooLong ? "text-rose-500 font-semibold" : ""}`}>
                {charCount}/2000 {t.charCount}
              </span>
              <span className="hidden sm:inline text-slate-300 dark:text-slate-600">|</span>
              <span className="hidden sm:inline text-[11px] text-slate-400">
                Enter ↵ {t.send} • Shift+Enter {t.inputPlaceholder ? "" : ""}
              </span>
            </div>

            <Button
              type="submit"
              size="sm"
              disabled={!canSubmit}
              isLoading={isLoading}
              className="gap-1.5 px-3 py-1 text-xs shadow-xs"
              title={t.send}
            >
              <span>{t.send}</span>
              <Send className="h-3.5 w-3.5 rtl:rotate-180" />
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
}
