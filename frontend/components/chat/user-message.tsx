"use client";

import React from "react";
import { ChatMessage } from "@/types/chat";
import { formatDate } from "@/lib/utils";
import { useLanguage } from "@/lib/i18n/context";
import { User } from "lucide-react";

interface UserMessageProps {
  message: ChatMessage;
}

export function UserMessage({ message }: UserMessageProps) {
  const { language } = useLanguage();

  return (
    <div className="flex w-full justify-end py-2">
      <div className="flex max-w-[85%] md:max-w-[75%] items-start gap-2.5 flex-row-reverse">
        {/* Avatar */}
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white dark:bg-slate-700 shadow-sm">
          <User className="h-4 w-4" />
        </div>

        {/* Content Bubble */}
        <div className="flex flex-col items-end">
          <div className="rounded-2xl rounded-tr-sm bg-slate-900 px-4 py-2.5 text-sm text-white shadow-sm dark:bg-emerald-600">
            <p className="whitespace-pre-wrap break-words leading-relaxed">{message.content}</p>
          </div>
          <span className="mt-1 px-1 text-[11px] text-slate-400 dark:text-slate-500 font-mono">
            {formatDate(message.timestamp, language)}
          </span>
        </div>
      </div>
    </div>
  );
}
