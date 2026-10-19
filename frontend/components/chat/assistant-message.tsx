"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/types/chat";
import { formatDate } from "@/lib/utils";
import { useLanguage } from "@/lib/i18n/context";
import { CitationList } from "@/components/evidence/citation-list";
import { SqlTable } from "@/components/results/sql-table";
import { DevPanel } from "@/components/debug/dev-panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ShieldCheck,
  Copy,
  Check,
  RotateCcw,
  AlertCircle,
  HelpCircle,
  ShieldAlert,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";

interface AssistantMessageProps {
  message: ChatMessage;
  onRetry?: (query: string) => void;
  showDevPanel?: boolean;
}

export function AssistantMessage({ message, onRetry, showDevPanel = false }: AssistantMessageProps) {
  const { t, language } = useLanguage();
  const [copied, setCopied] = useState(false);
  const response = message.response;

  const handleCopy = async () => {
    const textToCopy =
      response?.type === "rag_answer" || response?.type === "sql_answer"
        ? response.answer
        : response?.type === "clarification" || response?.type === "refusal"
        ? response.message
        : message.content;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      toast.success(t.copied);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy");
    }
  };

  const renderContent = () => {
    // 1. Pending / Loading
    if (message.status === "pending") {
      return (
        <div className="flex items-center gap-2 py-3 text-sm text-slate-500 dark:text-slate-400">
          <div className="flex gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:-0.3s]" />
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:-0.15s]" />
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce" />
          </div>
          <span className="text-xs font-medium">{t.sending}</span>
        </div>
      );
    }

    // 2. Client-level or Network Error
    if (message.status === "error" && !response) {
      return (
        <div className="rounded-xl border border-rose-200/80 bg-rose-50/70 p-4 text-xs text-rose-900 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
          <div className="flex items-center gap-2 font-semibold">
            <AlertCircle className="h-4 w-4 text-rose-600 dark:text-rose-400" />
            <span>{t.errorTitle}</span>
          </div>
          <p className="mt-1.5 text-rose-800 dark:text-rose-300">
            {message.error || "Connection error. Please try again."}
          </p>
        </div>
      );
    }

    if (!response) {
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
      );
    }

    // 3. Structured RAG Answer
    if (response.type === "rag_answer") {
      return (
        <div>
          <div className="prose prose-sm dark:prose-invert max-w-none text-slate-800 dark:text-slate-200 leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
          </div>
          <CitationList citations={response.citations} />
        </div>
      );
    }

    // 4. Structured SQL Answer
    if (response.type === "sql_answer") {
      return (
        <div>
          <div className="prose prose-sm dark:prose-invert max-w-none text-slate-800 dark:text-slate-200 leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
          </div>
          <SqlTable
            columns={response.columns}
            rows={response.rows}
            rowCount={response.row_count}
            sql={response.sql}
            truncated={response.truncated}
          />
        </div>
      );
    }

    // 5. Clarification Request
    if (response.type === "clarification") {
      return (
        <div className="rounded-xl border border-amber-200/80 bg-amber-50/60 p-4 text-xs dark:border-amber-900/40 dark:bg-amber-950/30">
          <div className="flex items-center gap-2 font-semibold text-amber-900 dark:text-amber-200">
            <HelpCircle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            <span>{t.clarificationTitle}</span>
          </div>
          <p className="mt-2 text-sm text-amber-950 dark:text-amber-100 font-medium leading-relaxed">
            {response.message}
          </p>
          <p className="mt-2 text-[11px] text-amber-800/80 dark:text-amber-300/80">
            {t.clarificationDesc}
          </p>
        </div>
      );
    }

    // 6. Refusal / Out of scope
    if (response.type === "refusal") {
      const reasonLabel = t.refusalReasons[response.reason] || response.reason;
      return (
        <div className="rounded-xl border border-slate-200/90 bg-slate-50/80 p-4 text-xs dark:border-slate-800 dark:bg-slate-900/70">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 font-semibold text-slate-900 dark:text-slate-100">
              <ShieldAlert className="h-4 w-4 text-slate-500 dark:text-slate-400" />
              <span>{t.refusalTitle}</span>
            </div>
            <Badge variant="neutral" size="sm">
              {response.reason}
            </Badge>
          </div>
          <p className="mt-2 text-sm text-slate-800 dark:text-slate-200 leading-relaxed">
            {response.message}
          </p>
          <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
            {reasonLabel}
          </p>
        </div>
      );
    }

    // 7. Error response from FastAPI
    if (response.type === "error") {
      return (
        <div className="rounded-xl border border-rose-200/80 bg-rose-50/70 p-4 text-xs dark:border-rose-900/50 dark:bg-rose-950/40">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-semibold text-rose-900 dark:text-rose-200">
              <AlertTriangle className="h-4 w-4 text-rose-600 dark:text-rose-400" />
              <span>{t.errorTitle}</span>
            </div>
            <Badge variant="danger" size="sm">
              {response.code}
            </Badge>
          </div>
          <p className="mt-2 text-sm text-rose-950 dark:text-rose-100">{response.message}</p>
          {response.retryable && onRetry && (
            <div className="mt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => onRetry(message.content)}
                className="gap-1.5 border-rose-300 text-rose-800 hover:bg-rose-100 dark:border-rose-800 dark:text-rose-300"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>{t.retry}</span>
              </Button>
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="flex w-full justify-start py-3">
      <div className="flex max-w-[92%] md:max-w-[85%] items-start gap-3">
        {/* Assistant Avatar */}
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white shadow-sm shadow-emerald-500/20">
          <ShieldCheck className="h-4 w-4" />
        </div>

        {/* Message Container */}
        <div className="flex-1 min-w-0">
          <div className="rounded-2xl rounded-tl-sm border border-slate-200/80 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            {renderContent()}

            {/* Actions & Timings */}
            {message.status !== "pending" && (
              <div className="mt-3 flex items-center justify-between border-t border-slate-100 pt-2.5 dark:border-slate-800/60 text-[11px] text-slate-400">
                <div className="flex items-center gap-2">
                  <span>{formatDate(message.timestamp, language)}</span>
                  {message.latencyMs !== undefined && (
                    <span className="font-mono">({Math.round(message.latencyMs)} ms)</span>
                  )}
                </div>

                <div className="flex items-center gap-1">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1 rounded-md px-2 py-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800 dark:hover:text-slate-200 transition-colors"
                    title={t.copyAnswer}
                  >
                    {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                    <span>{copied ? t.copied : t.copyAnswer}</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Optional Developer Details */}
          {showDevPanel && response && (
            <DevPanel response={response} clientLatencyMs={message.latencyMs} />
          )}
        </div>
      </div>
    </div>
  );
}
