"use client";

import React from "react";
import { useLanguage } from "@/lib/i18n/context";
import { ArrowRight, ArrowLeft, Sparkles, Database, BookOpen, AlertOctagon } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface WelcomeScreenProps {
  onSelectPrompt: (prompt: string) => void;
}

export function WelcomeScreen({ onSelectPrompt }: WelcomeScreenProps) {
  const { t, dir } = useLanguage();
  const Arrow = dir === "rtl" ? ArrowLeft : ArrowRight;

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6 text-center max-w-3xl mx-auto my-auto animate-in fade-in duration-300">
      {/* Brand Badge */}
      <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-200/80 bg-emerald-50/70 px-3 py-1.5 text-xs font-medium text-emerald-800 shadow-sm dark:border-emerald-800/40 dark:bg-emerald-950/40 dark:text-emerald-300">
        <Sparkles className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
        <span>Nile Wallet AI Support</span>
      </div>

      {/* Main Heading */}
      <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 dark:text-white leading-tight">
        {t.tagline}
      </h2>

      {/* Subtitle */}
      <p className="mt-3 max-w-xl text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed">
        {t.description}
      </p>

      {/* Example Prompt Grid */}
      <div className="mt-8 w-full text-start">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3 px-1">
          {t.examplesTitle}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {t.examples.map((item, idx) => (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.query)}
              className="group relative flex flex-col justify-between rounded-xl border border-slate-200/80 bg-white p-4 text-start shadow-xs transition-all hover:border-emerald-500/50 hover:bg-emerald-50/20 hover:shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:hover:border-emerald-500/40 dark:hover:bg-emerald-950/20"
            >
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                  {item.badge === "SQL" ? (
                    <Database className="h-3 w-3 text-blue-500" />
                  ) : item.badge === "RAG" ? (
                    <BookOpen className="h-3 w-3 text-emerald-500" />
                  ) : (
                    <AlertOctagon className="h-3 w-3 text-amber-500" />
                  )}
                  {item.category}
                </span>
                <Badge
                  variant={
                    item.badge === "SQL"
                      ? "info"
                      : item.badge === "RAG"
                      ? "default"
                      : item.badge === "Refusal"
                      ? "warning"
                      : "neutral"
                  }
                  size="sm"
                >
                  {item.badge}
                </Badge>
              </div>

              <p className="text-xs font-medium text-slate-800 dark:text-slate-200 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                &ldquo;{item.query}&rdquo;
              </p>

              <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 pt-2 border-t border-slate-100 dark:border-slate-800/80">
                <span>{item.label}</span>
                <Arrow className="h-3.5 w-3.5 transition-transform group-hover:translate-x-1 rtl:group-hover:-translate-x-1" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Grounding guarantee footnote */}
      <p className="mt-6 text-[11px] text-slate-400 dark:text-slate-500 max-w-lg">
        {t.disclaimer}
      </p>
    </div>
  );
}
