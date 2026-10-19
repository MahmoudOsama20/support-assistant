"use client";

import React from "react";
import { Citation } from "@/lib/api/types";
import { SourceCard } from "./source-card";
import { BookOpen } from "lucide-react";
import { useLanguage } from "@/lib/i18n/context";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  const { t } = useLanguage();

  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-4 rounded-xl border border-slate-200/80 bg-white/60 p-4 dark:border-slate-800 dark:bg-slate-900/40">
      <div className="flex items-center gap-2 mb-3 text-xs font-semibold text-slate-700 dark:text-slate-300">
        <BookOpen className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
        <span>{t.sourcesTitle} ({citations.length})</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {citations.map((citation, index) => (
          <SourceCard key={citation.chunk_id || index} citation={citation} index={index} />
        ))}
      </div>
    </div>
  );
}
