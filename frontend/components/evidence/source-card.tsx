"use client";

import React, { useState } from "react";
import { Citation } from "@/lib/api/types";
import { ChevronDown, ChevronUp, Hash, Tag } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface SourceCardProps {
  citation: Citation;
  index: number;
}

export function SourceCard({ citation, index }: SourceCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-3.5 text-xs transition-all hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60 dark:hover:bg-slate-900">
      <div
        className="flex items-start justify-between cursor-pointer gap-2"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 font-semibold text-[11px]">
            {index + 1}
          </div>
          <div className="min-w-0">
            <h4 className="font-semibold text-slate-900 dark:text-slate-100 truncate">
              {citation.title || citation.doc_id}
            </h4>
            {citation.heading && (
              <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                {citation.heading}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <Badge variant="outline" size="sm">
            {citation.doc_id}
          </Badge>
          <button
            className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
            aria-label={isExpanded ? "Collapse evidence" : "Expand evidence"}
          >
            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Snippet / Excerpt */}
      {citation.snippet && (
        <div className="mt-2.5 rounded-lg border border-slate-200/60 bg-white p-2.5 text-slate-700 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-300 font-mono text-[11px] leading-relaxed">
          <p className="line-clamp-3 italic before:content-['“'] after:content-['”']">
            {citation.snippet}
          </p>
        </div>
      )}

      {/* Expanded Metadata */}
      {isExpanded && (
        <div className="mt-3 pt-2.5 border-t border-slate-200/60 dark:border-slate-800/60 space-y-1.5 text-[11px] text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-2">
            <Hash className="h-3 w-3 opacity-70" />
            <span>Chunk:</span>
            <code className="text-slate-700 dark:text-slate-300 font-mono">{citation.chunk_id}</code>
          </div>
          {citation.status && (
            <div className="flex items-center gap-2">
              <Tag className="h-3 w-3 opacity-70" />
              <span>Status:</span>
              <span className="capitalize text-emerald-600 dark:text-emerald-400 font-medium">
                {citation.status}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
