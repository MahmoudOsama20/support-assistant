"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";

export function TypingIndicator() {
  return (
    <div className="flex w-full justify-start py-3">
      <div className="flex max-w-[80%] items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white shadow-sm shadow-emerald-500/20">
          <ShieldCheck className="h-4 w-4" />
        </div>
        <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-slate-200/80 bg-white px-4 py-3 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:-0.3s]" />
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:-0.15s]" />
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-bounce" />
        </div>
      </div>
    </div>
  );
}
