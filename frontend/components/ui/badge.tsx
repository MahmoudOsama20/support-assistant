import * as React from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info" | "neutral" | "outline";
  size?: "sm" | "md";
}

export function Badge({ className, variant = "default", size = "sm", children, ...props }: BadgeProps) {
  const base = "inline-flex items-center font-medium rounded-full select-none";

  const variants = {
    default: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300",
    success: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300",
    warning: "bg-amber-100 text-amber-900 dark:bg-amber-950/60 dark:text-amber-300",
    danger: "bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300",
    info: "bg-sky-100 text-sky-800 dark:bg-sky-950/60 dark:text-sky-300",
    neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    outline: "border border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-400",
  };

  const sizes = {
    sm: "px-2 py-0.5 text-xs gap-1",
    md: "px-2.5 py-1 text-xs font-semibold gap-1.5",
  };

  return (
    <span className={cn(base, variants[variant], sizes[size], className)} {...props}>
      {children}
    </span>
  );
}
