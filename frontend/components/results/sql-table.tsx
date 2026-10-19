"use client";

import React, { useState } from "react";
import { Copy, Check, Database, Code2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatAsCsv } from "./csv-export";
import { useLanguage } from "@/lib/i18n/context";
import { toast } from "sonner";

interface SqlTableProps {
  columns: string[];
  rows: (string | number | boolean | null)[][];
  rowCount: number;
  sql?: string;
  truncated?: boolean;
}

export function SqlTable({ columns, rows, rowCount, sql, truncated }: SqlTableProps) {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);
  const [showSql, setShowSql] = useState(false);

  const handleCopyCsv = async () => {
    try {
      const csv = formatAsCsv(columns, rows);
      await navigator.clipboard.writeText(csv);
      setCopied(true);
      toast.success(t.csvCopied);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy CSV");
    }
  };

  if (!columns || columns.length === 0) {
    return (
      <div className="mt-4 rounded-xl border border-slate-200/80 bg-slate-50 p-4 text-center text-xs text-slate-500 dark:border-slate-800 dark:bg-slate-900/60">
        <Database className="mx-auto mb-1.5 h-6 w-6 opacity-40" />
        <p>{t.noRowsReturned}</p>
      </div>
    );
  }

  return (
    <div className="mt-4 overflow-hidden rounded-xl border border-slate-200/80 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      {/* Table Header Bar */}
      <div className="flex items-center justify-between border-b border-slate-200/80 bg-slate-50/80 px-4 py-2.5 dark:border-slate-800 dark:bg-slate-800/40">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
          <Database className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <span>{t.sqlResultsTitle}</span>
          <span className="rounded-full bg-slate-200/70 px-2 py-0.5 text-[10px] font-normal text-slate-700 dark:bg-slate-700 dark:text-slate-300">
            {rowCount} {t.rowCount}
            {truncated && " (capped at 50 max)"}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {sql && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSql(!showSql)}
              className="h-7 text-xs px-2 gap-1"
              title="Toggle executed query metadata"
            >
              <Code2 className="h-3.5 w-3.5" />
              <span className="text-[11px]">SQL</span>
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={handleCopyCsv}
            className="h-7 text-xs px-2.5 gap-1.5"
            title={t.copyCsv}
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
            <span className="text-[11px]">{copied ? t.copied : t.copyCsv}</span>
          </Button>
        </div>
      </div>

      {/* Expandable Developer SQL view */}
      {showSql && sql && (
        <div className="border-b border-slate-200/80 bg-slate-900 p-3 text-emerald-400 font-mono text-xs overflow-x-auto dark:border-slate-800">
          <p className="text-[10px] text-slate-400 mb-1 uppercase font-sans tracking-wide">
            {t.sqlQuery}
          </p>
          <code>{sql}</code>
        </div>
      )}

      {/* Responsive Table Scroll Container */}
      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        <table className="w-full text-start text-xs border-collapse">
          <thead className="sticky top-0 bg-slate-100/90 backdrop-blur-sm dark:bg-slate-800/90 text-slate-700 dark:text-slate-300 font-medium border-b border-slate-200 dark:border-slate-700">
            <tr>
              {columns.map((col, idx) => (
                <th
                  key={idx}
                  className="px-3.5 py-2.5 text-start font-semibold whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-6 text-center text-slate-400 italic"
                >
                  {t.noRowsReturned}
                </td>
              </tr>
            ) : (
              rows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                >
                  {row.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className="px-3.5 py-2 whitespace-nowrap text-slate-800 dark:text-slate-200"
                    >
                      {cell === null || cell === undefined ? (
                        <span className="text-slate-400 italic text-[11px]">NULL</span>
                      ) : typeof cell === "boolean" ? (
                        <span className="font-mono text-xs">{cell ? "true" : "false"}</span>
                      ) : (
                        String(cell)
                      )}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
