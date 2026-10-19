"use client";

import React, { useState } from "react";
import { ChatResponse } from "@/lib/api/types";
import { formatMs } from "@/lib/utils";
import { useLanguage } from "@/lib/i18n/context";
import { Terminal, Copy, Check, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/dialog";
import { toast } from "sonner";

interface DevPanelProps {
  response?: ChatResponse;
  clientLatencyMs?: number;
}

export function DevPanel({ response, clientLatencyMs }: DevPanelProps) {
  const { t } = useLanguage();
  const [copiedId, setCopiedId] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  if (!response) return null;

  const handleCopyRequestId = async () => {
    try {
      await navigator.clipboard.writeText(response.request_id);
      setCopiedId(true);
      toast.success(t.copied);
      setTimeout(() => setCopiedId(false), 2000);
    } catch {
      // Ignore
    }
  };

  const timings = response.metadata?.timings_ms || {};
  const flags = response.metadata?.flags || {};
  const degraded = Boolean(response.metadata?.degraded);

  return (
    <>
      <div className="mt-3 rounded-xl border border-slate-200/80 bg-slate-900 text-slate-100 p-3.5 text-xs font-mono dark:border-slate-800">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Terminal className="h-3.5 w-3.5 text-emerald-400" />
            <span className="font-semibold text-slate-200">{t.devPanelTitle}</span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsModalOpen(true)}
              className="h-6 text-[11px] px-2 py-0 border-slate-700 text-slate-300 hover:bg-slate-800"
            >
              JSON
            </Button>
          </div>
        </div>

        {/* Primary Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2.5">
          {/* Request ID */}
          <div className="rounded bg-slate-800/80 p-2">
            <span className="text-[10px] text-slate-400 block">{t.requestId}</span>
            <div className="flex items-center justify-between gap-1 mt-0.5">
              <span className="truncate text-emerald-400 text-[11px]">
                {response.request_id ? `${response.request_id.slice(0, 10)}...` : "—"}
              </span>
              {response.request_id && (
                <button
                  onClick={handleCopyRequestId}
                  className="text-slate-400 hover:text-white"
                  title="Copy Request ID"
                >
                  {copiedId ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                </button>
              )}
            </div>
          </div>

          {/* Route & Confidence */}
          <div className="rounded bg-slate-800/80 p-2">
            <span className="text-[10px] text-slate-400 block">{t.route}</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="font-semibold text-white">{response.route}</span>
              {response.route_confidence !== undefined && (
                <span className="text-[10px] text-slate-400">
                  ({(response.route_confidence * 100).toFixed(0)}%)
                </span>
              )}
            </div>
          </div>

          {/* Intent (Part A) */}
          <div className="rounded bg-slate-800/80 p-2">
            <span className="text-[10px] text-slate-400 block">{t.intent}</span>
            <div className="truncate text-slate-300 mt-0.5 text-[11px]">
              {response.intent || "—"}
              {response.intent_confidence !== null && response.intent_confidence !== undefined && (
                <span className="text-[10px] text-slate-400 ml-1">
                  ({(response.intent_confidence * 100).toFixed(0)}%)
                </span>
              )}
            </div>
          </div>

          {/* Timings */}
          <div className="rounded bg-slate-800/80 p-2">
            <span className="text-[10px] text-slate-400 block">{t.timings}</span>
            <span className="text-emerald-400 font-semibold mt-0.5 block">
              {formatMs(timings.total || clientLatencyMs)}
            </span>
          </div>
        </div>

        {/* Degraded mode warning if active */}
        {degraded && (
          <div className="mt-2 flex items-center gap-1.5 rounded bg-amber-950/50 border border-amber-800/50 p-2 text-amber-300 text-[11px]">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
            <span>{t.degradedMode}</span>
          </div>
        )}

        {/* Detailed Timings / Flags */}
        <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
          {Object.entries(timings).map(([key, val]) => (
            <div key={key}>
              <span className="text-slate-500">{key}:</span>{" "}
              <span className="text-slate-300">{formatMs(val)}</span>
            </div>
          ))}

          {Object.entries(flags).map(([key, val]) => (
            <div key={key}>
              <span className="text-slate-500">{key}:</span>{" "}
              <span className={val ? "text-amber-400" : "text-slate-400"}>{String(val)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Raw JSON Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={t.rawJson}
        description={`Request ID: ${response.request_id}`}
        maxWidth="2xl"
      >
        <div className="relative">
          <pre className="max-h-[60vh] overflow-auto rounded-xl bg-slate-950 p-4 font-mono text-xs text-emerald-400">
            {JSON.stringify(response, null, 2)}
          </pre>
          <div className="mt-3 flex justify-end">
            <Button
              size="sm"
              variant="outline"
              onClick={async () => {
                await navigator.clipboard.writeText(JSON.stringify(response, null, 2));
                toast.success(t.copied);
              }}
            >
              <Copy className="h-3.5 w-3.5 mr-1" />
              Copy JSON
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}
