"use client";

import React, { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api/client";
import { useLanguage } from "@/lib/i18n/context";
import { AlertCircle, CheckCircle2, RefreshCw } from "lucide-react";
import { Modal } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

export function ConnectionBadge() {
  const { t } = useLanguage();
  const [status, setStatus] = useState<"checking" | "connected" | "disconnected">("checking");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const handleManualCheck = async () => {
    setStatus("checking");
    try {
      const res = await checkBackendHealth();
      setStatus(res.status === "ok" ? "connected" : "disconnected");
    } catch {
      setStatus("disconnected");
    } finally {
      setLastChecked(new Date());
    }
  };

  useEffect(() => {
    let isMounted = true;

    async function check() {
      try {
        const res = await checkBackendHealth();
        if (isMounted) {
          setStatus(res.status === "ok" ? "connected" : "disconnected");
          setLastChecked(new Date());
        }
      } catch {
        if (isMounted) {
          setStatus("disconnected");
          setLastChecked(new Date());
        }
      }
    }

    check();
    const interval = setInterval(check, 45000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const badgeConfig = {
    connected: {
      dot: "bg-emerald-500 ring-emerald-500/30",
      text: t.connected,
      icon: CheckCircle2,
      textColor: "text-emerald-700 dark:text-emerald-300",
      bg: "bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800/40",
    },
    checking: {
      dot: "bg-amber-400 ring-amber-400/30 animate-pulse",
      text: t.checking,
      icon: RefreshCw,
      textColor: "text-amber-700 dark:text-amber-300",
      bg: "bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800/40",
    },
    disconnected: {
      dot: "bg-rose-500 ring-rose-500/30",
      text: t.disconnected,
      icon: AlertCircle,
      textColor: "text-rose-700 dark:text-rose-300",
      bg: "bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800/40",
    },
  }[status];

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className={`group flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium transition-all ${badgeConfig.bg} ${badgeConfig.textColor} hover:opacity-90`}
        title={lastChecked ? `Last checked: ${lastChecked.toLocaleTimeString()}` : ""}
      >
        <span className={`h-2 w-2 rounded-full ring-4 ${badgeConfig.dot}`} />
        <span className="truncate">{badgeConfig.text}</span>
      </button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={t.connected.replace("Service Online", "Backend Service Connection")}
        description={lastChecked ? `Status checked at ${lastChecked.toLocaleTimeString()}` : ""}
        maxWidth="md"
      >
        <div className="space-y-4 pt-2">
          <div className="flex items-center gap-3 rounded-lg border p-3.5 bg-slate-50 dark:bg-slate-800/50 dark:border-slate-700">
            <span className={`h-3 w-3 rounded-full ring-4 ${badgeConfig.dot}`} />
            <div>
              <p className="font-semibold text-sm text-slate-900 dark:text-white">{badgeConfig.text}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {status === "connected"
                  ? "FastAPI service is actively responding to health checks on port 8000."
                  : t.backendOfflineMessage}
              </p>
            </div>
          </div>

          {status === "disconnected" && (
            <div className="rounded-lg border border-amber-200/80 bg-amber-50/70 p-3.5 text-xs text-amber-900 dark:border-amber-800/50 dark:bg-amber-950/40 dark:text-amber-200">
              <p className="font-semibold mb-1">How to start the backend service:</p>
              <pre className="mt-1.5 rounded bg-slate-900 p-2.5 font-mono text-[11px] text-emerald-400 overflow-x-auto">
                python src/serve.py
              </pre>
              <p className="mt-2 text-[11px] text-amber-800 dark:text-amber-300">
                {t.startBackendHint}
              </p>
            </div>
          )}

          <div className="flex justify-between items-center pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleManualCheck}
              isLoading={status === "checking"}
            >
              <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              Check Now
            </Button>
            <Button size="sm" onClick={() => setIsModalOpen(false)}>
              Close
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}
