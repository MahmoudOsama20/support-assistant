"use client";

import React from "react";
import { useLanguage } from "@/lib/i18n/context";
import { ConnectionBadge } from "./connection-badge";
import { Button } from "@/components/ui/button";
import { Menu, Terminal, Globe, Plus } from "lucide-react";

interface HeaderProps {
  onOpenSidebar: () => void;
  onNewChat: () => void;
  isDevPanelOpen: boolean;
  onToggleDevPanel: () => void;
  activeSessionTitle?: string;
}

export function Header({
  onOpenSidebar,
  onNewChat,
  isDevPanelOpen,
  onToggleDevPanel,
  activeSessionTitle,
}: HeaderProps) {
  const { language, toggleLanguage } = useLanguage();

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-200/80 bg-white/80 px-4 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/80">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenSidebar}
          className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 lg:hidden"
          aria-label="Open navigation"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm text-slate-900 dark:text-white">
            {activeSessionTitle || "Support Assistant"}
          </span>
          <span className="hidden sm:inline-block rounded-md bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
            Nile Wallet
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden sm:block">
          <ConnectionBadge />
        </div>

        {/* Developer Trace Toggle */}
        <Button
          variant={isDevPanelOpen ? "primary" : "outline"}
          size="sm"
          onClick={onToggleDevPanel}
          className="text-xs gap-1.5"
          title="Toggle Technical Developer Inspection Panel"
        >
          <Terminal className="h-3.5 w-3.5" />
          <span className="hidden md:inline">Dev Panel</span>
        </Button>

        {/* Quick Language Toggle */}
        <Button
          variant="outline"
          size="sm"
          onClick={toggleLanguage}
          className="text-xs gap-1"
          title="Switch Language / تبديل اللغة"
        >
          <Globe className="h-3.5 w-3.5" />
          <span>{language === "en" ? "عربي" : "EN"}</span>
        </Button>

        <Button
          variant="ghost"
          size="icon"
          onClick={onNewChat}
          className="lg:hidden"
          title="New Chat"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
