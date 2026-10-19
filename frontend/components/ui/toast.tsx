"use client";

import { Toaster as SonnerToaster, toast } from "sonner";
import { useLanguage } from "@/lib/i18n/context";

export function AppToaster() {
  const { dir } = useLanguage();

  return (
    <SonnerToaster
      position={dir === "rtl" ? "bottom-left" : "bottom-right"}
      dir={dir}
      toastOptions={{
        className:
          "font-sans border border-slate-200/80 bg-white text-slate-900 shadow-lg dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100",
      }}
    />
  );
}

export { toast };
