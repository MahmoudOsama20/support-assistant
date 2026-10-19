"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { SupportedLanguage, TranslationDict, translations } from "./translations";

interface LanguageContextType {
  language: SupportedLanguage;
  dir: "ltr" | "rtl";
  t: TranslationDict;
  setLanguage: (lang: SupportedLanguage) => void;
  toggleLanguage: () => void;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<SupportedLanguage>(() => {
    if (typeof window !== "undefined") {
      try {
        const stored = localStorage.getItem("nile_support_lang") as SupportedLanguage | null;
        if (stored === "en" || stored === "ar") return stored;
      } catch {}
    }
    return "en";
  });

  const setLanguage = (lang: SupportedLanguage) => {
    setLanguageState(lang);
    try {
      localStorage.setItem("nile_support_lang", lang);
    } catch {
      // Ignore
    }
  };

  const toggleLanguage = () => {
    setLanguage(language === "en" ? "ar" : "en");
  };

  const dir: "ltr" | "rtl" = language === "ar" ? "rtl" : "ltr";

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = language;
      document.documentElement.dir = dir;
    }
  }, [language, dir]);

  return (
    <LanguageContext.Provider
      value={{
        language,
        dir,
        t: translations[language],
        setLanguage,
        toggleLanguage,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
