import type { Metadata, Viewport } from "next";
import { LanguageProvider } from "@/lib/i18n/context";
import { AppToaster } from "@/components/ui/toast";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nile Wallet Support Assistant | Multilingual AI Customer Support",
  description:
    "Production-grade customer support assistant powered by hybrid RAG with verified citations and guarded read-only SQL queries.",
  keywords: [
    "AI Customer Support",
    "RAG",
    "Arabic NLP",
    "FastAPI",
    "Next.js",
    "SQL Agent",
    "Nile Wallet",
  ],
  authors: [{ name: "Mahmoud Osama" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#059669",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className="min-h-screen flex flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100 antialiased selection:bg-emerald-500 selection:text-white"
      >
        <LanguageProvider>
          {children}
          <AppToaster />
        </LanguageProvider>
      </body>
    </html>
  );
}
