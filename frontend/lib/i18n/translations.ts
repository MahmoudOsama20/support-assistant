export type SupportedLanguage = "en" | "ar";

export interface TranslationDict {
  appTitle: string;
  appSubtitle: string;
  tagline: string;
  description: string;
  newChat: string;
  chatHistory: string;
  noHistory: string;
  clearHistory: string;
  confirmClear: string;
  settings: string;
  language: string;
  theme: string;
  lightMode: string;
  darkMode: string;
  systemMode: string;
  connected: string;
  checking: string;
  disconnected: string;
  backendOfflineMessage: string;
  startBackendHint: string;
  inputPlaceholder: string;
  send: string;
  sending: string;
  charCount: string;
  emptyInputWarning: string;
  copyAnswer: string;
  copied: string;
  retry: string;
  sourcesTitle: string;
  sourceDoc: string;
  sourceChunk: string;
  evidenceSnippet: string;
  sqlResultsTitle: string;
  sqlQuery: string;
  rowCount: string;
  noRowsReturned: string;
  copyCsv: string;
  csvCopied: string;
  clarificationTitle: string;
  clarificationDesc: string;
  refusalTitle: string;
  refusalReasons: {
    out_of_scope: string;
    no_evidence: string;
    unsafe_request: string;
    policy: string;
  };
  errorTitle: string;
  devPanelTitle: string;
  devPanelSubtitle: string;
  requestId: string;
  route: string;
  intent: string;
  timings: string;
  flags: string;
  degradedMode: string;
  rawJson: string;
  examplesTitle: string;
  examplesSubtitle: string;
  examples: Array<{
    category: string;
    label: string;
    query: string;
    badge: string;
  }>;
  disclaimer: string;
  github: string;
}

export const translations: Record<SupportedLanguage, TranslationDict> = {
  en: {
    appTitle: "Nile Wallet Support",
    appSubtitle: "Multilingual AI Assistant",
    tagline: "Support, powered by evidence.",
    description: "Ask questions in English or Arabic. Get answers grounded in knowledge-base sources and structured business data.",
    newChat: "New Conversation",
    chatHistory: "Conversations",
    noHistory: "No conversation history yet",
    clearHistory: "Clear all chats",
    confirmClear: "Are you sure you want to delete all saved conversations?",
    settings: "Preferences",
    language: "Interface Language",
    theme: "Appearance",
    lightMode: "Light",
    darkMode: "Dark",
    systemMode: "System",
    connected: "Service Online",
    checking: "Connecting...",
    disconnected: "Backend Offline",
    backendOfflineMessage: "The FastAPI backend is currently unavailable.",
    startBackendHint: "Ensure the backend is running with 'python src/serve.py' on port 8000.",
    inputPlaceholder: "Ask about transfer limits, KYC verification, transactions, or account balance...",
    send: "Send message",
    sending: "Thinking...",
    charCount: "characters",
    emptyInputWarning: "Please enter a message before sending.",
    copyAnswer: "Copy response",
    copied: "Copied to clipboard",
    retry: "Retry query",
    sourcesTitle: "Cited Knowledge Base Evidence",
    sourceDoc: "Document",
    sourceChunk: "Chunk ID",
    evidenceSnippet: "Direct Grounding Passage",
    sqlResultsTitle: "Structured Database Query Results",
    sqlQuery: "Executed Read-Only Query",
    rowCount: "records returned",
    noRowsReturned: "No matching records found in the database.",
    copyCsv: "Export CSV",
    csvCopied: "CSV copied to clipboard",
    clarificationTitle: "Clarification Needed",
    clarificationDesc: "Your request is too ambiguous or brief. Please provide more details so we can assist you safely.",
    refusalTitle: "Request Not Supported",
    refusalReasons: {
      out_of_scope: "This question falls outside of Nile Wallet customer support scope.",
      no_evidence: "No authoritative evidence was found in the official documentation to substantiate this answer safely.",
      unsafe_request: "This request was withheld for security or safety policy compliance.",
      policy: "Restricted by business operations and compliance policy.",
    },
    errorTitle: "System Notice",
    devPanelTitle: "Developer Inspection",
    devPanelSubtitle: "Real backend trace, routing metadata, and timings",
    requestId: "Request ID",
    route: "Decided Route",
    intent: "Classified Intent (Part A)",
    timings: "Latency Breakdown",
    flags: "Agent Execution Flags",
    degradedMode: "Fallback Mode Activated",
    rawJson: "Inspect Full JSON Payload",
    examplesTitle: "Supported Inquiry Scenarios",
    examplesSubtitle: "Test the hybrid retrieval, read-only SQL, refusal, and clarification flows directly against the live backend:",
    examples: [
      {
        category: "RAG • Knowledge Base",
        label: "Transfer limits by tier",
        query: "What is the daily transfer limit for Basic accounts?",
        badge: "RAG",
      },
      {
        category: "RAG • Policy & KYC",
        label: "Identity verification procedure",
        query: "How do I upgrade from Basic to Plus verification?",
        badge: "RAG",
      },
      {
        category: "SQL • Database (Arabic)",
        label: "Customer balance lookup",
        query: "اعرض رصيد منى عادل",
        badge: "SQL",
      },
      {
        category: "SQL • Database (English)",
        label: "High-value transactions",
        query: "Show me transactions above 5000 EGP",
        badge: "SQL",
      },
      {
        category: "Safety • Out of Scope",
        label: "Irrelevant question refusal",
        query: "What is the weather like in Cairo today?",
        badge: "Refusal",
      },
      {
        category: "Routing • Short Query",
        label: "Ambiguous query clarification",
        query: "transfer",
        badge: "Clarify",
      },
    ],
    disclaimer: "Answers are grounded strictly in Nile Wallet's synthetic knowledge base and SQLite database with zero hallucination guarantee.",
    github: "GitHub Repository",
  },
  ar: {
    appTitle: "المساعد الذكي لنايل واليت",
    appSubtitle: "نظام الدعم الفني الذكي ثنائي اللغة",
    tagline: "دعم فني مدعوم بالأدلة والبيانات.",
    description: "اطرح أسئلتك باللغة العربية أو الإنجليزية. إجابات موثقة ومبنية بدقة على مصادر قاعدة المعرفة والبيانات المنظمة.",
    newChat: "محادثة جديدة",
    chatHistory: "سجل المحادثات",
    noHistory: "لا توجد محادثات سابقة",
    clearHistory: "مسح جميع المحادثات",
    confirmClear: "هل أنت متأكد من رغبتك في حذف جميع المحادثات المخزنة؟",
    settings: "التفضيلات",
    language: "لغة الواجهة",
    theme: "المظهر",
    lightMode: "فاتح",
    darkMode: "داكن",
    systemMode: "تلقائي",
    connected: "الخدمة متصلة",
    checking: "جارٍ الاتصال...",
    disconnected: "الخادم غير متصل",
    backendOfflineMessage: "خادم FastAPI غير متاح حاليًا.",
    startBackendHint: "تأكد من تشغيل الخادم عبر 'python src/serve.py' على المنفذ 8000.",
    inputPlaceholder: "اطرح سؤالاً عن حدود التحويل، توثيق الهوية، المعاملات، أو رصيد الحساب...",
    send: "إرسال الرسالة",
    sending: "جارٍ المعالجة...",
    charCount: "حرف",
    emptyInputWarning: "يرجى كتابة رسالة قبل الإرسال.",
    copyAnswer: "نسخ الإجابة",
    copied: "تم النسخ للحافظة",
    retry: "إعادة المحاولة",
    sourcesTitle: "المصادر والأدلة المقتبسة",
    sourceDoc: "المستند",
    sourceChunk: "رقم المقطع",
    evidenceSnippet: "النص الموثق من قاعدة المعرفة",
    sqlResultsTitle: "نتائج الاستعلام من قاعدة البيانات",
    sqlQuery: "استعلام SQL المنفذ (قراءة فقط)",
    rowCount: "سجلات تم استرجاعها",
    noRowsReturned: "لا توجد بيانات مطابقة في قاعدة البيانات.",
    copyCsv: "تصدير CSV",
    csvCopied: "تم نسخ ملف CSV للحافظة",
    clarificationTitle: "مطلوب توضيح إضافي",
    clarificationDesc: "طلبك قصير جداً أو يحتاج لتفاصيل إضافية لنتمكن من مساعدتك بشكل دقيق وآمن.",
    refusalTitle: "تعذر تقديم الإجابة",
    refusalReasons: {
      out_of_scope: "هذا السؤال خارج نطاق خدمة عملاء نايل واليت.",
      no_evidence: "لا تتوفر أدلة رسمية كافية في قاعدة المعرفة لتوثيق هذه الإجابة بأمان.",
      unsafe_request: "تم حجب هذا الطلب وفقاً لسياسات الأمان والحماية.",
      policy: "مقيد بموجب السياسات التشغيلية والتنظيمية.",
    },
    errorTitle: "تنبيه من النظام",
    devPanelTitle: "لوحة تتبع المطور",
    devPanelSubtitle: "تتبع المسار الفعلي، بيانات التوجيه، وقياسات زمن الاستجابة",
    requestId: "معرف الطلب (Request ID)",
    route: "المسار المختار (Route)",
    intent: "التصنيف التنبؤي (Part A Intent)",
    timings: "تفصيل زمن المعالجة",
    flags: "إشارات المعالجة (Flags)",
    degradedMode: "تم تفعيل الوضع الاحتياطي",
    rawJson: "فحص استجابة JSON الكاملة",
    examplesTitle: "سيناريوهات الدعم التجريبية",
    examplesSubtitle: "جرّب التوجيه التلقائي عبر RAG وقاعدة البيانات واستجابات التوضيح والرفض مباشرة مع الخادم الحي:",
    examples: [
      {
        category: "RAG • قاعدة المعرفة",
        label: "حدود التحويل حسب الفئات",
        query: "What is the daily transfer limit for Basic accounts?",
        badge: "RAG",
      },
      {
        category: "RAG • سياسات التوثيق",
        label: "إجراءات ترقية الحساب",
        query: "How do I upgrade from Basic to Plus verification?",
        badge: "RAG",
      },
      {
        category: "SQL • قاعدة البيانات (عربي)",
        label: "الاستعلام عن رصيد عميل",
        query: "اعرض رصيد منى عادل",
        badge: "SQL",
      },
      {
        category: "SQL • قاعدة البيانات (إنجليزي)",
        label: "المعاملات المالية الكبرى",
        query: "Show me transactions above 5000 EGP",
        badge: "SQL",
      },
      {
        category: "الأمان • خارج الاختصاص",
        label: "رفض الأسئلة الخارجة عن النطاق",
        query: "What is the weather like in Cairo today?",
        badge: "رفض",
      },
      {
        category: "التوجيه • استفسار موجز",
        label: "طلب توضيح للاستفسارات الغامضة",
        query: "تحويل",
        badge: "توضيح",
      },
    ],
    disclaimer: "الإجابات مستندة حصريًا إلى قاعدة المعرفة وقاعدة بيانات SQLite لنايل واليت مع ضمان عدم اختلاق أي معلومات.",
    github: "مستودع GitHub",
  },
};
