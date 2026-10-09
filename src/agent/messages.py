"""Fixed user-facing texts (ar/en). The LLM never writes refusals or clarifications at the route level."""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "clarify": {
        "en": "Could you clarify your question? For example: are you asking about a Nile Wallet policy or service, or about a specific account or transaction?",
        "ar": "هل يمكنك توضيح سؤالك؟ مثلاً: هل تسأل عن سياسة أو خدمة في نايل واليت، أم عن حساب أو معاملة محددة؟",
    },
    "out_of_scope": {
        "en": "Sorry, I can only help with Nile Wallet services and support data.",
        "ar": "عذراً، أستطيع المساعدة فقط في خدمات نايل واليت وبيانات الدعم.",
    },
    "unsafe_request": {
        "en": "Sorry, I can't help with that request.",
        "ar": "عذراً، لا أستطيع تنفيذ هذا الطلب.",
    },
    "policy": {
        "en": "Sorry, I can't help with that request.",
        "ar": "عذراً، لا أستطيع المساعدة في هذا الطلب.",
    },
    "no_evidence": {
        "en": "I couldn't find enough information to answer that.",
        "ar": "لم أجد معلومات كافية للإجابة عن هذا السؤال.",
    },
    "no_rows": {
        "en": "I found no records matching that request.",
        "ar": "لم أجد سجلات مطابقة لهذا الطلب.",
    },
}


def text_for(key: str, language: str) -> str:
    entry = MESSAGES[key]  # unknown key -> KeyError, loudly
    return entry.get(language, entry["en"])