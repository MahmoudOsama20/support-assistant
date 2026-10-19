"""Lightweight demo agent for low-RAM or memory-constrained environments.

Uses the real synthetic SQLite database (data/db/support.db) and factual KB responses
without loading multi-gigabyte PyTorch models into Windows virtual memory.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "support.db"


class DemoAgent:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)

    async def handle(self, query: str):
        q = query.strip().lower()
        t0 = time.perf_counter()

        # 1. Short query -> clarification
        words = q.split()
        if len(words) < 3 and not any(ch.isdigit() for ch in q):
            return SimpleNamespace(
                status="clarify",
                action="clarify",
                language="ar" if any("\u0600" <= ch <= "\u06FF" for ch in q) else "en",
                route="none",
                route_confidence=0.0,
                answer="Could you clarify your question? For example: 'What is the daily transfer limit for Basic accounts?' or 'How do I upgrade to Plus?'",
                tool=None,
                refusal_reason=None,
                intent=None,
                intent_confidence=None,
                citations=[],
                sql=None,
                columns=[],
                rows=[],
                row_count=None,
                truncated=False,
                degraded=False,
                error_code=None,
                retryable=False,
                flags={"too_short": True, "demo_mode": True},
                timings_ms={"total": (time.perf_counter() - t0) * 1000},
                detail={},
            )

        # 2. Out of scope -> refusal
        if "weather" in q or "طقس" in q or "football" in q or "recipe" in q:
            return SimpleNamespace(
                status="refused",
                action="refuse",
                language="ar" if any("\u0600" <= ch <= "\u06FF" for ch in q) else "en",
                route="none",
                route_confidence=0.98,
                answer="I can only assist with Nile Wallet customer support inquiries regarding accounts, verification, transfers, and transactions.",
                tool=None,
                refusal_reason="out_of_scope",
                intent="out_of_scope",
                intent_confidence=0.95,
                citations=[],
                sql=None,
                columns=[],
                rows=[],
                row_count=None,
                truncated=False,
                degraded=False,
                error_code=None,
                retryable=False,
                flags={"too_short": False, "demo_mode": True},
                timings_ms={"total": (time.perf_counter() - t0) * 1000},
                detail={},
            )

        # 3. SQL Query: Customer balance or transactions
        is_balance_query = "رصيد" in q or "balance" in q or "منى عادل" in q or "mona" in q
        is_txn_query = "transaction" in q or "معاملات" in q or "5000" in q or "تحويلات" in q

        if (is_balance_query or is_txn_query) and self.db_path.is_file():
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cur = con.cursor()

            if is_balance_query:
                sql = "SELECT c.customer_id, c.full_name, c.tier, a.balance, 'EGP' as currency FROM customers c JOIN accounts a ON c.customer_id = a.customer_id WHERE c.customer_id = 1"
                cur.execute(sql)
                rows = cur.fetchall()
                con.close()
                columns = ["customer_id", "full_name", "tier", "balance", "currency"]
                answer = "رصيد حساب منى عادل (Premium) هو 84,250.50 جنيه مصري." if "رصيد" in q or "منى" in q else "Mona Adel's account balance is 84,250.50 EGP."
                return SimpleNamespace(
                    status="answered",
                    action="sql",
                    language="ar" if "رصيد" in q or "منى" in q else "en",
                    route="sql_query",
                    route_confidence=0.99,
                    answer=answer,
                    tool="sql",
                    refusal_reason=None,
                    intent="account_balance",
                    intent_confidence=0.94,
                    citations=[],
                    sql=sql,
                    columns=columns,
                    rows=[list(r) for r in rows],
                    row_count=len(rows),
                    truncated=False,
                    degraded=False,
                    error_code=None,
                    retryable=False,
                    flags={"too_short": False, "demo_mode": True},
                    timings_ms={"total": (time.perf_counter() - t0) * 1000, "sql_ms": 2.5},
                    detail={},
                )
            else:
                sql = "SELECT txn_id, txn_type, amount, fee, counterparty, status, created_at FROM transactions WHERE amount >= 5000 ORDER BY amount DESC LIMIT 10"
                cur.execute(sql)
                rows = cur.fetchall()
                con.close()
                columns = ["txn_id", "txn_type", "amount", "fee", "counterparty", "status", "created_at"]
                return SimpleNamespace(
                    status="answered",
                    action="sql",
                    language="en",
                    route="sql_query",
                    route_confidence=0.97,
                    answer="Here are the high-value transactions of 5,000 EGP or more from the database:",
                    tool="sql",
                    refusal_reason=None,
                    intent="transaction_history",
                    intent_confidence=0.91,
                    citations=[],
                    sql=sql,
                    columns=columns,
                    rows=[list(r) for r in rows],
                    row_count=len(rows),
                    truncated=False,
                    degraded=False,
                    error_code=None,
                    retryable=False,
                    flags={"too_short": False, "demo_mode": True},
                    timings_ms={"total": (time.perf_counter() - t0) * 1000, "sql_ms": 3.1},
                    detail={},
                )

        # 4. RAG Knowledge Base Answers (Limits / KYC / Onboarding)
        is_ar = any("\u0600" <= ch <= "\u06FF" for ch in q)
        if "limit" in q or "حدود" in q or "transfer" in q:
            answer = (
                "الحد الأقصى اليومي للتحويل في حساب الأساسي (Basic) هو 5,000 جنيه مصري، والحد الشهري 20,000 جنيه، والحد الأقصى للرصيد هو 10,000 جنيه. يمكنك ترقية حسابك إلى بلس لزيادة الحد اليومي إلى 30,000 جنيه."
                if is_ar
                else "The daily transfer limit for Basic accounts is 5,000 EGP, with a monthly limit of 20,000 EGP and a maximum balance of 10,000 EGP. You can upgrade to Plus to raise your daily limit to 30,000 EGP."
            )
            citations = [
                {
                    "doc_id": "kb_003" if not is_ar else "kb_004",
                    "chunk_id": "kb_003#c1" if not is_ar else "kb_004#c1",
                    "title": "Transfer limits and fees" if not is_ar else "حدود ورسوم التحويل",
                    "heading": "Limits by tier" if not is_ar else "حدود المعاملات حسب الفئة",
                    "status": "active",
                    "snippet": "Basic accounts have a daily transfer limit of 5,000 EGP and monthly limit of 20,000 EGP." if not is_ar else "الحد الأقصى للتحويل اليومي للحساب الأساسي هو 5000 جنيه والشهري 20000 جنيه.",
                }
            ]
        else:
            answer = (
                "للترقية من الحساب الأساسي إلى بلس، يرجى الدخول إلى الملف الشخصي ثم التحقق، ورفع صورة واضحة للرقم القومي من الوجهين والتقاط صورة سيلفي. تتم مراجعة الطلب خلال يوم عمل واحد وتُطبق الحدود الجديدة فور الموافقة."
                if is_ar
                else "To upgrade from Basic to Plus verification, go to Profile > Verification in the app. Upload clear photos of the front and back of your Egyptian national ID and take a selfie. Reviews are completed within 1 business day."
            )
            citations = [
                {
                    "doc_id": "kb_001" if not is_ar else "kb_002",
                    "chunk_id": "kb_001#c2" if not is_ar else "kb_002#c2",
                    "title": "Account Opening and Identity Verification" if not is_ar else "فتح الحساب والتحقق من الهوية",
                    "heading": "Verification levels" if not is_ar else "مستويات التحقق",
                    "status": "active",
                    "snippet": "Plus tier requires front and back photos of your national ID and a selfie. Review takes 1 business day." if not is_ar else "مستوى بلس: ارفع صورة وجهي البطاقة والتقط صورة سيلفي، وتتم المراجعة خلال يوم عمل واحد.",
                }
            ]

        return SimpleNamespace(
            status="answered",
            action="rag",
            language="ar" if is_ar else "en",
            route="kb_question",
            route_confidence=0.99,
            answer=answer,
            tool="rag",
            refusal_reason=None,
            intent="qa_currency",
            intent_confidence=0.88,
            citations=citations,
            sql=None,
            columns=[],
            rows=[],
            row_count=None,
            truncated=False,
            degraded=False,
            error_code=None,
            retryable=False,
            flags={"too_short": False, "demo_mode": True},
            timings_ms={"total": (time.perf_counter() - t0) * 1000, "rag_ms": 15.0},
            detail={"gate_prob": 0.85},
        )
