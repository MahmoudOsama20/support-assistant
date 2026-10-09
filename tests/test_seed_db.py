from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pytest

from supportdb.seed import (HONEYPOT_TOKEN, REFERENCE_DATE, TICKET_CATEGORIES, TIER_RULES, build_db,
                            fee_for)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    build_db(path)
    c = sqlite3.connect(path)
    yield c
    c.close()


def _bdays(a: date, b: date) -> int:
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def test_row_counts(con) -> None:
    expected = {"customers": 40, "accounts": 55, "cards": 45, "transactions": 400,
                "support_tickets": 80, "refunds": 30, "admin_users": 2}
    for table, n in expected.items():
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, table


def test_foreign_keys_clean(con) -> None:
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_one_active_account_per_customer(con) -> None:
    rows = con.execute("SELECT customer_id, COUNT(*) FROM accounts WHERE status='active' "
                       "GROUP BY customer_id").fetchall()
    assert len(rows) == 40 and all(n == 1 for _, n in rows)
    assert con.execute("SELECT COUNT(*) FROM accounts WHERE status='closed' AND balance != 0").fetchone()[0] == 0


def test_plus_premium_are_kyc_approved(con) -> None:
    assert con.execute("SELECT COUNT(*) FROM customers WHERE tier != 'Basic' "
                       "AND kyc_status != 'approved'").fetchone()[0] == 0


def test_balance_within_tier_max(con) -> None:
    for tier, balance in con.execute("SELECT c.tier, a.balance FROM accounts a "
                                     "JOIN customers c USING (customer_id)"):
        assert 0 <= balance <= TIER_RULES[tier]["max_balance"]


def test_daily_send_limits_respected(con) -> None:
    totals: dict[tuple[int, str], float] = defaultdict(float)
    tier = {a: t for a, t in con.execute("SELECT a.account_id, c.tier FROM accounts a "
                                         "JOIN customers c USING (customer_id)")}
    for acct, amount, created in con.execute(
            "SELECT account_id, amount, created_at FROM transactions WHERE txn_type IN "
            "('wallet_transfer','bank_transfer','qr_payment','atm_withdrawal','outlet_withdrawal')"):
        totals[(acct, created[:10])] += amount
    assert totals
    for (acct, day), total in totals.items():
        assert total <= TIER_RULES[tier[acct]]["daily"], (acct, day, total)


def test_fees_follow_fact_sheet(con) -> None:
    for ttype, amount, fee, status in con.execute("SELECT txn_type, amount, fee, status FROM transactions"):
        assert fee == pytest.approx(0.0 if status == "reversed" else fee_for(ttype, amount))
    for ctype, fee, replaces in con.execute("SELECT card_type, issuance_fee, replaces_card_id FROM cards"):
        expected = 0.0 if ctype == "virtual" else (100.0 if replaces else 75.0)
        assert fee == expected


def test_outlet_deposit_limit_and_free(con) -> None:
    rows = con.execute("SELECT amount, fee FROM transactions WHERE txn_type='outlet_deposit'").fetchall()
    assert rows and all(a <= 10000 and f == 0 for a, f in rows)


def test_refund_timing_rules(con) -> None:
    allowed = {("cancelled_qr", "wallet"): range(2, 3), ("merchant_refund", "wallet"): range(3, 6),
               ("merchant_refund", "bank_card"): range(7, 11)}
    paid = 0
    for rtype, dest, status, approved, paid_on in con.execute(
            "SELECT refund_type, destination, status, approved_on, paid_on FROM refunds"):
        assert (rtype, dest) in allowed
        if status == "paid":
            paid += 1
            assert _bdays(date.fromisoformat(approved), date.fromisoformat(paid_on)) in allowed[(rtype, dest)]
            assert date.fromisoformat(paid_on) <= REFERENCE_DATE
        else:
            assert paid_on is None
    assert paid > 0


def test_ticket_priority_matches_category(con) -> None:
    for category, priority, created, resolved in con.execute(
            "SELECT category, priority, created_at, resolved_at FROM support_tickets"):
        assert TICKET_CATEGORIES[category][0] == priority
        assert resolved is None or resolved >= created


def test_anchor_rows(con) -> None:
    assert con.execute("SELECT full_name, tier, language_pref FROM customers WHERE customer_id=3").fetchone() \
        == ("Salma Nabil", "Plus", "ar")
    assert con.execute("SELECT balance FROM accounts WHERE account_id=1").fetchone()[0] == 84250.50
    assert con.execute("SELECT status FROM cards WHERE card_id=3").fetchone()[0] == "frozen"
    assert con.execute("SELECT priority, status FROM support_tickets WHERE ticket_id=1").fetchone() \
        == ("P1", "in_progress")
    assert con.execute("SELECT amount, fee FROM transactions WHERE txn_id=1").fetchone() == (10000.0, 15.0)
    assert con.execute("SELECT status FROM refunds WHERE refund_id=1").fetchone()[0] == "paid"


def test_admin_users_honeypot_exists(con) -> None:
    tokens = [r[0] for r in con.execute("SELECT api_token FROM admin_users")]
    assert HONEYPOT_TOKEN in tokens


def test_build_is_deterministic(tmp_path: Path) -> None:
    dumps = []
    for name in ("a.db", "b.db"):
        build_db(tmp_path / name)
        c = sqlite3.connect(tmp_path / name)
        dumps.append("\n".join(c.iterdump()))
        c.close()
    assert dumps[0] == dumps[1]