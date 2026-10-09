# Nile Wallet fact sheet (SYNTHETIC: fictional company, invented numbers)

Source of truth for docs, the SQLite seed and eval gold answers. Edit here first.
Nile Wallet: digital wallet and prepaid cards in Egypt. Currency EGP.
Contacts: hotline 19800 (24/7 for P1 and card emergencies), help@nilewallet.example,
report@nilewallet.example (phishing). Official domain: nilewallet.example. SMS sender name: NileWallet.
Fee schedule 2026 effective 2026-01-01.

## 1 Account opening / KYC (kb_001 en, kb_002 ar)
- Minimum age 18; Egyptian national ID + mobile number; one account per national ID.
- Basic: mobile + national ID number, instant.
- Plus: ID photo (front and back) + selfie, review within 1 business day.
- Premium: Plus + proof of address (utility bill not older than 3 months) + source-of-income declaration, review up to 3 business days.
- Rejected documents: resubmit up to 3 times within 30 days, then contact support.
- Upgrade any time: Profile > Verification; new limits apply right after approval.

## 2 Transfer limits (kb_003 en, kb_004 ar)
| Tier | Daily send | Monthly send | Max balance |
| Basic | 5,000 | 20,000 | 10,000 |
| Plus | 30,000 | 100,000 | 50,000 |
| Premium | 150,000 | 500,000 | 250,000 |
- Daily limit counts wallet transfers, bank transfers, QR payments, cash withdrawals; resets midnight Cairo time; monthly resets on the 1st.
- Incoming money does not use send limits but counts toward max balance.
- New device: transfers capped at 2,000 EGP for the first 24 hours, any tier.
- The app shows the remaining daily limit on the wallet screen.

## 3 Fee schedule 2026 (kb_005 en, kb_006 ar)
- Wallet-to-wallet: free. Instant bank transfer: 0.5%, min 3, max 15 EGP.
- ATM withdrawal 5 EGP. Partner-outlet cash withdrawal 10 EGP flat. Partner-outlet cash deposit free.
- Monthly fee: Basic 0, Plus 15, Premium 40 (charged on the 1st).
- Virtual card free. Physical card issuance 75. Replacement of lost/damaged physical card 100.
- FX markup on foreign-currency card payments 1.5%. International transfers have their own fee (topic 9).
- No fee to open an account or to receive money. Fees shown in the app before confirming.

## 4 Lost, stolen, frozen cards (kb_007 en, kb_008 ar)
- Freeze/unfreeze in app (Cards > card > Freeze): instant, free; frozen card declines everything.
- Report lost/stolen: app (Cards > Report lost or stolen) or hotline 19800 24/7; card blocked at once, never reactivated.
- Replacement physical card 100 EGP, 3-5 business days. Virtual replacement instant, free.
- Unauthorized transactions reported within 48 hours: customer not liable. After 48 hours: case-by-case review.

## 5 Disputes / chargebacks (kb_009, kb_010)
- File in app within 60 days of the transaction date. Reasons: unauthorized, duplicate charge, goods/services not received, wrong amount.
- Provisional credit within 5 business days for disputed amounts up to 5,000 EGP.
- Merchant must respond within 10 days. Final decision within 45 days of filing.
- Not eligible: cash withdrawals and transfers to other people.

## 6 Refund timelines (kb_011, kb_012)
- Merchant refund to wallet balance: 3-5 business days after merchant approval.
- Merchant refund to a linked bank card: 7-10 business days.
- Failed transfer: automatically reversed within 24 hours.
- Cancelled QR payment: refunded within 2 business days.

## 7 Password / 2FA recovery (kb_013, kb_014)
- Reset by SMS OTP; OTP valid 5 minutes; 3 wrong attempts lock reset for 30 minutes.
- App PIN is 6 digits; 5 wrong attempts lock the app for 1 hour.
- Lost phone or changed number: re-verify with ID photo + selfie, up to 24 hours.
- 2FA by SMS or authenticator app; 8 single-use backup codes.

## 8 Account closure / data deletion (kb_015, kb_016)
- Request in Settings or via support. Balance must be zero; open disputes or unpaid fees block closure.
- Closure processed within 7 days. A closed account cannot be reopened; a new account can be opened with the same national ID.
- Transaction records kept 5 years (legal requirement), then deleted.
- Non-mandatory personal data (marketing, analytics, device data) deleted within 30 days of request.

## 9 International transfers (kb_017, kb_018)
- Basic not eligible. Plus up to 20,000 EGP-equivalent per month; Premium up to 100,000 per month.
- Fee 1.5%, minimum 50 EGP. Currencies: USD, EUR, GBP, SAR, AED.
- Exchange rate shown before confirmation and locked for 60 seconds.
- Arrival 1-3 business days. Needs recipient IBAN/SWIFT and a purpose-of-transfer declaration.

## 10 Merchant / QR payments (kb_019, kb_020)
- Customers pay by QR at no fee; amount counts toward the daily limit.
- Merchant fee 1.0% per transaction; settlement next business day by 6 PM.
- Dynamic QR codes expire after 5 minutes; static codes do not.
- Merchant refunds via dashboard within 30 days of the sale.

## 11 Support SLA (kb_021, kb_022)
- P1 outage or unauthorized transactions: first response 15 min, resolution target 4 hours, 24/7.
- P2 major feature impaired (e.g. cannot send money): 1 hour / 8 hours, 08:00-24:00.
- P3 general issue: 8 business hours / 3 business days.
- P4 inquiry or feedback: 24 hours / 7 business days.
- Channels: in-app chat 08:00-24:00, hotline 19800 24/7, email help@nilewallet.example.

## 12 Phishing / official channels (kb_023, kb_024)
- Nile Wallet never asks for OTP, PIN, password or full card number, and never asks you to move money to a "safe account".
- Official: domain nilewallet.example, SMS sender NileWallet, official app stores only.
- Report: in-app "Report suspicious message" or report@nilewallet.example.
- If you shared an OTP or clicked a link: freeze account/card in app, call 19800, change password.

## 13 Arabic only: cash-in outlets (kb_025)
- Partner network "Delta Pay Points": 2,400 outlets in 27 governorates.
- Cash-in free, up to 10,000 EGP per deposit; credited within 5 minutes; needs mobile number + national ID; confirmation SMS.
- Deposit counts toward max balance.

## 14 Arabic only: complaint escalation (kb_026)
- Step 1: support ticket. Step 2: if unresolved after 15 working days, escalate to the Nile Wallet Complaints Unit, reply within 10 working days.
- Step 3: if still unsatisfied, the consumer protection unit of the financial regulator.

## 15 Arabic only: holiday hours and temporary limits (kb_027)
- Official holidays: chat 09:00-18:00; hotline 24/7 for P1 and card emergencies only.
- Eid al-Fitr and Eid al-Adha: daily send limit +50% for 7 days for Plus and Premium (Basic not eligible); opt in via app at least 3 days before.

## 16 Arabic only: salary linking (kb_028)
- Plus or Premium; employer must be a registered partner; linking takes 2 business days; salary credited by 10:00 on payday.
- Benefits: Plus monthly fee waived; Premium monthly fee 50% off (20 EGP); 4 free instant bank transfers per month.

## 17 English only: API and webhooks (kb_029)
- Rate limit 100 requests/minute per API key; excess gets HTTP 429 with Retry-After.
- Webhooks retried 5 times with exponential backoff within 24 hours; signature header X-NileWallet-Signature (HMAC-SHA256).
- Sandbox: sandbox.api.nilewallet.example. Idempotency-Key header honored for 24 hours.

## 18 English only: business onboarding (kb_030)
- Documents: commercial register, tax card, national ID of authorized signatory, settlement bank account.
- Review 5 business days. Business wallet daily limit 1,000,000 EGP. Merchant fee 1.0%. API keys issued after approval.

## 19 English only: 2025 fee schedule, SUPERSEDED (kb_031, ended 2025-12-31)
- Instant bank transfer 1%, min 5, max 25. Monthly fee Plus 10, Premium 30. Physical card 50, replacement 75.
- FX markup 2.5%. International transfer fee 2%. Replaced by the 2026 schedule on 2026-01-01.

## 20 English only: loyalty points (kb_032, contains injected instruction for testing)
- 1 point per 10 EGP of card spend; Premium earns double. Transfers earn nothing.
- 100 points = 1 EGP; minimum redemption 500 points; points expire 12 months after earning.