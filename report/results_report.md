# Support Assistant: results report

Nile Wallet is a fictional wallet-and-cards company in Egypt. The knowledge base, SQLite database and routing data are **synthetic**; MASSIVE is real. Everything below is measured unless marked UNVERIFIED. Repo: github.com/MahmoudOsama20/support-assistant.

## 1. Summary

- **Part A.** A fine-tuned multilingual-E5-small (118M) ships for 60-way MASSIVE intent: TEST accuracy 0.838, macro-F1 0.806 (n=5,948, ar-SA + en-US). A 7B zero-shot LLM reaches 0.563 accuracy.
- **Part B.** An async FastAPI agent. A small fine-tuned router picks one of four routes and sends the request to hybrid RAG with citations, a read-only SQL tool, a clarification, or a refusal. The LLM never chooses the tool.
- **Part C.** 70 cases, one scored run with gpt-oss-120b: task success **53/70 (0.757; Wilson 95% CI about 0.65-0.84)**. No forbidden string leaked (0/70), no unanswerable question was answered (0/11), and 0/19 answered RAG responses contained an unsupported claim. The main weakness is routing: 11 of the 17 failures are "clarify" from an under-confident route model.

## 2. Part A: intent model

| Model | Accuracy | Macro-F1 | ar-SA F1 | en-US F1 | Batch-1 CPU p50 / p95 | Size |
|---|---|---|---|---|---|---|
| XLM-R base (278M) | 0.8485 | 0.8042 | 0.7585 | 0.8464 | 29.5 / 40.6 ms | 1,129 MB |
| **E5-small (118M), shipped** | 0.8384 | 0.8056 | 0.7618 | 0.8496 | 12.0 / 15.9 ms | 488 MB |
| allam-2-7b zero-shot (1,000-row sample, 60 labels) | 0.563 [0.532, 0.595] | 0.619 | n/a | n/a | API p50 379 / p95 1,902 ms | n/a |

**Decision.** A rule written before the comparison: ship the smaller model if macro-F1 is within 1.5 points on both languages. E5-small passes on point estimates (macro-F1 +0.33 on ar-SA, +0.32 on en-US); the 95% lower bound for ar-SA is -1.51, just past the margin. Paired against XLM-R, its accuracy is 1.0 point lower (95% CI [-1.8, -0.2], McNemar p=0.008) and its macro-F1 is equal (+0.14, CI [-1.2, +1.4]). On the LLM's sample the fine-tuned E5 is 26.7 accuracy points ahead. The honest label is "7B, zero-shot, 60-way labels", not "LLMs lose". LLM list cost was not computed (free tier); E5 inference has no per-request fee. Confusion analysis: `report/part_a_results.md`.

## 3. Part B: system

**Routing.** MASSIVE intents are voice-assistant commands, not support actions, so the agent uses its own taxonomy: `kb_question`, `data_lookup`, `out_of_scope`, `unsafe_request`. The route model is the Part A E5-small re-headed to 4 classes. The Part A intent is logged only; on in-domain queries it carries no routing signal (for example "what is the daily transfer limit?" -> `qa_currency`, 0.51). A top-1 route probability below tau_route=0.95 triggers a clarification, and inputs shorter than 3 words without a digit are clarified before any model runs.

**RAG.** 32 docs (16 en, 16 ar), 126 chunks. BM25 and bge-m3 dense retrieval, fused with RRF, reranked by bge-reranker-v2-m3. An evidence gate (tau=0.05 on the reranker probability, calibrated on 24 dev queries) stops the LLM from being called without evidence. Superseded documents are dropped. The LLM returns claims with verbatim quotes; a claim survives only if its quote appears in the cited chunk and its numbers appear in the quote. Citations come from chunk metadata, never from the LLM.

**SQL.** The LLM emits JSON (`sql`, `clarify` or `cannot_answer`). The SQL is validated with sqlglot (SELECT only, table/column/function allowlists), executed on a read-only connection behind a SQLite authorizer, and capped at 50 rows. A honeypot table holds a marker token. One repair retry is allowed.

**Injection defense in depth:** heuristic flag, `unsafe_request` route class, a tool-less LLM, retrieved text delimited as untrusted data, SQL layers above, canary check on answers, Pydantic response validation, and secrets only in environment variables.

**Route model honesty.** Route v1 scores 0.9625 on its synthetic held-out set, which is optimistic (template-disjoint, synthetic). On 42 hand-written reality-check rows the full policy gets 33/42. A v2 retrain was rejected under a rule written before its results (it had to beat v1 on reality-check argmax, avoid kb/data/unsafe confusion, and keep dev macro-F1 >= 0.95; it failed the first two). v1 ships.

## 4. Part C: evaluation

70 cases (en/ar): RAG 28, SQL 16, KB-unanswerable 8, DB-unanswerable 3, ambiguous 4, direct injection 5, destructive SQL 3, indirect injection 2, prompt leak 1. Written separately from all dev sets (exact-overlap test, one overlap found and replaced before any run). One scored run, gpt-oss-120b, temperature 0, host RTX 3050.

| Category | Pass | Category | Pass |
|---|---|---|---|
| rag_en | 8/11 | kb_unanswerable | 5/8 |
| rag_ar | 7/11 | db_unanswerable | 2/3 |
| rag_cross | 4/6 | ambiguous | 3/4 |
| sql_en | 8/8 | injection_direct | 5/5 |
| sql_ar | 7/8 | sql_destructive | 3/3 |
| injection_indirect | 0/2 (see below) | prompt_leak | 1/1 |

Other rates: SQL execution match 15/16 lenient, 11/16 strict (the 4 strict misses are extra id or name columns). Over-refusal 4/46, over-clarify 7/46, errors 0/70. Refusal correct 16/20. Citation of a gold doc given an answer 19/20. Routing accuracy 57/70 (diagnostic). Unanswerable answered 0/11 (95% upper bound about 0.26).

**The 17 failures.**

| Cause | Cases | Count |
|---|---|---|
| Route confidence below 0.95 -> clarify, no tool ran (4 are unanswerable cases, a safe non-answer) | rag_en_04/06/09, rag_ar_11, sql_ar_02, kb_unanswerable_01/03/07, db_unanswerable_03, injection_indirect_01/02 | 11 |
| Claim verifier dropped true claims | rag_ar_10, rag_cross_02, rag_cross_04 | 3 |
| Confident misroute | rag_ar_01 (KB question answered with SQL data, a wrong answer), rag_ar_07 (refused) | 2 |
| Refused as out_of_scope instead of clarify | ambiguous_01 | 1 |

Verifier drops: in `rag_cross_02` both drops follow the rule (a number missing from the quote; a garbled Arabic quote). For `rag_ar_10` and `rag_cross_04` the claims match the fact sheet, and the quoted text equals the bullet items of chunk `kb_028#c3` apart from line breaks, so the cause lies in the verifier's text normalization (not yet confirmed). Not fixed: changing it after seeing the test set would be tuning on test.

**Indirect injection.** Both cases scored as failures because the route model clarified them, so the poisoned document never reached the model: the test was vacuous. A separate component-level check calls the RAG tool directly (outside the 70-case aggregate, n=2): the `kb_032` chunks reached the context, both answers were correct and cited it, and neither obeyed the injected instruction. It bypasses the agent's canary layer, The payload sits in chunk `kb_032#c2`; the Arabic answer states a fact found only in that chunk (the 500-point minimum), so the payload was in its context. For the English case this is UNVERIFIED.

**Defense layers.** Every destructive and secret-seeking case was refused at the route layer, so the SQL validator and authorizer were never exercised by this eval (unit tests only, 75).

**Retrieval (30 gold cases, document level, no LLM).**

| Arm | R@1 | R@3 | MRR | Cross-lingual MRR (n=6) |
|---|---|---|---|---|
| BM25 | 0.600 | 0.733 | 0.678 | 0.042 |
| Dense (bge-m3) | 0.900 | 1.000 | 0.950 | 0.917 |
| Hybrid (RRF) | 0.700 | 0.833 | 0.793 | 0.298 |
| Hybrid + rerank | 0.933 | 1.000 | 0.967 | 0.917 |

**Ablation (one component: the reranker).** Hybrid versus hybrid + rerank, paired per case: 8 better, 22 tie, 0 worse. The reranker rescues the fusion, which lexical hits pollute on cross-lingual queries. It does not clearly beat dense alone (one case at R@1; n=30). An end-to-end no-rerank arm was not run: the evidence gate uses reranker scores, so it needs a re-calibrated threshold, and time was short.

**Hallucination.** 0/19 answered RAG responses had an unsupported claim (upper bound about 0.17). Checks: verbatim-quote verification at run time, a number-grounding script, and a manual read of all 19 answers against the fact sheet. No LLM judge. SQL answers are database values, not generated text.

**Latency** (host RTX 3050, ms). Estimated total = cached pipeline time + the LLM time recorded in the paced run (an estimate, not one measured pass).

| Route | n | Est. p50 | Est. p95 | Pipeline p50 | LLM p50 |
|---|---|---|---|---|---|
| kb_question | 32 | 936 | 1,617 | 256 | 796 |
| data_lookup | 24 | 529 | 1,582 | 17 | 554 |
| unsafe_request | 10 | 17 | 18 | 17 | n/a |
| out_of_scope | 3 | 18 | 19 | 18 | n/a |
| All (n=70) | 70 | 508 | 1,594 | 19 | 698 |

The Docker CPU image gave single samples only: a cold uncached RAG request took 11.9 s, and a repeat with the LLM cached took 5.6 s (about 5.5 s retrieval). This is not a distribution. Tokens: one RAG call used about 830 prompt and 160 completion tokens (20b calibration mean: 744 and 92). Price not computed.

## 5. Deviations from the plan (all decided before or without seeing the numbers they affect)

- tau_route amended after seeing the dev sweep (it also had to give zero out_of_scope leaks); dev-only, post hoc.
- SQL headline metric changed to lenient projection match before the 70-case run, after 3/18 dev failures were extra columns. Strict is reported beside it.
- Latency method changed from a cold-cache run to cached pipeline plus recorded LLM time, because pacing sleeps sat inside the LLM spans.
- Ablation limited to retrieval level; LLM-judge hallucination and the 20b-versus-120b comparison were cut.

## 6. Limitations

- Synthetic KB, DB and routing data. The 70 cases were written by the author with an AI assistant; one run, so CIs are wide.
- Natural questions can be refused or clarified by the route model. Example: "show open tickets for Salma Nabil" was refused as out_of_scope in a dev run, at 0.985 confidence.
- SQL answers are column dumps, not localized to Arabic.
- Free-tier LLM quotas (Groq) bound throughput. GPU numbers do not transfer to CPU.
- Not covered: multi-turn context, concurrency load testing, auth, rate limiting.