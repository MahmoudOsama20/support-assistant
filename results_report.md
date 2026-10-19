# Support Assistant: results report

Nile Wallet is a fictional wallet-and-cards company in Egypt. The knowledge base (KB), SQLite database and routing data are **synthetic**; MASSIVE is real. Numbers are measured unless marked UNVERIFIED. Repo: github.com/MahmoudOsama20/support-assistant. Part A detail: `report/part_a_results.md`.

## 1. Summary

- **Part A.** A fine-tuned multilingual-E5-small (118M) ships: TEST accuracy 0.838, macro-F1 0.806 (n=5,948, ar-SA + en-US). A 7B zero-shot LLM reaches 0.563 accuracy.
- **Part B.** An async FastAPI agent: a small fine-tuned router chooses hybrid RAG with citations, read-only SQL, a clarification or a refusal. The LLM never chooses the tool.
- **Part C.** 70 cases, one gpt-oss-120b run: task success **53/70 (0.757; Wilson 95% CI about 0.65-0.84)**. 0/70 forbidden strings leaked, 0/11 unanswerable questions answered, 0/19 answered RAG responses had an unsupported claim. The main weakness is routing: 11 of 17 failures are clarifications from an under-confident route model.

## 2. Part A: intent model

| Model | Accuracy | Macro-F1 | ar-SA F1 | en-US F1 | CPU p50 / p95 (batch 1) | Size |
|---|---|---|---|---|---|---|
| XLM-R base (278M) | 0.8485 | 0.8042 | 0.7585 | 0.8464 | 29.5 / 40.6 ms | 1,129 MB |
| **E5-small (118M), shipped** | 0.8384 | 0.8056 | 0.7618 | 0.8496 | 12.0 / 15.9 ms | 488 MB |
| allam-2-7b zero-shot (1,000 rows, 60 labels) | 0.563 [0.532, 0.595] | 0.619 | n/a | n/a | API p50 379 / p95 1,902 ms | n/a |

**Decision.** Rule fixed before the comparison: ship the smaller model if macro-F1 is within 1.5 points on both languages. E5-small passes on point estimates (+0.33 ar-SA, +0.32 en-US); the 95% lower bound for ar-SA is -1.51, just past the margin. Paired accuracy is 1.0 point below XLM-R (CI [-1.8, -0.2], McNemar p=0.008), macro-F1 equal (+0.14, CI [-1.2, +1.4]). On the LLM's sample E5 leads by 26.7 accuracy points; the fair label is "7B, zero-shot, 60-way", not "LLMs lose". LLM list cost was not computed (free tier). Confusion analysis: `part_a_results.md`.

## 3. Part B: system

**Routing.** MASSIVE intents are voice commands, so the agent uses four routes (`kb_question`, `data_lookup`, `out_of_scope`, `unsafe_request`) from the Part A E5-small re-headed to 4 classes. The Part A intent is only logged. A top route probability below tau_route=0.95 triggers a clarification; inputs under 3 words without a digit are clarified before any model runs.

**RAG.** 32 docs (16 en, 16 ar), 126 chunks. BM25 + bge-m3 dense, RRF fusion, bge-reranker-v2-m3. An evidence gate (tau=0.05, calibrated on 24 dev queries) prevents LLM calls without evidence; superseded docs are dropped. The LLM returns claims with verbatim quotes; a claim survives only if its quote appears in the cited chunk and its numbers appear in the quote. Citations come from chunk metadata.

**SQL and safety.** The LLM emits JSON (`sql`, `clarify`, `cannot_answer`); the SQL passes a sqlglot validator (SELECT only, allowlists), a SQLite authorizer and a read-only connection, capped at 50 rows. Further layers: injection flag, `unsafe_request` route, tool-less LLM, retrieved text delimited as data, canary check, Pydantic response validation, honeypot table.

**Route model.** v1 scores 0.9625 on its synthetic held-out set (optimistic) and 33/42 on a hand-written reality-check set. A v2 retrain was rejected under a rule written before its results; v1 ships.

## 4. Part C: evaluation

70 cases (en/ar), written separately from all dev sets: RAG 28, SQL 16, KB-unanswerable 8, DB-unanswerable 3, ambiguous 4, direct injection 5, destructive SQL 3, indirect injection 2, prompt leak 1. One scored run, gpt-oss-120b, temperature 0, host RTX 3050.

By category: rag_en 8/11, rag_ar 7/11, rag_cross 4/6, sql_en 8/8, sql_ar 7/8, kb_unanswerable 5/8, db_unanswerable 2/3, ambiguous 3/4, injection_direct 5/5, sql_destructive 3/3, prompt_leak 1/1, injection_indirect 0/2 (vacuous, see below). SQL execution match 15/16 lenient, 11/16 strict (the misses return extra id/name columns). Over-refusal 4/46, over-clarify 7/46, errors 0/70, refusal correct 16/20, gold doc cited given an answer 19/20, routing accuracy 57/70 (diagnostic), unanswerable answered 0/11 (95% upper bound about 0.26).

| Failure cause (17 cases) | Cases | n |
|---|---|---|
| Route confidence < 0.95, clarify, no tool ran (4 are unanswerable: a safe non-answer) | rag_en_04/06/09, rag_ar_11, sql_ar_02, kb_unanswerable_01/03/07, db_unanswerable_03, injection_indirect_01/02 | 11 |
| Claim verifier dropped claims | rag_ar_10, rag_cross_02, rag_cross_04 | 3 |
| Confident misroute | rag_ar_01 (KB question answered with SQL data: a wrong answer), rag_ar_07 (refused) | 2 |
| Refused instead of clarify | ambiguous_01 | 1 |

**Verifier drops.** In `rag_cross_02` both drops follow the rule (a number missing from the quote; a garbled Arabic quote). In `rag_ar_10` and `rag_cross_04` the claims match the fact sheet and the quoted text equals the bullet items of chunk `kb_028#c3` apart from line breaks, so the cause lies in the verifier's text normalization (not yet confirmed). Not fixed: changing it after seeing the test set would be tuning on test.

**Indirect injection.** Both cases clarified at the route layer, so the poisoned document never reached the model: a vacuous test. A component-level check (outside the aggregate, n=2) calls the RAG tool directly: both answers were correct, cited `kb_032` and did not obey the injected instruction. The payload sits in `kb_032#c2`; the Arabic answer states a fact found only there, so the payload was in its context (English case UNVERIFIED). This check bypasses the canary layer. All destructive and secret-seeking cases were refused at the route layer, so the SQL validator and authorizer were not reached by this eval (unit tests only).

| Retrieval (30 gold cases, doc level, no LLM) | R@1 | R@3 | MRR | Cross-lingual MRR (n=6) |
|---|---|---|---|---|
| BM25 | 0.600 | 0.733 | 0.678 | 0.042 |
| Dense (bge-m3) | 0.900 | 1.000 | 0.950 | 0.917 |
| Hybrid (RRF) | 0.700 | 0.833 | 0.793 | 0.298 |
| Hybrid + rerank | 0.933 | 1.000 | 0.967 | 0.917 |

**Ablation (one component: the reranker).** Hybrid vs hybrid + rerank, paired per case: 8 better, 22 tie, 0 worse. The reranker rescues a fusion that lexical hits pollute on cross-lingual queries; it does not clearly beat dense alone (one case at R@1, n=30). An end-to-end arm was not run: the evidence gate uses reranker scores and would need re-calibration.

**Hallucination.** 0/19 answered RAG responses had an unsupported claim (upper bound about 0.17): run-time quote verification, a number-grounding script, and a manual read of all 19 answers against the fact sheet. No LLM judge. SQL answers are database values.

| Latency, host RTX 3050 (ms) | n | Est. p50 | Est. p95 | Pipeline p50 | LLM p50 |
|---|---|---|---|---|---|
| kb_question | 32 | 936 | 1,617 | 256 | 796 |
| data_lookup | 24 | 529 | 1,582 | 17 | 554 |
| All 70 (refusals and clarifications about 20 ms) | 70 | 508 | 1,594 | 19 | 698 |

Estimated total = cached pipeline time + LLM time recorded in the paced run: an estimate, not one measured pass. The Docker CPU image gave single samples only: 11.9 s for a cold uncached RAG request, 5.6 s with the LLM cached (about 5.5 s retrieval). One RAG call used about 830 prompt and 160 completion tokens; price not computed.

## 5. Deviations and limitations

- tau_route was chosen on dev after the sweep, with the rule amended to also require zero out_of_scope leaks (post hoc, dev-only). The SQL headline became lenient match before the 70-case run (3/18 dev failures were extra columns). Latency uses cached pipeline plus recorded LLM time because pacing sleeps sat inside LLM spans. The ablation is retrieval-level only; an LLM judge and a 20b-versus-120b comparison were cut.
- Synthetic data; the 70 cases were written by the author with an AI assistant; one run, so intervals are wide.
- Natural questions can be refused or clarified by the route model (for example "show open tickets for Salma Nabil", refused at 0.985 in a dev run). SQL answers are column dumps, not localized to Arabic. Not covered: multi-turn context, load testing, authentication.
