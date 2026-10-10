# Live-modification cheat sheet (interview, Monday)

**Rules to say out loud:** (1) change one thing, (2) run the tests, (3) measure on a DEV set, never on the 70 eval cases, (4) say what you expect before you run it. Dev sets: `data/kb/calibration_queries.jsonl`, `data/route/reality_check.jsonl`, `src/sqltool/dev_cases.py`.

**Verify any change:** `python -m pytest -q` (about 15-30 s, 364 tests) and then `python src/agent/ask_agent.py --json "your query"`. A new query makes a real LLM call (quota); cached queries do not.

**Files I have NOT read in this session** (names and signatures come from the project notes): `agent/policy.py`, `agent/rules.py`, `agent/messages.py`, `rag/context.py`, `rag/guard.py`, `agent/schemas.py`, `sqltool/schema.py`, `llm/client.py`. Open each once on Sunday and check the line numbers below.

## 1. Routing

| Request | Where | Change | Verify | Watch out |
|---|---|---|---|---|
| Fewer clarifications | `agent/policy.py`: `DEFAULT_TAU_ROUTE = 0.95` (also `AgentConfig.tau_route` in `agent/core.py`; CLI `ask_agent.py --tau 0.85`) | Lower the value, e.g. 0.85 | `python src/agent/ask_agent.py --tau 0.85 "<query>"`; `python src/route/policy_sweep.py --run-dir results/runs/route-e5small --device cpu` (dev table) | Lower tau reduces clarifies but raises unsafe and out_of_scope leaks. Read the sweep table; do not tune on the 70 cases. |
| Stricter or looser short-query rule | `agent/rules.py`: `is_too_short(text, min_words=3)`; called in `agent/core.py` | Change `min_words` | `pytest tests/test_agent_rules.py` | "my card" is 2 words, so clarified before any model. |
| Make a route safer | `agent/policy.py` `decide()` | Map a class to `refuse` or `clarify` | `pytest tests/test_agent_policy.py` | Keep refusals reason-coded (`refusal_reason`). |
| New deterministic rule | `agent/core.py` `handle()`, next to the `too_short` block | Add a check that returns `make(status="clarify"/"refused", ...)` | A unit test with a fake route model (see `tests/test_agent_core.py`) | Add the flag to `flags` so it shows in the trace. |

## 2. RAG

| Request | Where | Change | Verify | Watch out |
|---|---|---|---|---|
| Change the evidence gate | `rag/context.py`: `DEFAULT_TAU = 0.05` (sigmoid of the top reranker logit); `build_rag_tool(..., tau=)` in `agent/build.py` | Raise or lower | Calibration set via `python src/rag/retrieval_eval.py` | Raise: more refusals; lower: more LLM calls on weak evidence. |
| More or fewer context chunks | `rag/context.py`: `CONTEXT_FETCH = 6`, `CONTEXT_K = 4` | Edit the constants | `pytest -q`; rerun a query and read `detail.context_ids` in the trace | More chunks mean more prompt tokens. |
| Remove the reranker (live ablation) | `agent/build.py` `build_agent`: the `CrossEncoderReranker(device=dev)` argument of `HybridRetriever` | Pass `None` | `python src/eval/retrieval_arms.py` shows the arms without the LLM | The evidence gate uses reranker scores; with `None` the gate sees RRF scores, so it needs recalibration. Say so. |
| Change fusion | `rag/retriever.py`: `rrf_fuse(rankings, k=60, top=20)` | Change `k`, or fuse with weights | `retrieval_arms.py` (hybrid row) | Dense alone already ties hybrid+rerank (n=30). |
| Stricter or looser claim check | `rag/answer.py`: `MIN_QUOTE_WORDS = 3`, `_plain()`, `IGNORED_NUMBERS` | Edit | `pytest -q` | Known weak spot: quotes that differ from a bullet-list chunk only in line breaks were dropped (`rag_ar_10`, `rag_cross_04`). Cause not confirmed. Do not "fix" it against the 70 cases. |
| Add or remove a KB doc | `data/kb/docs/kb_033.md` (Markdown with YAML front matter) | Add the file | `python src/kb/docs.py --expect-count 33`; dense index rebuilds automatically | Tests that expect 32 docs or 126 chunks will fail: update them. Eval gold ids stay unchanged. |

## 3. SQL

| Request | Where | Change | Verify | Watch out |
|---|---|---|---|---|
| Different row cap | `sqltool/schema.py`: `MAX_ROWS = 50` (generated SQL adds `LIMIT MAX_ROWS+1`) | Edit | `pytest tests/test_sqltool.py` | The truncation logic depends on the +1. |
| Allow a column or table | `sqltool/schema.py`: `ALLOWED_SCHEMA`, `BLOCKED_COLUMNS` | Add the column | `python src/sqltool/ask_sql.py --sql "SELECT ..."` (skips the LLM) | Never allow `admin_users`. Restart the service: the schema prompt is built once at startup. |
| Show the SQL prompt | `python src/sqltool/ask_sql.py --show-prompt` | None | n/a | Good for explaining how the model sees the schema. |
| Localize SQL answers to Arabic | `sqltool/format.py` `format_answer(result, language)` | Add Arabic column labels | A unit test | Currently a column dump in both languages (disclosed limitation). |
| More repairs | `sqltool/generate.py`: `answer_sql_question(..., max_repairs=1)` | Change the default | `ask_sql.py --dev-cases` (18 cases) | Each repair is another LLM call and more tokens. |

## 4. Service, LLM, tracing

| Request | Where | Change | Verify | Watch out |
|---|---|---|---|---|
| Different LLM | env: `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`; compose passes `LLM_MODEL` | Set the variable | `python src/llm/smoke.py` | The cache is keyed by the request body, so a new model makes real calls. `LLM_MIN_INTERVAL_S` paces calls. |
| Timeouts | `AgentConfig.tool_timeout_s = 45` (`agent/core.py`); `REQUEST_TIMEOUT_S` env (default 60) in `service/app.py`; SQL `timeout_s=2.0` | Edit | `pytest tests/test_service.py` (timeout test) | Keep tool timeout below the request timeout. |
| Concurrency | `AgentConfig.model_concurrency = 2`; the `Semaphore(1)` in `build_rag_tool` | Edit | A parallel `Invoke-RestMethod` smoke | Retrieval is serialized because encoder thread-safety is unverified. |
| New endpoint | `service/app.py` `create_app()` | Add `@app.get("/version")` | `TestClient` test, pattern in `tests/test_service.py` | Tests inject `FakeAgent`, no models load. |
| New response field | `agent/schemas.py` (shared base) plus `to_response` | Add the field | `pytest tests/test_schemas.py tests/test_service.py` | Responses use `extra="forbid"`, so an unknown field fails loudly. |
| New trace field | `service/trace_log.py` `build_record()` | Read it from `result.detail` | `tests/test_service.py::test_rag_answer_writes_trace_line` | Never log the query by default (`TRACE_LOG_QUERY`). |
| Change message wording | `agent/messages.py` `text_for(key, language)` | Edit the string | `pytest tests/test_agent_core.py` | Arabic wording was written by the assistant: review it. |
| Add an injection pattern | `rag/guard.py` `looks_like_injection` | Add the pattern | `pytest -q` | It only sets a flag; routing is unchanged. |

## 5. Bigger asks: answer in words first

- **Add a new tool:** write `async def tool(query, language)` returning an outcome object, add a route class and a `decide()` mapping, add a `_map_*` method in `agent/core.py`, build it in `agent/build.py`. A new route class also needs retraining, so say "I would first add a deterministic rule, then retrain on collected data".
- **Fix the 11 under-confident clarifies:** second routing signal (customer-name check for SQL, retrieval gate for RAG) then re-measure on a fresh eval set, not these 70 cases.
- **Serve on GPU or scale out:** one worker per replica, models per replica, retrieval serialization (see the production note).

## 6. Questions to rehearse (answers are in the report)

1. Why a separate routing taxonomy, and is it "using the Part A model"? It re-heads the Part A E5-small. MASSIVE intents are voice commands, so they carry no routing signal on in-domain queries; the intent is logged only.
2. Why 0.95? Dev sweep, rule amended after seeing the table, post hoc, dev-only.
3. Why not let the LLM pick the tool? Latency, determinism, testability, injection surface.
4. What is your weakest result? Routing: 11 of the 17 failures. Say what it costs (a safe non-answer) and what you would do (second signal).
5. Why were `rag_ar_10` and `rag_cross_04` dropped? The quoted text equals the chunk text apart from line breaks; verifier normalization; cause not confirmed; not fixed on test data.
6. How do you know there is no hallucination? 0/19 by quote verification, a number check, and a manual read; no LLM judge, n=19.