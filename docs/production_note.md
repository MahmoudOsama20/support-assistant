# Production note (1 page)

## Deployment and scaling
- One uvicorn worker per container: models load once at startup (about 17 s locally, longer on CPU), then a warm-up request runs. Scale horizontally behind a load balancer; each replica needs its own copy of the models (about 2 GB for bge-m3 and the reranker, plus 2 x 0.5 GB classifiers).
- Inference runs in threads behind semaphores, and retrieval is serialized (thread safety of the shared encoder is UNVERIFIED). CPU retrieval took about 5.5 s in a single Docker sample versus about 0.3 s on the GPU host, so size replicas for the reranker, not the LLM.
- The LLM dominates cost and latency. Free-tier quotas (30 requests/min, 200K tokens/day per model) are a hard ceiling; production needs a paid tier or a self-hosted model.

## Monitoring (from the JSONL trace, one line per request)
- p50/p95 per route and per span (preprocess, models, rag, sql, llm). Judge latency only on uncached LLM calls; the cache and pacing sleeps otherwise distort it.
- Route mix, refusal and clarify rates, and the clarify rate for natural questions (over-clarify was 7/46 in the eval).
- Invalid-citation and dropped-claim rates, SQL rejection rate by code, LLM 429 and timeout rates, error rate by `error_code`.
- Drift: route confidence histogram, share of queries below tau_route, retrieval gate probability.
- Alerts: error rate, p95 above target, LLM 429 spike, sudden growth in refusals.

## Failure modes and responses
| Failure | Behaviour today | Next step |
|---|---|---|
| LLM down or quota exhausted | `error/llm_unavailable`, retryable, no retry storm | Extractive fallback: return cited chunks without generation |
| Route model under-confident or wrong on natural questions (measured: 11 of 17 eval failures) | Clarify or refuse | Second signal: rescue `out_of_scope` with a customer-name/ID check for SQL and the retrieval gate for RAG; retrain on real traffic |
| Claim verifier drops true claims (quotes of bullet-list chunks that differ from the text only in line breaks) | Refusal `no_evidence` | Find the normalization gap, make quote matching insensitive to line breaks and list markers; re-measure on a fresh set |
| Wrong-route answer (KB question answered with data) | Wrong answer, no signal | Cross-check SQL answers against route confidence; log for review |
| Poisoned or stale documents | Superseded docs dropped; retrieved text delimited as data; canary check | Ingest-time scan, document approval workflow |
| SQL timeout or schema drift | Refusal, or a loud startup failure for a missing DB | Schema check at startup; query timeout alerts |
| Index stale after a KB edit | Dense index rebuilt from a cache keyed on the chunks | Rebuild in CI and ship with the image |

## Security
Read-only SQL with allowlists and an authorizer, honeypot table, no LLM tools, secrets only in the environment, query text not logged by default (`TRACE_LOG_QUERY=1` opts in). Gaps: no authentication or per-user data scoping (any caller can ask about any customer), no rate limiting, no PII redaction in traces.

## Cost
Per RAG request about 830 prompt and 160 completion tokens (one observed call); LLM price not computed (free tier). The classifiers cost CPU only. Cost control: the evidence gate and the clarify/refuse routes skip the LLM entirely (28 of the 70 eval cases finished in under 0.5 s in pass 1, i.e. without an LLM call), and responses are cached.

## Updates and rollback
Version the models, KB, route data and prompts together (git tag plus an image tag). Roll back by redeploying the previous image; the DB is mounted read-only and untouched. Gate each model or KB update on the 70-case eval plus a fresh held-out sample; never tune on the eval set.