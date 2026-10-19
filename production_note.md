# Production note

## Deployment and scaling
- One uvicorn worker per container; models load once at startup (about 17 s on the GPU host, longer on CPU), then a warm-up request runs. Scale horizontally; each replica holds its own models (about 2 GB for the retrieval models plus 2 x 0.5 GB classifiers). Retrieval is serialized (encoder thread-safety UNVERIFIED), and CPU retrieval took about 5.5 s in a single Docker sample versus about 0.3 s on the GPU host, so size replicas for the reranker.
- The LLM dominates cost and latency. Free-tier quotas (30 requests/min, 200K tokens/day per model) are a hard ceiling: production needs a paid tier or a self-hosted model.

## Monitoring (JSONL trace, one line per request)
p50/p95 per route and span (judge latency on uncached calls only); route mix, refusal and clarify rates; dropped-claim and invalid-citation rates; SQL rejections by code; LLM 429, timeout and error rates by `error_code`; drift in route confidence and retrieval gate probability. Alert on error rate, p95, 429 spikes and growth in refusals.

## Failure modes
| Failure | Behaviour today | Next step |
|---|---|---|
| LLM down or quota exhausted | `error/llm_unavailable`, retryable | Extractive fallback: cited chunks without generation |
| Route under-confident or wrong on natural questions (11 of 17 eval failures) | Clarify or refuse | Second signal: customer-name/ID check for SQL, retrieval gate for RAG; retrain on real traffic |
| Verifier drops true claims (bullet-list quotes that differ only in line breaks) | Refusal `no_evidence` | Find the normalization gap; re-measure on fresh data |
| Wrong-route answer (KB question answered with data) | Wrong answer, no signal | Cross-check SQL answers against route confidence; review log |
| Poisoned or stale docs, SQL timeout, stale index | Superseded docs dropped, text delimited, canary check, refusal, index rebuilt from a chunk-keyed cache | Ingest scan, schema check at startup, rebuild the index in CI |

## Security and cost
**Security:** read-only SQL (allowlists, authorizer, honeypot), no LLM tools, secrets in the environment, query text not logged by default. Gaps: no authentication or per-user data scoping, no rate limiting, no PII redaction in traces.
**Cost:** about 830 prompt and 160 completion tokens per RAG request (one observed call); price not computed (free tier). The evidence gate and the clarify/refuse routes skip the LLM (28 of 70 eval cases finished under 0.5 s), and responses are cached.

## Updates and rollback
Version models, KB, route data and prompts together; roll back by redeploying the previous image (the DB is mounted read-only). Gate each update on the eval plus fresh held-out data; never tune on the eval set.
