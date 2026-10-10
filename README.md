# Support Assistant: intent model + agentic RAG service

AI Engineer take-home. A fine-tuned intent classifier (Part A) and an async service (Parts B-D) that routes each request to hybrid RAG with citations, a safe read-only SQL tool, a clarification or a refusal. Domain: **Nile Wallet**, a fictional Egyptian wallet company. The KB, database and routing data are synthetic.

Results: `report/results_report.md` (Part A detail: `report/part_a_results.md`). Production note: `docs/production_note.md`.

## Architecture

```
Client -> FastAPI (/chat, request-id middleware)
  -> preprocess -> short-query rule (<3 words, no digit -> clarify)
  -> route model (E5-small, 4 classes) + intent model (MASSIVE, logged only), concurrently
  -> deterministic policy decide(): rag | sql | clarify | refuse
       rag    : BM25 + bge-m3 dense -> RRF -> bge-reranker -> evidence gate -> LLM claims + verbatim-quote check -> citations
       sql    : LLM JSON -> sqlglot validator -> authorizer -> read-only SQLite (<=50 rows)
  -> Pydantic response validation -> JSON + one JSONL trace line (logs/traces.jsonl)
```

The LLM never picks the tool. Invalid response shapes become `error/invalid_response`, never a half-formed answer.

## Setup (Windows PowerShell, Python 3.12)

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/supportdb/seed.py                 # data/db/support.db (deterministic)
python src/kb/docs.py --expect-count 32      # validates the KB
python scripts/hf_models.py fetch --repo-id MahmoudOsama20/nile-wallet-classifiers   # classifier weights, about 1 GB
python scripts/download_massive.py           # MASSIVE (needed by the Part A tests)
```

Verified on a fresh clone in this order: `pip install`, `seed.py`, `fetch`, `download_massive.py`, `pytest` gives 364 passed. The first service start also downloads bge-m3 and the reranker (several GB) and builds the dense index. `download_massive.py` rewrites the timestamp in the tracked `data/massive/MANIFEST.json`; restore it with `git checkout data/massive/MANIFEST.json` if you do not want that change.

Create `.env` at the repo root (gitignored) with `GROQ_API_KEY=...`. Optional: `LLM_MODEL` (default `openai/gpt-oss-120b`; the results use it), `LLM_MIN_INTERVAL_S` (pacing for free-tier limits).

**Model weights are not in git.** The service needs `results/runs/e5-small-e15/model` (intent) and `results/runs/route-e5small/model` (route). Fetch them from the Hugging Face Hub (about 1 GB): `python scripts/hf_models.py fetch --repo-id MahmoudOsama20/nile-wallet-classifiers` (the author's upload; the same script's `upload` command recreates it from local weights). The route model was trained on the data at git tag `route-data-v1` (`generate.py` at HEAD produces the rejected v2 data).

## Run

```powershell
python src/serve.py                          # http://127.0.0.1:8000
Invoke-RestMethod http://127.0.0.1:8000/health
$b = [Text.Encoding]::UTF8.GetBytes((@{query="what is the daily transfer limit?"} | ConvertTo-Json))
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json; charset=utf-8" -Body $b
python src/agent/ask_agent.py --json "اعرض رصيد منى عادل"      # CLI, no server
python src/service/trace_stats.py --exclude-cached              # p50/p95 per route
```

Endpoints: `POST /chat` (`{"query": "..."}`, 1-2000 chars), `GET /health`. Response types: `rag_answer`, `sql_answer`, `clarification`, `refusal`, `error`. Every response carries a `request_id` (also the `X-Request-ID` header).

## Docker

```powershell
docker compose build
docker compose run --rm -e HF_HUB_OFFLINE=0 api python src/download_models.py   # once; fills the hf_cache volume
docker compose up -d
docker compose ps                            # healthy after the models load (start_period 180 s)
```

CPU image. Models are not baked in. The DB and both classifiers are mounted read-only. Compose defaults `LLM_MODEL` to `openai/gpt-oss-20b` (dev); set `$env:LLM_MODEL` to match the results.

## Tests and evaluation

```powershell
python -m pytest -q                          # 364 passed (also on a fresh clone)
python src/eval/build_cases.py               # refuses to overwrite: the 70-case file is the test set
python src/eval/run_eval.py --name final-120b          # one scored run (spent; uses LLM quota)
python src/eval/retrieval_arms.py            # retrieval arms and the reranker ablation (no LLM)
python src/eval/latency_run.py --name latency-120b     # latency from the warm LLM cache
python src/eval/hallucination_check.py results/eval/final-120b.json
python src/eval/indirect_check.py            # component-level indirect-injection check
```

Part A training: configs `configs/classifier.yaml` (XLM-R) and `configs/classifier_e5small.yaml` (E5-small, shipped); entry point `src/classifier/train.py`; trained on Kaggle (2x T4). Results and error analysis: `report/part_a_results.md`. Shipped model:

```powershell
python src/classifier/train.py --config configs/classifier_e5small.yaml --run-name e5-small-e15 --epochs 15 --learning-rate 1e-4 --seed 42
```

(The config defaults are lr 5e-5 and 5 epochs; the shipped run overrode them.) Route model: `python src/route/train_route.py --run-name route-e5small` with `configs/route.yaml`, on the data at tag `route-data-v1`.

## Layout

`src/classifier` Part A. `src/kb`, `src/rag` knowledge base and retrieval. `src/sqltool` SQL tool. `src/route` routing data and model. `src/agent` policy, core, builder, schemas. `src/service` FastAPI and tracing. `src/eval` evaluation. `data/kb`, `data/route`, `data/eval` tracked data; `results/` metrics.

## Assumptions and limitations

- **Synthetic data:** KB (`data/kb/fact_sheet.md` is the source of truth), SQLite rows and routing data are invented. Business days are Mon-Fri.
- **Routing taxonomy:** MASSIVE intents are voice commands, so routing uses its own four classes. The Part A intent is logged only.
- **tau_route = 0.95** was chosen on dev after the sweep; the stated rule was amended to also require zero out_of_scope leaks. **tau = 0.05** (evidence gate) was calibrated on 24 dev queries.
- **Route v2** was tried and rejected under a rule written before its results; v1 ships. The reality-check set (42 hand-written rows) is dev-only.
- **Known weakness:** natural short questions can be refused or clarified by the route model (see the report). A second signal would fix it (production note).
- **LLM:** Groq free tier, gpt-oss-120b for the reported run, temperature 0, on-disk response cache. Quotas bound throughput.
- **SQL:** sqlglot 30.21.0. Answers are column dumps, not localized. The validator and authorizer layers are unit-tested but were not reached by any eval case.
- **Not built:** `/metrics`, `/trace/{id}`, multi-turn memory, auth, rate limiting, LLM-judge hallucination scoring.