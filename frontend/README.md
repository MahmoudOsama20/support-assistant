# Nile Wallet Support Assistant — Production Web UI

A responsive, production-style Next.js web application for the Nile Wallet multilingual support assistant. It integrates directly with the existing FastAPI backend (`/chat` and `/health`) to demonstrate hybrid RAG with verified citations, guarded read-only SQL queries, refusal handling, and clarification flows in English and Arabic.

---

## Features

- **Multilingual Support (English & Arabic):** Full bidirectional UI with genuine RTL layout, custom Arabic typography, and instant language switching.
- **RAG Evidence & Citations:** Interactive source cards displaying verified document IDs, chunk IDs, section titles, and verbatim grounding snippets.
- **Structured SQL Tables:** Automatic rendering of database results with column headers, formatted nulls/booleans, row counters, horizontal responsiveness, and one-click CSV export.
- **Clarification & Refusal States:** Distinct visual representations for short/ambiguous queries (clarification) and out-of-scope/unsupported queries (refusal).
- **Technical Developer Panel:** Real-time inspection of backend `request_id`, classification intent (Part A), route confidence, execution flags, latency breakdown, and raw JSON response payload.
- **Session Management:** Client-side persistence in `localStorage` supporting multiple conversation sessions, title generation, deletion, and clean resets.
- **Service Health Monitoring:** Continuous backend health detection with actionable instructions when offline.
- **Dual Connection Modes:** Direct client API calls or secure same-origin Next.js proxy routes (`/api/chat` and `/api/health`) preventing CORS issues in deployed environments.
- **Dark Mode:** Polished dark and light themes with preference persistence.

---

## Tech Stack

- **Framework:** Next.js 16 (App Router)
- **Language:** TypeScript with strict type checking
- **Styling:** Tailwind CSS v4 with custom RTL typography
- **Icons:** Lucide React
- **Markdown:** ReactMarkdown with Remark-GFM
- **Testing:** Vitest & React Testing Library (12 unit and component tests)
- **Notifications:** Sonner toasts with RTL support

---

## Prerequisites

- **Node.js:** v20+ (tested on v24.18)
- **Python:** 3.12+ (for running the existing backend service)

---

## Quick Start

### 1. Start the FastAPI Backend

In the project root, activate your Python virtual environment and launch the backend:

```powershell
.\venv\Scripts\Activate.ps1
python src/serve.py
```

The backend starts at `http://127.0.0.1:8000`. Verify with:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 2. Start the Frontend

In a separate terminal, navigate to `frontend/`:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Configuration

Copy `.env.example` to `.env.local`:

```powershell
cp .env.example .env.local
```

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | The backend FastAPI service URL used by the server-side proxy route. |
| `NEXT_PUBLIC_API_BASE_URL` | *(empty)* | Optional: Set to `http://127.0.0.1:8000` to call the backend directly from the browser instead of using the Next.js proxy route. |

---

## Available Scripts

From the `frontend` directory:

```powershell
npm run dev          # Start Next.js development server on port 3000
npm run build        # Build optimized production bundle
npm run start        # Run production server
npm run lint         # Run ESLint (passes with 0 errors, 0 warnings)
npm run type-check   # Run TypeScript strict type verification
npm run test         # Run Vitest unit & component test suite
```

---

## Verification & Test Results

All test suites and production build checks pass:

- **Type Check:** `npm run type-check` — 0 errors
- **Linter:** `npm run lint` — 0 errors, 0 warnings
- **Unit Tests:** `npm run test` — 3 test files, 12 tests passed
- **Production Build:** `npm run build` — Successful Turbopack build generating static and dynamic routes
- **Backend Tests:** `pytest -q` — 364 passed in 40s (backend intact)

---

## Architecture & Integration

```
[Browser Client]
       │
       ▼
 [Next.js App Router (Port 3000)]
  ├── / (Main Chat UI, RTL/LTR, Theme, State)
  ├── /api/chat   ──(HTTP Proxy with timeout)──▶ [FastAPI Backend (Port 8000)]
  └── /api/health ──(HTTP Proxy)───────────────▶  ├── Route Classifier (E5)
                                                  ├── Hybrid RAG (BGE-M3 + BM25)
                                                  ├── SQLite (Read-Only)
                                                  └── Groq LLM
```

---

## Deployment to Vercel

1. Import the repository into Vercel.
2. Set the **Root Directory** to `frontend`.
3. Add the Environment Variable:
   - `BACKEND_URL`: URL of your hosted FastAPI backend (e.g., on Render, Fly.io, or Railway).
4. Deploy! Next.js will serve the UI and proxy API calls seamlessly without exposing private credentials or incurring CORS issues.
