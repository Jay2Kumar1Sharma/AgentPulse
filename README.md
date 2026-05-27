# AgentEval Dashboard

AgentEval Dashboard is a full-stack evaluation and monitoring platform for AI agent runs. It scores answer quality, tool-call accuracy, retrieval/context relevance, latency, cost, safety flags, hallucination risk, human-review requirement, and overall reliability.

The project is designed for an AI Agent Engineering portfolio: it demonstrates FastAPI, SQLAlchemy, SQLite, Streamlit, heuristic AI evaluation, optional LLM-as-a-judge evaluation, secure API key handling, sample data, and tests.

## Why This Project Matters

AI agents need more than final-answer checks. Production teams monitor whether agents used the right tools, grounded responses in retrieved context, avoided unsafe behavior, stayed within latency/cost budgets, and escalated risky cases for human review. This project turns those concerns into a local, inspectable dashboard.

## Features

- FastAPI backend with typed request/response models
- SQLite persistence through SQLAlchemy ORM
- Heuristic metrics that work without API keys
- Optional LLM-as-a-judge support for OpenAI, Anthropic, and Google Gemini
- Safe fallback when LLM judging is disabled, unavailable, or misconfigured
- Streamlit dashboard for summary metrics, failure review, charts, run inspection, and new run submission
- Realistic seed data for demo-ready local runs
- Pytest coverage for metrics and LLM judge fallback behavior

## Tech Stack

- Python 3.11+
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite
- Streamlit
- Pandas
- Requests
- pytest
- python-dotenv
- Optional SDKs: `openai`, `anthropic`, `google-generativeai`

## Architecture

```text
Dashboard (Streamlit)
  -> HTTP requests
FastAPI backend
  -> Routers
  -> Evaluator service
  -> Heuristic metrics + optional LLM judge
  -> SQLAlchemy ORM
SQLite database
```

Project layout:

```text
backend/app/
  config.py              Environment-based settings
  database.py            SQLAlchemy engine, sessions, SQLite initialization
  main.py                FastAPI app and router wiring
  models/                AgentRun and EvaluationResult ORM models
  schemas/               Pydantic API schemas
  services/              Metrics, evaluator, optional LLM judge
  routers/               Evaluation API endpoints
  utils/seed_data.py     Sample data loader
dashboard/
  streamlit_app.py       Streamlit dashboard
data/
  sample_agent_runs.json
  sample_evaluations.json
docs/
  architecture.md
  evaluation_metrics.md
```

## Setup

Create and activate a virtual environment before installing dependencies.

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Do not install packages globally.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```powershell
Copy-Item .env.example .env
```

Required local settings:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
LLM_PROVIDER=openai
LLM_JUDGE_ENABLED=false
LLM_JUDGE_MODEL=gpt-4o-mini
DATABASE_URL=sqlite:///./agent_eval.db
```

`.env` is ignored by Git. Commit only `.env.example`.

## Run Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"AgentEval Dashboard API"}
```

## Seed Sample Data

```powershell
.\venv\Scripts\python.exe -B -m backend.app.utils.seed_data --reset
```

This loads 10 demo cases covering correct answers, slow latency, wrong answers, hallucination, missing tools, extra tools, unsafe responses, poor retrieval, high cost, and human-review cases.

## Run Dashboard

```powershell
.\venv\Scripts\streamlit.exe run dashboard\streamlit_app.py
```

The dashboard reads from `http://127.0.0.1:8000` by default. Override with:

```powershell
$env:AGENTEVAL_API_URL="http://127.0.0.1:8000"
```

## API Examples

Run a new evaluation:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/evaluations/run -ContentType "application/json" -Body '{ "user_query": "What is the refund window?", "expected_answer": "Customers can request refunds within 30 days.", "actual_answer": "Customers can request refunds within 30 days.", "retrieved_context": "Refunds are available within 30 days.", "expected_tool_calls": [], "actual_tool_calls": [], "latency_ms": 500, "cost_usd": 0.01, "safety_flags": [], "metadata_json": {"case_id":"demo"} }'
```

List evaluations:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/evaluations
```

Dashboard summary:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/evaluations/summary
```

## Evaluation Metrics

The evaluator computes:

- exact match
- keyword overlap
- context relevance
- answer faithfulness
- tool-call accuracy
- latency score
- cost score
- safety score
- hallucination risk
- pass/fail status
- failure category
- failure reason
- recommendation
- human-review flag

See [docs/evaluation_metrics.md](docs/evaluation_metrics.md) for formulas, pass criteria, limitations, and LLM judge behavior.

## Optional LLM Judge

By default:

```env
LLM_JUDGE_ENABLED=false
```

To enable:

1. Set `LLM_JUDGE_ENABLED=true`.
2. Set `LLM_PROVIDER` to `openai`, `anthropic`, or `gemini`.
3. Add only the matching key to your local `.env`.

If the selected key is missing or the provider call fails, the app falls back to heuristic evaluation and records a warning in `optional_llm_judge_result`. The system never returns API keys in logs, terminal output, API responses, dashboard UI, or docs.

## Test

```powershell
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider backend\tests
```

## Security Notes

- Do not hardcode API keys.
- Do not print, log, expose, or return API keys.
- Commit `.env.example`, never `.env`.
- Keep `venv/`, `*.db`, caches, and local secrets out of Git.

## Screenshots

Screenshots can be added after running the local dashboard:

- `docs/screenshots/dashboard-summary.png`
- `docs/screenshots/evaluation-inspector.png`
- `docs/screenshots/new-evaluation-form.png`

## Future Improvements

- Add Alembic migrations for production database evolution
- Add authentication for multi-user deployments
- Add CSV export for evaluation history
- Add batch evaluation uploads
- Add trend charts by day/model/agent version
- Add richer RAG metrics such as citation coverage and retrieval precision
- Add CI workflow for tests and linting
