# AgentEval Dashboard

AgentEval Dashboard is a full-stack evaluation and monitoring platform for AI agent runs. It evaluates answer quality, tool-call accuracy, retrieval/context relevance, latency, cost, safety flags, hallucination risk, human-review requirements, and overall reliability.

The project is built to work without paid API keys. Heuristic evaluation is always available, and optional LLM-as-a-judge evaluation can be enabled through local environment variables only.

## Features

- FastAPI backend with SQLite persistence
- SQLAlchemy models for agent runs and evaluation results
- Heuristic scoring for answer quality, retrieval relevance, faithfulness, tools, latency, cost, safety, and hallucination risk
- Optional LLM-as-a-judge support for OpenAI, Anthropic, and Google Gemini
- Streamlit dashboard with summary metrics, failure review, charts, run inspection, and new evaluation submission
- Seed script with realistic sample agent runs
- Pytest coverage for core metrics and LLM judge fallback behavior

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

Do not install packages globally for this project.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```powershell
Copy-Item .env.example .env
```

`.env` is ignored by Git. Never commit it.

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

## LLM Judge

By default, `LLM_JUDGE_ENABLED=false`, so the app uses heuristic evaluation only.

To enable optional LLM judging:

1. Set `LLM_JUDGE_ENABLED=true`.
2. Set `LLM_PROVIDER` to `openai`, `anthropic`, or `gemini`.
3. Add only the matching API key to your local `.env`.

If the judge is enabled but the required key is missing, invalid, or the provider call fails, the app falls back to heuristic evaluation and returns a clear warning in `optional_llm_judge_result`. API keys are never printed, logged, returned, or committed.

## Run Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Seed Sample Data

```powershell
.\venv\Scripts\python.exe -B -m backend.app.utils.seed_data --reset
```

## Run Dashboard

```powershell
.\venv\Scripts\streamlit.exe run dashboard\streamlit_app.py
```

The dashboard reads from `http://127.0.0.1:8000` by default. Override with:

```powershell
$env:AGENTEVAL_API_URL="http://127.0.0.1:8000"
```

## Test

```powershell
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider backend\tests
```

## Security Notes

- Do not hardcode API keys.
- Do not print, log, expose, or return API keys.
- Commit `.env.example` only.
- Keep `.env`, local databases, virtual environments, and caches out of Git.
