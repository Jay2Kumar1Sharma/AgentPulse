# AgentEval Dashboard

AgentEval Dashboard is a full-stack evaluation and monitoring platform for AI agent runs. It is designed to evaluate answer quality, tool-call accuracy, retrieval and context relevance, latency, cost, safety flags, hallucination risk, human-review requirements, and overall reliability.

## Phase 1 Status

This phase initializes the project structure, local development environment guidance, dependency manifest, secret handling template, and Git repository foundation.

## Project Goal

Build a production-style portfolio project for AI agent evaluation using FastAPI, SQLAlchemy, SQLite, Streamlit, and heuristic-first evaluation logic with optional LLM-as-a-judge support.

The app must work locally without paid API keys. Optional LLM judge integrations will use environment variables only and must safely fall back to heuristic evaluation when credentials are unavailable.

## Planned Architecture

```text
agent-eval-dashboard/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── routers/
│   │   └── utils/
│   └── tests/
├── dashboard/
│   └── components/
├── data/
├── docs/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Virtual Environment Setup

Create and activate a virtual environment before installing dependencies.

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

After activating the virtual environment, install dependencies with:

```bash
pip install -r requirements.txt
```

Do not install packages globally for this project.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`. It is ignored by Git and should only contain local secrets or machine-specific settings.

## Security Notes

- Do not hardcode API keys.
- Do not print, log, expose, or return API keys.
- Commit `.env.example` only.
- Keep `.env` local and ignored by Git.

## Next Phase

Phase 2 will add the FastAPI backend foundation, configuration loading, database connection setup, and a `/health` endpoint.
