# Architecture

AgentEval Dashboard is a local-first evaluation system for AI agent runs. It separates data collection, scoring, persistence, and visualization so each layer can be tested and extended independently.

## System Design

```text
User / Reviewer
  |
  v
Streamlit Dashboard
  |
  | HTTP JSON
  v
FastAPI Backend
  |
  +--> Evaluation Router
  |      POST /evaluations/run
  |      GET  /evaluations
  |      GET  /evaluations/{id}
  |      GET  /evaluations/summary
  |
  +--> Evaluator Service
  |      heuristic metrics
  |      optional LLM judge
  |      failure taxonomy
  |      human-review decision
  |
  +--> SQLAlchemy ORM
         AgentRun
         EvaluationResult
  |
  v
SQLite database
```

## Data Flow

1. A user submits an agent run through the dashboard or API.
2. FastAPI validates the request with Pydantic schemas.
3. The evaluator stores the raw `AgentRun`.
4. Heuristic metrics calculate deterministic scores.
5. If enabled, the LLM judge evaluates the same run using provider credentials from environment variables.
6. If the judge is disabled, missing credentials, or fails, the evaluator stores a safe fallback result.
7. The final `EvaluationResult` is saved in SQLite.
8. The dashboard fetches summary, table, failed cases, charts, and individual details from the API.

## Backend Architecture

- `config.py`: loads `.env` with `python-dotenv` and exposes typed runtime settings.
- `database.py`: creates the SQLAlchemy engine/session factory and initializes SQLite tables.
- `models/evaluation.py`: defines `AgentRun` and `EvaluationResult`.
- `schemas/evaluation.py`: defines request and response contracts.
- `services/metrics.py`: contains heuristic metric calculations and failure taxonomy.
- `services/evaluator.py`: orchestrates persistence, scoring, optional LLM judging, and summary aggregation.
- `services/llm_judge.py`: resolves provider credentials, calls selected providers, validates JSON, and handles fallback.
- `routers/evaluations.py`: exposes the evaluation API.

## Dashboard Architecture

The Streamlit dashboard uses the FastAPI backend as its only data source.

Main sections:

- Summary metrics
- Evaluation submission form
- Evaluation table
- Average metric breakdown chart
- Hallucination risk distribution
- Failed case review table
- Single evaluation inspector

The dashboard can target another backend by setting `AGENTEVAL_API_URL`.

## Evaluation Flow

```text
EvaluationRequest
  -> AgentRun row
  -> exact match
  -> keyword overlap
  -> context relevance
  -> answer faithfulness
  -> tool-call accuracy
  -> latency score
  -> cost score
  -> safety score
  -> hallucination risk
  -> weighted overall score
  -> pass/fail
  -> failure category
  -> recommendation
  -> EvaluationResult row
```

## LLM Judge Flow

```text
LLM_JUDGE_ENABLED=false
  -> skip provider call
  -> heuristic-only result

LLM_JUDGE_ENABLED=true
  -> resolve LLM_PROVIDER
  -> select matching API key
  -> if key missing: safe fallback
  -> call provider
  -> require valid JSON
  -> validate fields
  -> store structured judge result
```

Supported providers:

- `openai` with `OPENAI_API_KEY`
- `anthropic` with `ANTHROPIC_API_KEY`
- `gemini` or `google` with `GOOGLE_API_KEY`

## Fallback Behavior

The app remains usable without internet or API keys.

Fallback happens when:

- `LLM_JUDGE_ENABLED=false`
- selected provider key is missing
- provider call raises an error
- provider returns invalid JSON
- returned fields fail validation

Fallback records a structured warning in `optional_llm_judge_result` and continues using heuristic scores.

## Storage

SQLite is used for local development:

- `agent_runs` stores raw inputs and metadata.
- `evaluation_results` stores metric scores, pass/fail state, taxonomy, recommendations, and optional LLM judge output.

JSON-like fields are stored as text for SQLite simplicity and converted back into Python structures in response schemas.

## Security

- Secrets are loaded only from environment variables.
- `.env` is ignored by Git.
- `.env.example` contains placeholders only.
- API keys are never returned in API responses or shown in the dashboard.
- LLM prompts request concise reasoning summaries only, not hidden chain-of-thought.
