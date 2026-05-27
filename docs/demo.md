# Demo Guide

Use this checklist for a clean local portfolio demo.

## One-Time Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Keep `LLM_JUDGE_ENABLED=false` for a no-key demo.

## Seed Data

```powershell
.\venv\Scripts\python.exe -B -m backend.app.utils.seed_data --reset
```

## Start Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Start Dashboard

Open a second terminal:

```powershell
venv\Scripts\activate
.\venv\Scripts\streamlit.exe run dashboard\streamlit_app.py
```

## Demo Flow

1. Show summary metrics and pass/fail distribution.
2. Open the failed cases table and explain the failure taxonomy.
3. Inspect one hallucination or retrieval failure.
4. Submit a new evaluation from the dashboard form.
5. Refresh the summary and show the new run in the table.
6. Explain that optional LLM judging is environment-driven and safely falls back when no key is present.

## Screenshot Placeholders

Save final screenshots in `docs/screenshots/`:

- `dashboard-summary.png`
- `evaluation-inspector.png`
- `new-evaluation-form.png`
