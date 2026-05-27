import argparse
import json
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from backend.app.database import SessionLocal, initialize_database
from backend.app.models import AgentRun, EvaluationResult
from backend.app.schemas import EvaluationRequest
from backend.app.services.evaluator import build_evaluation_summary, evaluate_agent_run


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_RUNS_PATH = PROJECT_ROOT / "data" / "sample_agent_runs.json"
DEFAULT_SAMPLE_EVALUATIONS_PATH = PROJECT_ROOT / "data" / "sample_evaluations.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def seed_database(sample_runs_path: Path, reset: bool = False) -> dict[str, Any]:
    initialize_database()
    sample_runs = load_json(sample_runs_path)

    with SessionLocal() as db:
        if reset:
            db.execute(delete(EvaluationResult))
            db.execute(delete(AgentRun))
            db.commit()

        created_results = []
        for item in sample_runs:
            payload = EvaluationRequest.model_validate(item)
            created_results.append(evaluate_agent_run(payload, db))

        summary = build_evaluation_summary(db)

    return {
        "inserted": len(created_results),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the local AgentEval database with sample evaluations.")
    parser.add_argument(
        "--sample-runs",
        type=Path,
        default=DEFAULT_SAMPLE_RUNS_PATH,
        help="Path to the sample agent runs JSON file.",
    )
    parser.add_argument(
        "--sample-evaluations",
        type=Path,
        default=DEFAULT_SAMPLE_EVALUATIONS_PATH,
        help="Path to sample evaluation metadata. Loaded to validate the companion fixture exists.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing agent runs and evaluation results before seeding.",
    )
    args = parser.parse_args()

    load_json(args.sample_evaluations)
    result = seed_database(args.sample_runs, reset=args.reset)
    summary = result["summary"]

    print(f"Inserted {result['inserted']} sample evaluations.")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average score: {summary['average_score']}")
    print(f"Pass rate: {summary['pass_rate']}")
    print(f"Human review count: {summary['human_review_count']}")
    print(f"Hallucination risk counts: {summary['hallucination_risk_counts']}")


if __name__ == "__main__":
    main()
