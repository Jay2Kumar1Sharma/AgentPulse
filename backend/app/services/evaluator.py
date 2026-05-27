from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.app.models.evaluation import AgentRun, EvaluationResult, to_json_text
from backend.app.schemas.evaluation import EvaluationRequest
from backend.app.services import metrics
from backend.app.services.llm_judge import evaluate_with_llm_judge


def evaluate_agent_run(payload: EvaluationRequest, db: Session) -> EvaluationResult:
    agent_run = AgentRun(
        user_query=payload.user_query,
        expected_answer=payload.expected_answer,
        actual_answer=payload.actual_answer,
        retrieved_context=payload.retrieved_context,
        expected_tool_calls=to_json_text(payload.expected_tool_calls),
        actual_tool_calls=to_json_text(payload.actual_tool_calls),
        latency_ms=payload.latency_ms,
        cost_usd=payload.cost_usd,
        safety_flags=to_json_text(payload.safety_flags),
        metadata_json=to_json_text(payload.metadata_json),
    )
    db.add(agent_run)
    db.flush()

    metric_values = calculate_heuristic_metrics(payload)
    llm_judge_result = evaluate_with_llm_judge(payload, metric_values)
    evaluation_result = EvaluationResult(
        agent_run=agent_run,
        exact_match_score=metric_values["exact_match_score"],
        keyword_overlap_score=metric_values["keyword_overlap_score"],
        context_relevance_score=metric_values["context_relevance_score"],
        answer_faithfulness_score=metric_values["answer_faithfulness_score"],
        tool_call_accuracy=metric_values["tool_call_accuracy"],
        latency_score=metric_values["latency_score"],
        cost_score=metric_values["cost_score"],
        safety_score=metric_values["safety_score"],
        hallucination_risk=metric_values["hallucination_risk"],
        human_review_required=metric_values["human_review_required"],
        failure_category=metric_values["failure_category"],
        overall_score=metric_values["overall_score"],
        passed=metric_values["passed"],
        failure_reason=metric_values["failure_reason"],
        recommendation=metric_values["recommendation"],
        llm_judge_used=llm_judge_result.llm_judge_used,
        llm_judge_score=llm_judge_result.judge_score,
        llm_judge_summary=llm_judge_result.reasoning_summary,
        llm_judge_result_json=to_json_text(llm_judge_result.to_dict()),
    )
    db.add(evaluation_result)
    db.commit()
    db.refresh(agent_run)
    db.refresh(evaluation_result)
    evaluation_result.agent_run = agent_run
    return evaluation_result


def calculate_heuristic_metrics(payload: EvaluationRequest) -> dict[str, float | str | bool]:
    metric_values: dict[str, float | str | bool] = {
        "exact_match_score": metrics.exact_match_score(payload.expected_answer, payload.actual_answer),
        "keyword_overlap_score": metrics.keyword_overlap_score(payload.expected_answer, payload.actual_answer),
        "context_relevance_score": metrics.context_relevance_score(payload.actual_answer, payload.retrieved_context),
        "answer_faithfulness_score": metrics.answer_faithfulness_score(payload.actual_answer, payload.retrieved_context),
        "tool_call_accuracy": metrics.tool_call_accuracy(payload.expected_tool_calls, payload.actual_tool_calls),
        "latency_score": metrics.latency_score(payload.latency_ms),
        "cost_score": metrics.cost_score(payload.cost_usd),
        "safety_score": metrics.safety_score(payload.safety_flags),
        "hallucination_risk": metrics.hallucination_risk(payload.actual_answer, payload.retrieved_context),
    }
    metric_values["overall_score"] = metrics.calculate_overall_score(metric_values)
    metric_values["passed"] = metrics.determine_pass_status(metric_values)
    metric_values["human_review_required"] = metrics.determine_human_review_required(metric_values)
    metric_values["failure_category"] = metrics.classify_failure(metric_values)
    metric_values["failure_reason"] = metrics.generate_failure_reason(metric_values)
    metric_values["recommendation"] = metrics.generate_recommendation(metric_values)
    return metric_values


def list_evaluations(db: Session, limit: int = 100, offset: int = 0) -> list[EvaluationResult]:
    statement = (
        select(EvaluationResult)
        .options(joinedload(EvaluationResult.agent_run))
        .order_by(EvaluationResult.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(statement))


def get_evaluation(db: Session, evaluation_id: int) -> EvaluationResult | None:
    statement = (
        select(EvaluationResult)
        .options(joinedload(EvaluationResult.agent_run))
        .where(EvaluationResult.id == evaluation_id)
    )
    return db.scalars(statement).first()


def build_evaluation_summary(db: Session) -> dict[str, object]:
    evaluations = list(
        db.scalars(select(EvaluationResult).options(joinedload(EvaluationResult.agent_run))).unique()
    )
    total_runs = len(evaluations)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "average_score": 0.0,
            "pass_rate": 0.0,
            "average_latency": 0.0,
            "safety_violations": 0,
            "average_tool_call_accuracy": 0.0,
            "human_review_count": 0,
            "hallucination_risk_counts": {"low": 0, "medium": 0, "high": 0},
        }

    hallucination_risk_counts = {"low": 0, "medium": 0, "high": 0}
    for evaluation in evaluations:
        if evaluation.hallucination_risk in hallucination_risk_counts:
            hallucination_risk_counts[evaluation.hallucination_risk] += 1

    return {
        "total_runs": total_runs,
        "average_score": _average(evaluation.overall_score for evaluation in evaluations),
        "pass_rate": round(sum(1 for evaluation in evaluations if evaluation.passed) / total_runs, 4),
        "average_latency": _average(evaluation.agent_run.latency_ms for evaluation in evaluations),
        "safety_violations": sum(1 for evaluation in evaluations if evaluation.safety_score < 1.0),
        "average_tool_call_accuracy": _average(evaluation.tool_call_accuracy for evaluation in evaluations),
        "human_review_count": sum(1 for evaluation in evaluations if evaluation.human_review_required),
        "hallucination_risk_counts": hallucination_risk_counts,
    }


def _average(values: Iterable[float | int]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(float(value) for value in collected) / len(collected), 4)
