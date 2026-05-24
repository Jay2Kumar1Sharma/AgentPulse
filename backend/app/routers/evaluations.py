from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResultResponse,
    EvaluationSummaryResponse,
)
from backend.app.services.evaluator import (
    build_evaluation_summary,
    evaluate_agent_run,
    get_evaluation,
    list_evaluations,
)


router = APIRouter(prefix="/evaluations", tags=["evaluations"])
DbSession = Annotated[Session, Depends(get_db)]


@router.post("/run", response_model=EvaluationResultResponse, status_code=status.HTTP_201_CREATED)
def run_evaluation(payload: EvaluationRequest, db: DbSession) -> EvaluationResultResponse:
    return evaluate_agent_run(payload, db)


@router.get("", response_model=list[EvaluationResultResponse])
def get_evaluations(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[EvaluationResultResponse]:
    return list_evaluations(db, limit=limit, offset=offset)


@router.get("/summary", response_model=EvaluationSummaryResponse)
def get_evaluation_summary(db: DbSession) -> EvaluationSummaryResponse:
    return build_evaluation_summary(db)


@router.get("/{evaluation_id}", response_model=EvaluationResultResponse)
def get_evaluation_detail(evaluation_id: int, db: DbSession) -> EvaluationResultResponse:
    evaluation = get_evaluation(db, evaluation_id)
    if evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation {evaluation_id} was not found.",
        )
    return evaluation
