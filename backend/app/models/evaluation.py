from datetime import datetime, timezone
import json
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def from_json_text(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_context: Mapped[str] = mapped_column(Text, nullable=False)
    expected_tool_calls: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    actual_tool_calls: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    safety_flags: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    evaluation_result: Mapped["EvaluationResult"] = relationship(
        back_populates="agent_run",
        cascade="all, delete-orphan",
        uselist=False,
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    exact_match_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_overlap_score: Mapped[float] = mapped_column(Float, nullable=False)
    context_relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    answer_faithfulness_score: Mapped[float] = mapped_column(Float, nullable=False)
    tool_call_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    latency_score: Mapped[float] = mapped_column(Float, nullable=False)
    cost_score: Mapped[float] = mapped_column(Float, nullable=False)
    safety_score: Mapped[float] = mapped_column(Float, nullable=False)
    hallucination_risk: Mapped[str] = mapped_column(String(20), nullable=False)
    human_review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_category: Mapped[str] = mapped_column(String(50), nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_reason: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    llm_judge_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    llm_judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_judge_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_judge_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    agent_run: Mapped[AgentRun] = relationship(back_populates="evaluation_result")

    @property
    def optional_llm_judge_result(self) -> dict[str, Any] | None:
        if not self.llm_judge_result_json:
            return None
        return from_json_text(self.llm_judge_result_json, None)
