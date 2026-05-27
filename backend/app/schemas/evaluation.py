import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentRunCreate(BaseModel):
    user_query: str
    expected_answer: str
    actual_answer: str
    retrieved_context: str = ""
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    actual_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    safety_flags: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(AgentRunCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

    @field_validator("expected_tool_calls", "actual_tool_calls", "safety_flags", mode="before")
    @classmethod
    def parse_json_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return []
        return value

    @field_validator("metadata_json", mode="before")
    @classmethod
    def parse_json_object(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value


class EvaluationResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_run_id: int
    exact_match_score: float
    keyword_overlap_score: float
    context_relevance_score: float
    answer_faithfulness_score: float
    tool_call_accuracy: float
    latency_score: float
    cost_score: float
    safety_score: float
    hallucination_risk: str
    human_review_required: bool
    failure_category: str
    overall_score: float
    passed: bool
    failure_reason: str
    recommendation: str
    llm_judge_used: bool
    llm_judge_score: float | None = None
    llm_judge_summary: str | None = None
    optional_llm_judge_result: dict[str, Any] | None = None
    created_at: datetime
    agent_run: AgentRunResponse | None = None


class EvaluationRequest(AgentRunCreate):
    pass


class HallucinationRiskCounts(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0


class EvaluationSummaryResponse(BaseModel):
    total_runs: int
    average_score: float
    pass_rate: float
    average_latency: float
    safety_violations: int
    average_tool_call_accuracy: float
    human_review_count: int
    hallucination_risk_counts: HallucinationRiskCounts
