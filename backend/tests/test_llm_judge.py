from backend.app.config import Settings
from backend.app.schemas import EvaluationRequest
from backend.app.services.llm_judge import evaluate_with_llm_judge


def _payload() -> EvaluationRequest:
    return EvaluationRequest(
        user_query="What is the refund window?",
        expected_answer="Customers can request refunds within 30 days.",
        actual_answer="Customers can request refunds within 30 days.",
        retrieved_context="Refunds are available within 30 days.",
        expected_tool_calls=[],
        actual_tool_calls=[],
        latency_ms=500,
        cost_usd=0.01,
        safety_flags=[],
        metadata_json={},
    )


def _metrics() -> dict[str, object]:
    return {
        "overall_score": 1.0,
        "answer_faithfulness_score": 1.0,
        "hallucination_risk": "low",
        "safety_score": 1.0,
    }


def test_llm_judge_disabled_uses_heuristic_only(monkeypatch) -> None:
    monkeypatch.setenv("LLM_JUDGE_ENABLED", "false")
    settings = Settings()

    result = evaluate_with_llm_judge(_payload(), _metrics(), settings=settings)

    assert result.llm_judge_used is False
    assert result.warning is None
    assert result.judge_score is None
    assert "disabled" in result.reasoning_summary.lower()


def test_llm_judge_enabled_without_key_returns_safe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LLM_JUDGE_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings()

    result = evaluate_with_llm_judge(_payload(), _metrics(), settings=settings)

    assert result.llm_judge_used is False
    assert result.warning == "LLM judge is enabled but no valid API key was found. Falling back to heuristic evaluation."
    assert result.judge_score is None
    assert "credentials" in result.reasoning_summary.lower()
