from backend.app.services.metrics import (
    answer_faithfulness_score,
    calculate_overall_score,
    classify_failure,
    context_relevance_score,
    cost_score,
    determine_human_review_required,
    determine_pass_status,
    exact_match_score,
    generate_failure_reason,
    generate_recommendation,
    hallucination_risk,
    keyword_overlap_score,
    latency_score,
    normalize_text,
    safety_score,
    tool_call_accuracy,
)


def test_normalize_text_removes_case_punctuation_and_extra_spaces() -> None:
    assert normalize_text("  Hello, WORLD!  ") == "hello world"


def test_exact_match_uses_normalized_text() -> None:
    assert exact_match_score("Paris is in France.", "paris is in france") == 1.0
    assert exact_match_score("Paris", "London") == 0.0


def test_keyword_overlap_rewards_shared_meaningful_terms() -> None:
    score = keyword_overlap_score(
        "The refund policy allows returns within 30 days.",
        "Customers can return items within 30 days under the refund policy.",
    )

    assert score > 0.6


def test_context_relevance_and_faithfulness_use_retrieved_context() -> None:
    context = "Paris is the capital city of France and sits on the Seine river."
    supported_answer = "Paris is the capital of France."
    unsupported_answer = "Paris is the capital of France and hosted the first Moon landing."

    assert context_relevance_score(supported_answer, context) == 1.0
    assert answer_faithfulness_score(supported_answer, context) == 1.0
    assert answer_faithfulness_score(unsupported_answer, context) < 1.0


def test_tool_call_accuracy_handles_exact_partial_missing_and_extra_calls() -> None:
    expected = [{"name": "search_docs", "arguments": {"query": "refund policy"}}]
    actual_exact = [{"name": "search_docs", "arguments": {"query": "refund policy"}}]
    actual_partial = [{"name": "search_docs", "arguments": {"query": "pricing"}}]
    actual_extra = actual_exact + [{"name": "calculator", "arguments": {"expression": "2+2"}}]

    assert tool_call_accuracy(expected, actual_exact) == 1.0
    assert tool_call_accuracy(expected, actual_partial) == 0.7
    assert tool_call_accuracy(expected, []) == 0.0
    assert tool_call_accuracy(expected, actual_extra) == 0.8


def test_latency_cost_and_safety_scores_are_bounded() -> None:
    assert latency_score(500) == 1.0
    assert 0.0 < latency_score(5_000) < 1.0
    assert latency_score(20_000) == 0.0

    assert cost_score(0.0) == 1.0
    assert cost_score(0.5) == 0.2
    assert cost_score(2.0) == 0.0

    assert safety_score([]) == 1.0
    assert safety_score(["self_harm"]) == 0.75
    assert safety_score(["a", "b", "c", "d", "e"]) == 0.0


def test_hallucination_risk_uses_unsupported_context_terms() -> None:
    context = "The order shipped on Tuesday with tracking number ABC123."
    low_risk_answer = "The order shipped on Tuesday."
    high_risk_answer = "The order was refunded, upgraded, and delivered by drone."

    assert hallucination_risk(low_risk_answer, context) == "low"
    assert hallucination_risk(high_risk_answer, context) == "high"


def test_overall_score_and_pass_status_follow_required_thresholds() -> None:
    metrics = {
        "exact_match_score": 1.0,
        "keyword_overlap_score": 1.0,
        "context_relevance_score": 1.0,
        "answer_faithfulness_score": 1.0,
        "tool_call_accuracy": 1.0,
        "latency_score": 1.0,
        "cost_score": 1.0,
        "safety_score": 1.0,
        "hallucination_risk": "low",
    }
    metrics["overall_score"] = calculate_overall_score(metrics)

    assert metrics["overall_score"] == 1.0
    assert determine_pass_status(metrics) is True

    metrics["tool_call_accuracy"] = 0.5
    metrics["overall_score"] = calculate_overall_score(metrics)

    assert determine_pass_status(metrics) is False


def test_failure_category_reason_and_recommendation_are_consistent() -> None:
    metrics = {
        "overall_score": 0.7,
        "safety_score": 1.0,
        "tool_call_accuracy": 0.2,
        "context_relevance_score": 0.9,
        "latency_score": 1.0,
        "cost_score": 1.0,
        "hallucination_risk": "low",
    }

    assert classify_failure(metrics) == "TOOL_CALL_FAILURE"
    assert "tool calls" in generate_failure_reason(metrics).lower()
    assert "tool" in generate_recommendation(metrics).lower()


def test_human_review_required_for_borderline_or_high_risk_cases() -> None:
    borderline_metrics = {
        "overall_score": 0.72,
        "safety_score": 1.0,
        "tool_call_accuracy": 1.0,
        "hallucination_risk": "low",
    }
    high_risk_metrics = {
        "overall_score": 0.9,
        "safety_score": 1.0,
        "tool_call_accuracy": 1.0,
        "hallucination_risk": "high",
    }

    assert determine_human_review_required(borderline_metrics) is True
    assert determine_human_review_required(high_risk_metrics) is True
