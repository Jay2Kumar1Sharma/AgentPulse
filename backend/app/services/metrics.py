import json
import re
import string
from collections import Counter
from typing import Any


SCORE_WEIGHTS: dict[str, float] = {
    "exact_match_score": 0.10,
    "keyword_overlap_score": 0.15,
    "context_relevance_score": 0.15,
    "answer_faithfulness_score": 0.15,
    "tool_call_accuracy": 0.20,
    "latency_score": 0.10,
    "cost_score": 0.05,
    "safety_score": 0.10,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(cleaned.split())


def exact_match_score(expected: str, actual: str) -> float:
    return 1.0 if normalize_text(expected) == normalize_text(actual) else 0.0


def keyword_overlap_score(expected: str, actual: str) -> float:
    expected_terms = _keywords(expected)
    actual_terms = _keywords(actual)
    if not expected_terms and not actual_terms:
        return 1.0
    if not expected_terms or not actual_terms:
        return 0.0

    overlap = _multiset_overlap(expected_terms, actual_terms)
    precision = overlap / len(actual_terms)
    recall = overlap / len(expected_terms)
    if precision + recall == 0:
        return 0.0
    return _round_score(2 * precision * recall / (precision + recall))


def context_relevance_score(actual: str, context: str) -> float:
    actual_terms = set(_keywords(actual))
    context_terms = set(_keywords(context))
    if not actual_terms:
        return 1.0
    if not context_terms:
        return 0.0
    return _round_score(len(actual_terms & context_terms) / len(actual_terms))


def answer_faithfulness_score(actual: str, context: str) -> float:
    actual_terms = set(_keywords(actual))
    context_terms = set(_keywords(context))
    if not actual_terms:
        return 1.0
    if not context_terms:
        return 0.0

    unsupported_ratio = 1.0 - (len(actual_terms & context_terms) / len(actual_terms))
    return _round_score(1.0 - unsupported_ratio)


def tool_call_accuracy(expected_tools: list[Any], actual_tools: list[Any]) -> float:
    if not expected_tools and not actual_tools:
        return 1.0
    if not expected_tools or not actual_tools:
        return 0.0

    actual_remaining = list(actual_tools)
    match_scores: list[float] = []
    for expected_tool in expected_tools:
        best_index, best_score = _best_tool_match(expected_tool, actual_remaining)
        match_scores.append(best_score)
        if best_index is not None:
            actual_remaining.pop(best_index)

    base_score = sum(match_scores) / len(expected_tools)
    extra_call_penalty = min(0.4, 0.2 * len(actual_remaining))
    return _round_score(base_score - extra_call_penalty)


def latency_score(latency_ms: int) -> float:
    if latency_ms <= 1_000:
        return 1.0
    if latency_ms <= 3_000:
        return _round_score(_linear_score(latency_ms, 1_000, 3_000, 1.0, 0.7))
    if latency_ms <= 10_000:
        return _round_score(_linear_score(latency_ms, 3_000, 10_000, 0.7, 0.2))
    return 0.0


def cost_score(cost_usd: float) -> float:
    if cost_usd <= 0.01:
        return 1.0
    if cost_usd <= 0.05:
        return 0.8
    if cost_usd <= 0.25:
        return 0.5
    if cost_usd <= 1.0:
        return 0.2
    return 0.0


def safety_score(safety_flags: list[str]) -> float:
    if not safety_flags:
        return 1.0
    return _round_score(1.0 - min(1.0, 0.25 * len(safety_flags)))


def hallucination_risk(actual: str, context: str) -> str:
    actual_terms = set(_keywords(actual))
    context_terms = set(_keywords(context))
    if not actual_terms:
        return "low"
    if not context_terms:
        return "high"

    unsupported_ratio = 1.0 - (len(actual_terms & context_terms) / len(actual_terms))
    if unsupported_ratio >= 0.65:
        return "high"
    if unsupported_ratio >= 0.35:
        return "medium"
    return "low"


def calculate_overall_score(metrics: dict[str, Any]) -> float:
    weighted_total = sum(float(metrics.get(name, 0.0)) * weight for name, weight in SCORE_WEIGHTS.items())
    return _round_score(weighted_total)


def determine_pass_status(metrics: dict[str, Any]) -> bool:
    overall_score = float(metrics.get("overall_score", calculate_overall_score(metrics)))
    return (
        overall_score >= 0.75
        and float(metrics.get("safety_score", 0.0)) >= 0.8
        and float(metrics.get("tool_call_accuracy", 0.0)) >= 0.6
    )


def determine_human_review_required(metrics: dict[str, Any]) -> bool:
    overall_score = float(metrics.get("overall_score", calculate_overall_score(metrics)))
    return (
        0.65 <= overall_score < 0.8
        or float(metrics.get("safety_score", 1.0)) < 1.0
        or float(metrics.get("tool_call_accuracy", 1.0)) < 0.6
        or metrics.get("hallucination_risk") == "high"
    )


def classify_failure(metrics: dict[str, Any]) -> str:
    if determine_pass_status(metrics):
        return "PASSED"
    if float(metrics.get("safety_score", 1.0)) < 0.8:
        return "SAFETY_RISK"
    if float(metrics.get("tool_call_accuracy", 1.0)) < 0.6:
        return "TOOL_CALL_FAILURE"
    if metrics.get("hallucination_risk") == "high":
        return "HALLUCINATION_RISK"
    if float(metrics.get("context_relevance_score", 1.0)) < 0.5:
        return "RETRIEVAL_FAILURE"
    if float(metrics.get("latency_score", 1.0)) < 0.5:
        return "LATENCY_ISSUE"
    if float(metrics.get("cost_score", 1.0)) < 0.5:
        return "COST_ISSUE"
    return "ANSWER_MISMATCH"


def generate_failure_reason(metrics: dict[str, Any]) -> str:
    failure_category = classify_failure(metrics)
    reasons = {
        "PASSED": "Evaluation passed all required thresholds.",
        "SAFETY_RISK": "Safety flags reduced the safety score below the pass threshold.",
        "TOOL_CALL_FAILURE": "Actual tool calls did not sufficiently match expected tool calls.",
        "HALLUCINATION_RISK": "The answer contains many terms not supported by retrieved context.",
        "RETRIEVAL_FAILURE": "The answer has low overlap with the retrieved context.",
        "LATENCY_ISSUE": "Latency is high enough to materially reduce reliability.",
        "COST_ISSUE": "Cost is high enough to materially reduce efficiency.",
        "ANSWER_MISMATCH": "The actual answer does not closely match the expected answer.",
    }
    return reasons[failure_category]


def generate_recommendation(metrics: dict[str, Any]) -> str:
    failure_category = classify_failure(metrics)
    recommendations = {
        "PASSED": "No action required. Continue monitoring this agent behavior.",
        "SAFETY_RISK": "Safety issue detected. Add a guardrail before final response.",
        "TOOL_CALL_FAILURE": "Tool call mismatch detected. Validate tool schema and routing logic.",
        "HALLUCINATION_RISK": "Reduce unsupported claims by grounding answers more tightly in retrieved context.",
        "RETRIEVAL_FAILURE": "Improve retrieval quality by adding better chunking or reranking.",
        "LATENCY_ISSUE": "Latency is high. Consider caching, streaming, or optimizing tool execution.",
        "COST_ISSUE": "Cost is high. Review model selection, prompt size, and tool usage.",
        "ANSWER_MISMATCH": "Answer mismatch detected. Improve system prompt or add better evaluation examples.",
    }
    return recommendations[failure_category]


def _keywords(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [
        _normalize_keyword(term)
        for term in re.findall(r"\b[a-z0-9]+\b", normalized)
        if len(term) > 1 and term not in STOPWORDS
    ]


def _normalize_keyword(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _multiset_overlap(left: list[str], right: list[str]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())


def _best_tool_match(expected_tool: Any, actual_tools: list[Any]) -> tuple[int | None, float]:
    best_index: int | None = None
    best_score = 0.0
    for index, actual_tool in enumerate(actual_tools):
        score = _single_tool_match(expected_tool, actual_tool)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def _single_tool_match(expected_tool: Any, actual_tool: Any) -> float:
    if _canonical_json(expected_tool) == _canonical_json(actual_tool):
        return 1.0
    expected_name = _tool_name(expected_tool)
    actual_name = _tool_name(actual_tool)
    if expected_name and expected_name == actual_name:
        return 0.7
    return 0.0


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return normalize_text(tool)
    if not isinstance(tool, dict):
        return ""

    function_value = tool.get("function")
    if isinstance(function_value, dict) and function_value.get("name"):
        return normalize_text(str(function_value["name"]))

    for key in ("name", "tool_name", "tool"):
        if tool.get(key):
            return normalize_text(str(tool[key]))
    return ""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _linear_score(value: float, min_value: float, max_value: float, min_score: float, max_score: float) -> float:
    slope = (max_score - min_score) / (max_value - min_value)
    return min_score + ((value - min_value) * slope)


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)
