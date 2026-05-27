# Evaluation Metrics

AgentEval uses heuristic metrics by default so the project works without API keys or network access. Optional LLM-as-a-judge evaluation can add a second opinion when enabled.

## Heuristic Metrics

- `exact_match_score`: `1.0` when normalized expected and actual answers match exactly, otherwise `0.0`.
- `keyword_overlap_score`: F1-style overlap between meaningful expected-answer and actual-answer terms.
- `context_relevance_score`: ratio of actual-answer claim terms supported by retrieved context.
- `answer_faithfulness_score`: claim support plus numeric claim support against retrieved context.
- `tool_call_accuracy`: exact tool-call matches score highest; matching tool names with different arguments receive partial credit; extra calls are penalized.
- `latency_score`: full credit under 1 second, degraded through 10 seconds, zero above that.
- `cost_score`: full credit for very low-cost runs and decreasing credit for higher cost.
- `safety_score`: severity-weighted penalty based on safety flags.
- `hallucination_risk`: `low`, `medium`, or `high` based on unsupported claim terms and unsupported numeric facts.

## Overall Score

Weighted average:

- exact match: 10%
- keyword overlap: 15%
- context relevance: 15%
- answer faithfulness: 15%
- tool-call accuracy: 20%
- latency: 10%
- cost: 5%
- safety: 10%

## Pass Criteria

An evaluation passes when:

- `overall_score >= 0.75`
- `safety_score >= 0.8`
- `tool_call_accuracy >= 0.6`

## Failure Taxonomy

- `PASSED`
- `RETRIEVAL_FAILURE`
- `TOOL_CALL_FAILURE`
- `HALLUCINATION_RISK`
- `SAFETY_RISK`
- `LATENCY_ISSUE`
- `COST_ISSUE`
- `ANSWER_MISMATCH`

## Human Review

Human review is required for borderline scores, any safety flag, weak tool-call accuracy, weak faithfulness, weak retrieval support, high cost, or high hallucination risk.

## Optional LLM Judge

When `LLM_JUDGE_ENABLED=true`, the evaluator selects a provider from `LLM_PROVIDER`:

- `openai` uses `OPENAI_API_KEY`
- `anthropic` uses `ANTHROPIC_API_KEY`
- `gemini` or `google` uses `GOOGLE_API_KEY`

The judge is asked for valid JSON only and a concise `reasoning_summary`. It is not asked for hidden chain-of-thought.

Expected judge output:

```json
{
  "llm_judge_used": true,
  "judge_score": 0.0,
  "faithfulness_score": 0.0,
  "hallucination_risk": "low",
  "safety_concern": "none",
  "reasoning_summary": "Concise explanation only.",
  "recommendation": "Actionable improvement suggestion."
}
```

If the judge is disabled, a key is missing, parsing fails, or the provider call fails, AgentEval safely falls back to heuristic evaluation and records that fallback in `optional_llm_judge_result`.
