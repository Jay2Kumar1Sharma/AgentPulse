# Evaluation Metrics

AgentEval uses heuristic metrics by default so the project works without API keys, internet access, or paid services. Optional LLM-as-a-judge evaluation can add a second opinion when enabled.

All numeric metric scores are normalized to `0.0` through `1.0`.

## Metric Reference

| Metric | Purpose | Logic |
| --- | --- | --- |
| `exact_match_score` | Checks strict answer equality | Normalizes case, punctuation, and whitespace, then compares expected and actual answers. |
| `keyword_overlap_score` | Measures answer similarity | Computes F1-style overlap between meaningful expected-answer and actual-answer terms. |
| `context_relevance_score` | Checks retrieval support | Measures how many actual-answer claim terms appear in retrieved context. |
| `answer_faithfulness_score` | Estimates grounding | Combines claim-term support with numeric claim support. |
| `tool_call_accuracy` | Checks agent tool behavior | Scores exact tool-call matches, gives partial credit for matching tool names, and penalizes extra calls. |
| `latency_score` | Tracks responsiveness | Full credit under 1 second, gradual penalty through 10 seconds, zero after 10 seconds. |
| `cost_score` | Tracks run efficiency | Full credit for very low cost, lower scores as cost increases. |
| `safety_score` | Tracks guardrail risk | Applies severity-weighted penalties for safety flags. |
| `hallucination_risk` | Flags unsupported claims | Uses unsupported claim terms and unsupported numeric facts to return `low`, `medium`, or `high`. |

## Scoring Details

### Exact Match

`1.0` if normalized expected and actual answers are identical, otherwise `0.0`.

### Keyword Overlap

Uses meaningful terms after lowercasing, punctuation removal, stopword removal, and light suffix normalization. The score is an F1-style balance of precision and recall.

### Context Relevance

Measures the ratio of actual-answer claim terms supported by retrieved context. Low values suggest retrieval mismatch or an answer not grounded in supplied context.

### Answer Faithfulness

Combines:

- lexical support from retrieved context
- numeric support for claims such as `30`, `99.9%`, or `1000`

Unsupported numeric claims reduce faithfulness and can raise hallucination risk.

### Tool-Call Accuracy

- exact match: `1.0`
- same tool name with different arguments: partial credit
- missing expected tools: `0.0`
- extra tool calls: penalty

### Latency

- `<= 1000 ms`: `1.0`
- `1001-3000 ms`: gradual decline to `0.7`
- `3001-10000 ms`: gradual decline to `0.2`
- `> 10000 ms`: `0.0`

### Cost

Cost scoring is intentionally simple for local demos:

- `<= $0.01`: `1.0`
- `<= $0.05`: `0.8`
- `<= $0.25`: `0.5`
- `<= $1.00`: `0.2`
- `> $1.00`: `0.0`

### Safety

Safety flags have weighted penalties. Higher-severity categories like self-harm, violence, hate, cyber-policy violations, and unsafe instructions reduce score more than generic flags.

## Overall Score

Weighted average:

| Metric | Weight |
| --- | ---: |
| exact match | 10% |
| keyword overlap | 15% |
| context relevance | 15% |
| answer faithfulness | 15% |
| tool-call accuracy | 20% |
| latency | 10% |
| cost | 5% |
| safety | 10% |

## Pass Criteria

An evaluation passes when:

- `overall_score >= 0.75`
- `safety_score >= 0.8`
- `tool_call_accuracy >= 0.6`

## Failure Taxonomy

| Category | Meaning |
| --- | --- |
| `PASSED` | Run passed required thresholds. |
| `RETRIEVAL_FAILURE` | Answer may be correct, but retrieved context does not support it. |
| `TOOL_CALL_FAILURE` | Tool use is missing, incorrect, or insufficient. |
| `HALLUCINATION_RISK` | Answer includes unsupported claims or numeric facts. |
| `SAFETY_RISK` | Safety flags indicate guardrail risk. |
| `LATENCY_ISSUE` | Run was too slow. |
| `COST_ISSUE` | Run was too expensive. |
| `ANSWER_MISMATCH` | Actual answer does not match expected answer closely enough. |

## Human Review Rules

Human review is required when any of these are true:

- score is borderline
- any safety flag exists
- tool-call accuracy is weak
- faithfulness is weak
- retrieval support is weak
- cost score is low
- hallucination risk is high

## Heuristic vs LLM Judge

Heuristic evaluation is deterministic, free, and always available. It is best for regression tests, local demos, and quick monitoring.

LLM-as-a-judge evaluation is optional. It can provide a more nuanced review of correctness, faithfulness, hallucination risk, safety concern, tool-call reasoning quality, and recommendations.

## Optional LLM Judge Output

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

## Fallback Behavior

If the judge is disabled, a key is missing, parsing fails, or the provider call fails, AgentEval safely falls back to heuristic evaluation and records that fallback in `optional_llm_judge_result`.

Example fallback:

```json
{
  "llm_judge_used": false,
  "warning": "LLM judge is enabled but no valid API key was found. Falling back to heuristic evaluation.",
  "judge_score": null,
  "faithfulness_score": null,
  "hallucination_risk": null,
  "safety_concern": null,
  "reasoning_summary": "LLM judge was skipped because API credentials were unavailable.",
  "recommendation": "Add the required API key in your local .env file or disable LLM_JUDGE_ENABLED."
}
```

## Limitations

- Heuristic overlap cannot fully understand semantic equivalence.
- Context relevance is based on term and numeric support, not deep entailment.
- Tool-call scoring is schema-light and intended for local demo reliability.
- Safety flags are assumed to be supplied by the agent runner or policy layer.
- LLM judging may vary by provider/model and should complement, not replace, deterministic checks.
