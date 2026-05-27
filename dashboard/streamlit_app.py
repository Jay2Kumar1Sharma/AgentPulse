import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("AGENTEVAL_API_URL", "http://127.0.0.1:8000")
METRIC_COLUMNS = [
    "exact_match_score",
    "keyword_overlap_score",
    "context_relevance_score",
    "answer_faithfulness_score",
    "tool_call_accuracy",
    "latency_score",
    "cost_score",
    "safety_score",
]


st.set_page_config(page_title="AgentEval Dashboard", page_icon="AE", layout="wide")


def fetch_json(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10)
def load_dashboard_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = fetch_json("/evaluations/summary")
    evaluations = fetch_json("/evaluations")
    return summary, evaluations


def flatten_evaluations(evaluations: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        agent_run = evaluation.get("agent_run") or {}
        rows.append(
            {
                "evaluation_id": evaluation.get("id"),
                "query": agent_run.get("user_query", ""),
                "overall_score": evaluation.get("overall_score"),
                "passed": evaluation.get("passed"),
                "latency_ms": agent_run.get("latency_ms"),
                "safety_score": evaluation.get("safety_score"),
                "hallucination_risk": evaluation.get("hallucination_risk"),
                "human_review_required": evaluation.get("human_review_required"),
                "failure_category": evaluation.get("failure_category"),
                "created_at": evaluation.get("created_at"),
            }
        )
    return pd.DataFrame(rows)


def metric_breakdown_frame(evaluations: list[dict[str, Any]]) -> pd.DataFrame:
    if not evaluations:
        return pd.DataFrame(columns=["metric", "average_score"])

    frame = pd.DataFrame(evaluations)
    rows = [
        {
            "metric": metric.replace("_", " ").replace("score", "").strip().title(),
            "average_score": round(float(frame[metric].mean()), 4),
        }
        for metric in METRIC_COLUMNS
        if metric in frame
    ]
    return pd.DataFrame(rows).set_index("metric")


def format_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return "None"
    return json.dumps(value, indent=2)


def parse_json_input(raw_value: str, fallback: Any, expected_type: type) -> Any:
    if not raw_value.strip():
        return fallback
    parsed = json.loads(raw_value)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} JSON.")
    return parsed


def parse_safety_flags(raw_value: str) -> list[str]:
    return [flag.strip() for flag in raw_value.split(",") if flag.strip()]


def render_submission_form() -> dict[str, Any] | None:
    with st.expander("Add Evaluation", expanded=False):
        with st.form("new_evaluation_form"):
            text_columns = st.columns(2)
            with text_columns[0]:
                user_query = st.text_area("User query", height=90)
                expected_answer = st.text_area("Expected answer", height=110)
                actual_answer = st.text_area("Actual answer", height=110)
            with text_columns[1]:
                retrieved_context = st.text_area("Retrieved context", height=180)
                expected_tool_calls_raw = st.text_area("Expected tool calls", value="[]", height=90)
                actual_tool_calls_raw = st.text_area("Actual tool calls", value="[]", height=90)

            numeric_columns = st.columns(3)
            latency_ms = numeric_columns[0].number_input("Latency ms", min_value=0, value=1000, step=100)
            cost_usd = numeric_columns[1].number_input("Cost USD", min_value=0.0, value=0.01, step=0.001, format="%.4f")
            safety_flags_raw = numeric_columns[2].text_input("Safety flags")
            metadata_raw = st.text_area("Metadata JSON", value="{}", height=80)

            submitted = st.form_submit_button("Evaluate Run", type="primary")

        if not submitted:
            return None

        try:
            payload = {
                "user_query": user_query,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "retrieved_context": retrieved_context,
                "expected_tool_calls": parse_json_input(expected_tool_calls_raw, [], list),
                "actual_tool_calls": parse_json_input(actual_tool_calls_raw, [], list),
                "latency_ms": int(latency_ms),
                "cost_usd": float(cost_usd),
                "safety_flags": parse_safety_flags(safety_flags_raw),
                "metadata_json": parse_json_input(metadata_raw, {}, dict),
            }
            result = post_json("/evaluations/run", payload)
        except (json.JSONDecodeError, ValueError) as error:
            st.error(f"Invalid form value: {error}")
            return None
        except requests.RequestException as error:
            st.error(f"Evaluation request failed: {error}")
            return None

        load_dashboard_data.clear()
        st.success(f"Evaluation {result['id']} created.")
        return result


def render_created_result(evaluation: dict[str, Any]) -> None:
    result_columns = st.columns(4)
    result_columns[0].metric("Overall score", f"{evaluation.get('overall_score', 0):.2f}")
    result_columns[1].metric("Passed", "Yes" if evaluation.get("passed") else "No")
    result_columns[2].metric("Failure category", evaluation.get("failure_category", ""))
    result_columns[3].metric("Human review", "Yes" if evaluation.get("human_review_required") else "No")
    st.write(evaluation.get("failure_reason", ""))
    st.write(evaluation.get("recommendation", ""))


def render_summary(summary: dict[str, Any]) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Total runs", summary.get("total_runs", 0))
    first_row[1].metric("Average score", f"{summary.get('average_score', 0):.2f}")
    first_row[2].metric("Pass rate", f"{summary.get('pass_rate', 0) * 100:.0f}%")
    first_row[3].metric("Average latency", f"{summary.get('average_latency', 0):.0f} ms")

    second_row = st.columns(3)
    second_row[0].metric("Safety violations", summary.get("safety_violations", 0))
    second_row[1].metric("Tool-call accuracy", f"{summary.get('average_tool_call_accuracy', 0):.2f}")
    second_row[2].metric("Human-review count", summary.get("human_review_count", 0))


def render_failed_cases(evaluations: list[dict[str, Any]]) -> None:
    failed = [evaluation for evaluation in evaluations if not evaluation.get("passed")]
    if not failed:
        st.success("All evaluated runs passed.")
        return

    failed_rows = []
    for evaluation in failed:
        agent_run = evaluation.get("agent_run") or {}
        failed_rows.append(
            {
                "evaluation_id": evaluation.get("id"),
                "query": agent_run.get("user_query", ""),
                "failure_category": evaluation.get("failure_category"),
                "failure_reason": evaluation.get("failure_reason"),
                "recommendation": evaluation.get("recommendation"),
            }
        )
    st.dataframe(pd.DataFrame(failed_rows), use_container_width=True, hide_index=True)


def render_inspector(evaluations: list[dict[str, Any]]) -> None:
    if not evaluations:
        st.info("No evaluations available.")
        return

    evaluation_by_id = {evaluation["id"]: evaluation for evaluation in evaluations}
    selected_id = st.selectbox("Evaluation ID", sorted(evaluation_by_id.keys(), reverse=True))
    evaluation = evaluation_by_id[selected_id]
    agent_run = evaluation.get("agent_run") or {}

    st.subheader("Run Details")
    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown("**Input query**")
        st.write(agent_run.get("user_query", ""))
        st.markdown("**Expected answer**")
        st.write(agent_run.get("expected_answer", ""))
        st.markdown("**Actual answer**")
        st.write(agent_run.get("actual_answer", ""))
    with detail_columns[1]:
        st.markdown("**Retrieved context**")
        st.write(agent_run.get("retrieved_context", ""))
        st.markdown("**Expected tool calls**")
        st.code(format_json(agent_run.get("expected_tool_calls")), language="python")
        st.markdown("**Actual tool calls**")
        st.code(format_json(agent_run.get("actual_tool_calls")), language="python")

    st.subheader("Metric Scores")
    metric_values = {
        metric.replace("_", " ").replace("score", "").strip().title(): evaluation.get(metric)
        for metric in METRIC_COLUMNS
    }
    st.dataframe(pd.DataFrame([metric_values]), use_container_width=True, hide_index=True)

    result_columns = st.columns(4)
    result_columns[0].metric("Overall score", f"{evaluation.get('overall_score', 0):.2f}")
    result_columns[1].metric("Passed", "Yes" if evaluation.get("passed") else "No")
    result_columns[2].metric("Hallucination risk", str(evaluation.get("hallucination_risk", "")).title())
    result_columns[3].metric("Human review", "Yes" if evaluation.get("human_review_required") else "No")

    st.subheader("Guardrail Signals")
    guardrail_columns = st.columns(4)
    guardrail_columns[0].metric("Failure category", evaluation.get("failure_category", ""))
    guardrail_columns[1].metric("Safety score", f"{evaluation.get('safety_score', 0):.2f}")
    guardrail_columns[2].metric("Tool accuracy", f"{evaluation.get('tool_call_accuracy', 0):.2f}")
    guardrail_columns[3].metric("Faithfulness", f"{evaluation.get('answer_faithfulness_score', 0):.2f}")

    st.markdown("**Failure reason**")
    st.write(evaluation.get("failure_reason", ""))
    st.markdown("**Recommendation**")
    st.write(evaluation.get("recommendation", ""))


st.title("AgentEval Dashboard")
st.caption("Evaluation and monitoring platform for AI agent reliability.")

created_result = render_submission_form()
if created_result is not None:
    render_created_result(created_result)

try:
    summary_data, evaluation_data = load_dashboard_data()
except requests.RequestException as error:
    st.error(f"Could not reach AgentEval API at {API_BASE_URL}.")
    st.code(str(error))
    st.stop()

render_summary(summary_data)

st.divider()
st.subheader("Evaluation Table")
evaluation_frame = flatten_evaluations(evaluation_data)
if evaluation_frame.empty:
    st.info("No evaluations found. Seed sample data or submit a run through the API.")
else:
    st.dataframe(evaluation_frame, use_container_width=True, hide_index=True)

st.divider()
chart_columns = st.columns(2)
with chart_columns[0]:
    st.subheader("Metric Breakdown")
    breakdown = metric_breakdown_frame(evaluation_data)
    if breakdown.empty:
        st.info("No metric data available.")
    else:
        st.bar_chart(breakdown)

with chart_columns[1]:
    st.subheader("Hallucination Risk Distribution")
    risk_counts = summary_data.get("hallucination_risk_counts", {})
    risk_frame = pd.DataFrame(
        [{"risk": risk.title(), "count": count} for risk, count in risk_counts.items()]
    ).set_index("risk")
    st.bar_chart(risk_frame)

st.divider()
st.subheader("Failed Cases")
render_failed_cases(evaluation_data)

st.divider()
st.subheader("Single Evaluation Inspector")
render_inspector(evaluation_data)
