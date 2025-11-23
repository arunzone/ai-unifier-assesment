"""
Evaluation Page - Retrieval Accuracy Curves
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Evaluation", page_icon="🎯", layout="wide")
st.title("Retrieval Accuracy Evaluation")


def get_connection_string():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "rag_user")
    password = os.getenv("POSTGRES_PASSWORD", "rag_password")
    db = os.getenv("POSTGRES_DB", "rag_evaluation")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@st.cache_data(ttl=60)
def load_benchmark_runs():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT
            id,
            run_id,
            top_k,
            total_questions,
            hits,
            accuracy_percent,
            median_retrieval_time_ms,
            avg_retrieval_time_ms,
            meets_latency_requirement,
            details,
            created_at
        FROM benchmark_runs
        ORDER BY created_at ASC
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading benchmark runs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_evaluation_questions():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT COUNT(*) as count FROM evaluation_questions
        """
        df = pd.read_sql(query, engine)
        return df.iloc[0]["count"] if not df.empty else 0
    except Exception:
        return 0


# Load data
df = load_benchmark_runs()
question_count = load_evaluation_questions()

# Summary metrics
st.header("Benchmark Summary")

if df.empty:
    st.info("No benchmark data available. Run benchmarks with --save flag to see results.")
    st.code("python -m ai_unifier_assesment.benchmark --save", language="bash")
else:
    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]

    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        st.metric("Latest Accuracy", f"{latest['accuracy_percent']}%")
    with col3:
        st.metric("Evaluation Questions", question_count)
    with col4:
        status = "PASS" if latest["meets_latency_requirement"] else "FAIL"
        st.metric("Latency Requirement", status)

    st.markdown("---")

    # Accuracy over time
    st.header("Retrieval Accuracy Trends")

    fig_accuracy = go.Figure()
    fig_accuracy.add_trace(
        go.Scatter(
            x=df["created_at"],
            y=df["accuracy_percent"],
            mode="lines+markers",
            name="Accuracy %",
            line=dict(color="blue", width=3),
            marker=dict(size=10),
        )
    )

    fig_accuracy.update_layout(
        title="Top-K Retrieval Accuracy Over Time",
        xaxis_title="Benchmark Run Time",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
    )

    # Add target line at 90%
    fig_accuracy.add_hline(
        y=90,
        line_dash="dash",
        annotation_text="90% Target",
        line_color="green",
    )

    st.plotly_chart(fig_accuracy, use_container_width=True)

    # Hits vs Misses breakdown
    st.header("Hit Rate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of hits over runs
        fig_hits = go.Figure()
        fig_hits.add_trace(
            go.Bar(
                x=df["run_id"],
                y=df["hits"],
                name="Hits",
                marker_color="green",
            )
        )
        fig_hits.add_trace(
            go.Bar(
                x=df["run_id"],
                y=df["total_questions"] - df["hits"],
                name="Misses",
                marker_color="red",
            )
        )
        fig_hits.update_layout(
            title="Hits vs Misses per Run",
            xaxis_title="Run ID",
            yaxis_title="Count",
            barmode="stack",
        )
        st.plotly_chart(fig_hits, use_container_width=True)

    with col2:
        # Latest run pie chart
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Hits", "Misses"],
                    values=[latest["hits"], latest["total_questions"] - latest["hits"]],
                    marker_colors=["green", "red"],
                    hole=0.4,
                )
            ]
        )
        fig_pie.update_layout(title=f"Latest Run ({latest['run_id']})")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Latency vs Accuracy scatter
    st.header("Latency vs Accuracy Correlation")

    fig_scatter = px.scatter(
        df,
        x="median_retrieval_time_ms",
        y="accuracy_percent",
        size="total_questions",
        color="meets_latency_requirement",
        hover_data=["run_id", "hits"],
        title="Retrieval Latency vs Accuracy",
        labels={
            "median_retrieval_time_ms": "Median Retrieval Time (ms)",
            "accuracy_percent": "Accuracy (%)",
            "meets_latency_requirement": "Meets 300ms Requirement",
        },
    )

    # Add requirement line
    fig_scatter.add_vline(
        x=300,
        line_dash="dash",
        annotation_text="300ms Requirement",
        line_color="purple",
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed results table
    st.header("Benchmark Run History")

    display_df = df[
        [
            "run_id",
            "top_k",
            "total_questions",
            "hits",
            "accuracy_percent",
            "median_retrieval_time_ms",
            "meets_latency_requirement",
            "created_at",
        ]
    ].copy()
    display_df["meets_latency_requirement"] = display_df["meets_latency_requirement"].map({1: "Yes", 0: "No"})
    display_df = display_df.rename(
        columns={
            "run_id": "Run ID",
            "top_k": "Top-K",
            "total_questions": "Questions",
            "hits": "Hits",
            "accuracy_percent": "Accuracy %",
            "median_retrieval_time_ms": "Median Latency (ms)",
            "meets_latency_requirement": "Meets Requirement",
            "created_at": "Time",
        }
    )

    st.dataframe(display_df, use_container_width=True)

    # Per-question details for latest run
    st.header("Latest Run - Per Question Details")

    if latest["details"]:
        details = latest["details"]
        if isinstance(details, list) and len(details) > 0:
            details_df = pd.DataFrame(details)

            # Color-coded hit/miss
            fig_details = px.bar(
                details_df,
                x="question_id",
                y="retrieval_time_ms",
                color="hit",
                title=f"Per-Question Retrieval Time (Run: {latest['run_id']})",
                labels={
                    "question_id": "Question ID",
                    "retrieval_time_ms": "Retrieval Time (ms)",
                    "hit": "Hit",
                },
                color_discrete_map={True: "green", False: "red"},
            )

            fig_details.add_hline(
                y=300,
                line_dash="dash",
                annotation_text="300ms",
                line_color="purple",
            )

            st.plotly_chart(fig_details, use_container_width=True)

            # Show missed questions
            missed = details_df[~details_df["hit"]]
            if not missed.empty:
                st.subheader("Missed Questions")
                st.dataframe(missed[["question_id", "question", "retrieval_time_ms"]], use_container_width=True)
        else:
            st.info("No detailed results available for this run.")

# Refresh button
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()
