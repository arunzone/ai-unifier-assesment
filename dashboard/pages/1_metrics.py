"""
Metrics Page - Latency and Cost over time
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Metrics", page_icon="📈", layout="wide")
st.title("Latency & Cost Metrics")


def get_connection_string():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "rag_user")
    password = os.getenv("POSTGRES_PASSWORD", "rag_password")
    db = os.getenv("POSTGRES_DB", "rag_evaluation")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@st.cache_data(ttl=60)
def load_chat_metrics():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT
            id,
            session_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            latency_ms,
            created_at
        FROM chat_metrics
        ORDER BY created_at DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading chat metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_benchmark_latencies():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT
            run_id,
            median_retrieval_time_ms,
            avg_retrieval_time_ms,
            min_retrieval_time_ms,
            max_retrieval_time_ms,
            created_at
        FROM benchmark_runs
        ORDER BY created_at ASC
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading benchmark latencies: {e}")
        return pd.DataFrame()


# Load data
chat_df = load_chat_metrics()
benchmark_df = load_benchmark_latencies()

# Chat Metrics Section
st.header("Chat Interaction Metrics")

if chat_df.empty:
    st.info("No chat metrics data available. Use the chat API to generate data.")
else:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Interactions", len(chat_df))
    with col2:
        st.metric("Avg Latency", f"{chat_df['latency_ms'].mean():.0f} ms")
    with col3:
        st.metric("Total Cost", f"${chat_df['cost_usd'].sum():.4f}")
    with col4:
        st.metric("Avg Tokens/Turn", f"{chat_df['total_tokens'].mean():.0f}")

    st.subheader("Latency Over Time")
    fig_latency = px.line(
        chat_df.sort_values("created_at"),
        x="created_at",
        y="latency_ms",
        title="Chat Response Latency",
        labels={"latency_ms": "Latency (ms)", "created_at": "Time"},
    )
    fig_latency.add_hline(y=chat_df["latency_ms"].mean(), line_dash="dash", annotation_text="Average", line_color="red")
    st.plotly_chart(fig_latency, use_container_width=True)

    st.subheader("Cost Over Time")
    chat_df_sorted = chat_df.sort_values("created_at")
    chat_df_sorted["cumulative_cost"] = chat_df_sorted["cost_usd"].cumsum()

    fig_cost = px.area(
        chat_df_sorted,
        x="created_at",
        y="cumulative_cost",
        title="Cumulative Cost (USD)",
        labels={"cumulative_cost": "Cumulative Cost ($)", "created_at": "Time"},
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("Token Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_tokens = px.histogram(
            chat_df,
            x="total_tokens",
            nbins=30,
            title="Total Tokens per Interaction",
            labels={"total_tokens": "Total Tokens"},
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    with col2:
        fig_breakdown = go.Figure()
        fig_breakdown.add_trace(go.Box(y=chat_df["prompt_tokens"], name="Prompt"))
        fig_breakdown.add_trace(go.Box(y=chat_df["completion_tokens"], name="Completion"))
        fig_breakdown.update_layout(title="Token Breakdown")
        st.plotly_chart(fig_breakdown, use_container_width=True)

st.markdown("---")

# Benchmark Latencies Section
st.header("Retrieval Latency Trends")

if benchmark_df.empty:
    st.info("No benchmark data available. Run benchmarks with --save flag.")
else:
    st.subheader("Retrieval Time Over Benchmark Runs")

    fig_retrieval = go.Figure()
    fig_retrieval.add_trace(
        go.Scatter(
            x=benchmark_df["created_at"],
            y=benchmark_df["median_retrieval_time_ms"],
            mode="lines+markers",
            name="Median",
            line=dict(color="blue", width=2),
        )
    )
    fig_retrieval.add_trace(
        go.Scatter(
            x=benchmark_df["created_at"],
            y=benchmark_df["avg_retrieval_time_ms"],
            mode="lines+markers",
            name="Average",
            line=dict(color="green", width=2),
        )
    )
    fig_retrieval.add_trace(
        go.Scatter(
            x=benchmark_df["created_at"],
            y=benchmark_df["max_retrieval_time_ms"],
            mode="lines",
            name="Max",
            line=dict(color="red", dash="dash"),
        )
    )
    fig_retrieval.add_trace(
        go.Scatter(
            x=benchmark_df["created_at"],
            y=benchmark_df["min_retrieval_time_ms"],
            mode="lines",
            name="Min",
            line=dict(color="orange", dash="dash"),
        )
    )

    # Add 300ms requirement line
    fig_retrieval.add_hline(
        y=300,
        line_dash="dot",
        annotation_text="300ms Requirement",
        line_color="purple",
    )

    fig_retrieval.update_layout(
        title="Retrieval Latency Trends",
        xaxis_title="Time",
        yaxis_title="Retrieval Time (ms)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_retrieval, use_container_width=True)

# Refresh button
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()
