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
            timestamp as created_at,
            endpoint,
            session_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost as cost_usd,
            latency_ms,
            metadata
        FROM metrics
        WHERE endpoint = 'chat'
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading chat metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_agent_metrics():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT
            id,
            timestamp as created_at,
            endpoint,
            session_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost as cost_usd,
            latency_ms,
            metadata
        FROM metrics
        WHERE endpoint = 'agent'
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading agent metrics: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_all_metrics():
    try:
        engine = create_engine(get_connection_string())
        query = """
        SELECT
            id,
            timestamp as created_at,
            endpoint,
            session_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost as cost_usd,
            latency_ms,
            metadata
        FROM metrics
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading all metrics: {e}")
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
agent_df = load_agent_metrics()
all_metrics_df = load_all_metrics()
benchmark_df = load_benchmark_latencies()

# Add endpoint filter
st.sidebar.header("Filters")
endpoint_filter = st.sidebar.selectbox("Select Endpoint", options=["All", "Chat", "Agent"], index=0)

# Apply filter
if endpoint_filter == "Chat":
    display_df = chat_df
elif endpoint_filter == "Agent":
    display_df = agent_df
else:
    display_df = all_metrics_df

# Overview Metrics Section
st.header("Overview Metrics")

if display_df.empty:
    st.info(f"No metrics data available for {endpoint_filter}. Use the API to generate data.")
else:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", len(display_df))
    with col2:
        st.metric("Avg Latency", f"{display_df['latency_ms'].mean():.0f} ms")
    with col3:
        st.metric("Total Cost", f"${display_df['cost_usd'].sum():.4f}")
    with col4:
        st.metric("Avg Tokens/Request", f"{display_df['total_tokens'].mean():.0f}")

st.markdown("---")

# Detailed Metrics Section
st.header(f"{endpoint_filter} Detailed Metrics")

if display_df.empty:
    st.info(f"No {endpoint_filter.lower()} metrics data available. Use the API to generate data.")
else:
    st.subheader("Latency Over Time")

    # If showing all endpoints, color by endpoint
    if endpoint_filter == "All" and not display_df.empty:
        fig_latency = px.line(
            display_df.sort_values("created_at"),
            x="created_at",
            y="latency_ms",
            color="endpoint",
            title="Response Latency by Endpoint",
            labels={"latency_ms": "Latency (ms)", "created_at": "Time", "endpoint": "Endpoint"},
        )
    else:
        fig_latency = px.line(
            display_df.sort_values("created_at"),
            x="created_at",
            y="latency_ms",
            title=f"{endpoint_filter} Response Latency",
            labels={"latency_ms": "Latency (ms)", "created_at": "Time"},
        )
        fig_latency.add_hline(
            y=display_df["latency_ms"].mean(), line_dash="dash", annotation_text="Average", line_color="red"
        )

    st.plotly_chart(fig_latency, use_container_width=True)

    st.subheader("Cost Over Time")
    display_df_sorted = display_df.sort_values("created_at").copy()

    # If showing all endpoints, show separate cumulative costs
    if endpoint_filter == "All" and "endpoint" in display_df_sorted.columns:
        fig_cost = go.Figure()
        for endpoint in display_df_sorted["endpoint"].unique():
            endpoint_df = display_df_sorted[display_df_sorted["endpoint"] == endpoint].copy()
            endpoint_df["cumulative_cost"] = endpoint_df["cost_usd"].cumsum()
            fig_cost.add_trace(
                go.Scatter(
                    x=endpoint_df["created_at"],
                    y=endpoint_df["cumulative_cost"],
                    mode="lines",
                    name=endpoint.title(),
                    fill="tonexty" if endpoint != display_df_sorted["endpoint"].unique()[0] else "tozeroy",
                )
            )
        fig_cost.update_layout(
            title="Cumulative Cost by Endpoint", xaxis_title="Time", yaxis_title="Cumulative Cost ($)"
        )
    else:
        display_df_sorted["cumulative_cost"] = display_df_sorted["cost_usd"].cumsum()
        fig_cost = px.area(
            display_df_sorted,
            x="created_at",
            y="cumulative_cost",
            title=f"Cumulative Cost (USD) - {endpoint_filter}",
            labels={"cumulative_cost": "Cumulative Cost ($)", "created_at": "Time"},
        )

    st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("Token Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_tokens = px.histogram(
            display_df,
            x="total_tokens",
            nbins=30,
            title="Total Tokens per Request",
            labels={"total_tokens": "Total Tokens"},
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    with col2:
        fig_breakdown = go.Figure()
        fig_breakdown.add_trace(go.Box(y=display_df["prompt_tokens"], name="Prompt"))
        fig_breakdown.add_trace(go.Box(y=display_df["completion_tokens"], name="Completion"))
        fig_breakdown.update_layout(title="Token Breakdown")
        st.plotly_chart(fig_breakdown, use_container_width=True)

    # Endpoint comparison if showing all
    if endpoint_filter == "All" and not display_df.empty and "endpoint" in display_df.columns:
        st.subheader("Endpoint Comparison")

        comparison_df = (
            display_df.groupby("endpoint")
            .agg(
                {
                    "latency_ms": ["mean", "median", "min", "max"],
                    "cost_usd": "sum",
                    "total_tokens": "mean",
                    "id": "count",
                }
            )
            .reset_index()
        )

        comparison_df.columns = [
            "Endpoint",
            "Avg Latency",
            "Median Latency",
            "Min Latency",
            "Max Latency",
            "Total Cost",
            "Avg Tokens",
            "Request Count",
        ]

        col1, col2 = st.columns(2)

        with col1:
            fig_comparison_latency = px.bar(
                comparison_df,
                x="Endpoint",
                y="Avg Latency",
                title="Average Latency by Endpoint",
                labels={"Avg Latency": "Latency (ms)"},
            )
            st.plotly_chart(fig_comparison_latency, use_container_width=True)

        with col2:
            fig_comparison_cost = px.bar(
                comparison_df,
                x="Endpoint",
                y="Total Cost",
                title="Total Cost by Endpoint",
                labels={"Total Cost": "Cost ($)"},
            )
            st.plotly_chart(fig_comparison_cost, use_container_width=True)

        st.dataframe(
            comparison_df.style.format(
                {
                    "Avg Latency": "{:.2f} ms",
                    "Median Latency": "{:.2f} ms",
                    "Min Latency": "{:.2f} ms",
                    "Max Latency": "{:.2f} ms",
                    "Total Cost": "${:.4f}",
                    "Avg Tokens": "{:.0f}",
                    "Request Count": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

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
