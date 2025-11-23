"""
RAG Evaluation Dashboard

Main entry point for the Streamlit dashboard showing:
- Latency and cost metrics over time
- Retrieval accuracy curves
"""

import os

import streamlit as st

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAG Evaluation Dashboard")
st.markdown("---")

st.markdown("""
## Welcome to the RAG Evaluation Dashboard

This dashboard provides insights into your RAG system's performance:

### Pages

- **Metrics**: View latency and cost metrics over time for chat interactions
- **Evaluation**: Analyze retrieval accuracy curves from benchmark runs

### Getting Started

1. Run benchmarks with `--save` flag to persist results:
   ```bash
   python -m ai_unifier_assesment.benchmark --save
   ```

2. Use the chat API to generate metrics data

3. Navigate to the pages in the sidebar to view visualizations

### Configuration

The dashboard connects to PostgreSQL using these environment variables:
- `POSTGRES_HOST` (default: localhost)
- `POSTGRES_PORT` (default: 5432)
- `POSTGRES_USER` (default: rag_user)
- `POSTGRES_PASSWORD` (default: rag_password)
- `POSTGRES_DB` (default: rag_evaluation)
""")

# Show connection status
postgres_host = os.getenv("POSTGRES_HOST", "localhost")
postgres_port = os.getenv("POSTGRES_PORT", "5432")
postgres_db = os.getenv("POSTGRES_DB", "rag_evaluation")

st.sidebar.markdown("### Database Connection")
st.sidebar.info(f"Host: {postgres_host}:{postgres_port}\nDB: {postgres_db}")
