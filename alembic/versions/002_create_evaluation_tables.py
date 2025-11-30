"""create evaluation tables

Revision ID: 002
Revises: 001
Create Date: 2025-11-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create evaluation_questions table
    op.create_table(
        "evaluation_questions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("ground_truth_answer", sa.Text(), nullable=False),
        sa.Column("ground_truth_contexts", sa.JSON(), nullable=False),
        sa.Column("source_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    # Create benchmark_runs table
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=50), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("total_questions", sa.Integer(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False),
        sa.Column("accuracy_percent", sa.Float(), nullable=False),
        sa.Column("median_retrieval_time_ms", sa.Float(), nullable=False),
        sa.Column("avg_retrieval_time_ms", sa.Float(), nullable=False),
        sa.Column("min_retrieval_time_ms", sa.Float(), nullable=False),
        sa.Column("max_retrieval_time_ms", sa.Float(), nullable=False),
        sa.Column("meets_latency_requirement", sa.Integer(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
        if_not_exists=True,
    )

    # Create chat_metrics table
    op.create_table(
        "chat_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    # Create indexes
    op.create_index("ix_benchmark_runs_run_id", "benchmark_runs", ["run_id"], unique=True, if_not_exists=True)
    op.create_index("ix_chat_metrics_session_id", "chat_metrics", ["session_id"], unique=False, if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_chat_metrics_session_id", table_name="chat_metrics")
    op.drop_index("ix_benchmark_runs_run_id", table_name="benchmark_runs")
    op.drop_table("chat_metrics")
    op.drop_table("benchmark_runs")
    op.drop_table("evaluation_questions")
