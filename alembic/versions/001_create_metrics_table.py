"""create metrics table

Revision ID: 001
Revises:
Create Date: 2025-01-25

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("endpoint", sa.String(length=50), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cost", sa.DECIMAL(precision=10, scale=8), nullable=False),
        sa.Column("latency_ms", sa.DECIMAL(precision=10, scale=2), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_metrics_timestamp", "metrics", ["timestamp"], unique=False)
    op.create_index("ix_metrics_endpoint", "metrics", ["endpoint"], unique=False)
    op.create_index("ix_metrics_session_id", "metrics", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_metrics_session_id", table_name="metrics")
    op.drop_index("ix_metrics_endpoint", table_name="metrics")
    op.drop_index("ix_metrics_timestamp", table_name="metrics")
    op.drop_table("metrics")
