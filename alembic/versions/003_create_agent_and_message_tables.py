"""create agent and message tables

Revision ID: 003
Revises: 002
Create Date: 2025-11-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create agent_executions table
    op.create_table(
        "agent_executions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("task_description", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("working_directory", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("failure_reason", sa.String(length=100), nullable=True),
        sa.Column("error_details", sa.Text(), nullable=True),
        sa.Column("final_code_length", sa.Integer(), nullable=True),
        sa.Column("test_output", sa.Text(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("total_cost", sa.Float(), nullable=True),
        sa.Column("execution_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index("ix_agent_executions_session_id", "agent_executions", ["session_id"], unique=False, if_not_exists=True)
    op.create_index("ix_agent_executions_status", "agent_executions", ["status"], unique=False, if_not_exists=True)
    op.create_index("ix_agent_executions_started_at", "agent_executions", ["started_at"], unique=False, if_not_exists=True)
    op.create_index("ix_agent_executions_failure_reason", "agent_executions", ["failure_reason"], unique=False, if_not_exists=True)

    # Create message_store table (used by LangChain PostgresChatMessageHistory)
    op.create_table(
        "message_store",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("message", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index("ix_message_store_session_id", "message_store", ["session_id"], unique=False, if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_message_store_session_id", table_name="message_store")
    op.drop_table("message_store")
    op.drop_index("ix_agent_executions_failure_reason", table_name="agent_executions")
    op.drop_index("ix_agent_executions_started_at", table_name="agent_executions")
    op.drop_index("ix_agent_executions_status", table_name="agent_executions")
    op.drop_index("ix_agent_executions_session_id", table_name="agent_executions")
    op.drop_table("agent_executions")
