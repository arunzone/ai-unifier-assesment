import logging
import os

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)


def run_migrations():
    """Run database migrations automatically on startup."""
    try:
        alembic_cfg = Config("alembic.ini")

        alembic_cfg.set_main_option(
            "sqlalchemy.url",
            f"postgresql+psycopg://{os.getenv('POSTGRES_USER', 'rag_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'rag_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'rag_evaluation')}",
        )

        logger.info("Running database migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise
