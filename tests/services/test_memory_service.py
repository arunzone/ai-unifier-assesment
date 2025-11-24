from unittest.mock import Mock, patch

from assertpy import assert_that
from langchain_community.chat_message_histories import PostgresChatMessageHistory

from ai_unifier_assesment.config import Settings, PostgresConfig
from ai_unifier_assesment.services.memory_service import MemoryService


def create_mock_settings(window_size: int = 5) -> Settings:
    settings = Mock(spec=Settings)
    settings.memory_window_size = window_size
    settings.postgres = PostgresConfig(
        host="localhost",
        port=5432,
        user="test_user",
        password="test_pass",
        database="test_db",
    )
    return settings


@patch.object(PostgresChatMessageHistory, "__init__", return_value=None)
def test_get_session_history_should_create_postgres_history(mock_init):
    service = MemoryService(create_mock_settings())

    history = service.get_session_history("session-1")

    assert_that(history).is_instance_of(PostgresChatMessageHistory)
    mock_init.assert_called_once_with(
        session_id="session-1",
        connection_string="postgresql://test_user:test_pass@localhost:5432/test_db",
    )


def test_get_trimmer_should_return_trimmer():
    service = MemoryService(create_mock_settings(window_size=10))

    trimmer = service.get_trimmer()

    assert_that(trimmer).is_not_none()
