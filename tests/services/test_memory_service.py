from unittest.mock import Mock

from assertpy import assert_that
from langchain_core.chat_history import InMemoryChatMessageHistory

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.services.memory_service import MemoryService


def create_mock_settings(window_size: int = 5) -> Settings:
    settings = Mock(spec=Settings)
    settings.memory_window_size = window_size
    return settings


def test_get_session_history_should_create_new_history_for_new_session():
    service = MemoryService(create_mock_settings())

    history = service.get_session_history("session-1")

    assert_that(history).is_instance_of(InMemoryChatMessageHistory)


def test_get_session_history_should_return_same_history_for_same_session():
    service = MemoryService(create_mock_settings())

    history1 = service.get_session_history("session-1")
    history2 = service.get_session_history("session-1")

    assert_that(history1).is_same_as(history2)


def test_get_session_history_should_return_different_history_for_different_sessions():
    service = MemoryService(create_mock_settings())

    history1 = service.get_session_history("session-1")
    history2 = service.get_session_history("session-2")

    assert_that(history1).is_not_same_as(history2)


def test_get_trimmer_should_return_trimmer():
    service = MemoryService(create_mock_settings(window_size=10))

    trimmer = service.get_trimmer()

    assert_that(trimmer).is_not_none()
