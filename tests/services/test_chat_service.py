from dataclasses import dataclass
from unittest.mock import Mock, patch

from ai_unifier_assesment.services.memory_service import MemoryService
import pytest
from assertpy import assert_that

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.chat_service import ChatService
from ai_unifier_assesment.services.stream_metrics import StreamMetrics, TokenCounter
from ai_unifier_assesment.repositories.metrics_repository import MetricsRepository


@dataclass
class Chunk:
    content: str | None


async def async_iter(items):
    for item in items:
        yield item


def create_service_with_mocks():
    """Create ChatService with all required mocks."""
    mock_model = Mock(spec=Model)
    mock_llm = Mock()
    mock_model.stream_model.return_value = mock_llm

    mock_metrics = Mock(spec=StreamMetrics)
    mock_metrics.build_stats.return_value = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost": 0.00075,
        "latency_ms": 500,
    }

    mock_memory_service = Mock(spec=MemoryService)
    mock_trimmer = Mock()
    mock_memory_service.get_trimmer.return_value = mock_trimmer

    mock_history = Mock()
    mock_history.messages = []
    mock_memory_service.get_session_history.return_value = mock_history

    mock_token_counter = Mock(spec=TokenCounter)
    mock_token_counter.count_message_tokens.return_value = 100
    mock_token_counter.count_text_tokens.return_value = 50

    mock_metrics_repo = Mock(spec=MetricsRepository)

    service = ChatService(mock_model, mock_metrics, mock_memory_service, mock_token_counter, mock_metrics_repo)
    return service, mock_model, mock_metrics, mock_memory_service, mock_token_counter


@pytest.mark.asyncio
async def test_should_return_stream_response():
    service, _, _, _, _ = create_service_with_mocks()
    chunks = [Chunk("Hello"), Chunk(" world"), Chunk(None)]

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_chain = Mock()
        mock_chain.astream.return_value = async_iter(chunks)
        mock_rwmh.return_value = mock_chain

        result = [chunk async for chunk in service.stream_response("test", "session_id")]

    assert_that(result).is_equal_to(
        [
            "data: Hello\n\n",
            "data:  world\n\n",
            'event: stats\ndata: {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.00075, "latency_ms": 500}\n\n',
        ]
    )


@pytest.mark.asyncio
async def test_should_call_chain_with_message_and_session_config():
    service, _, _, _, _ = create_service_with_mocks()
    chunks = [Chunk("Hello")]

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_chain = Mock()
        mock_chain.astream.return_value = async_iter(chunks)
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_chain.astream.assert_called_once_with(
            {"message": "test"}, config={"configurable": {"session_id": "session_id"}}
        )


@pytest.mark.asyncio
async def test_should_count_tokens_using_token_counter():
    service, _, _, _, mock_token_counter = create_service_with_mocks()
    chunks = [Chunk("Hello")]

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_chain = Mock()
        mock_chain.astream.return_value = async_iter(chunks)
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_token_counter.count_message_tokens.assert_called_once()
        mock_token_counter.count_text_tokens.assert_called_once_with("Hello")


@pytest.mark.asyncio
async def test_should_build_stats_at_end():
    service, _, mock_metrics, _, _ = create_service_with_mocks()
    chunks = [Chunk("Hello")]

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_chain = Mock()
        mock_chain.astream.return_value = async_iter(chunks)
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_metrics.build_stats.assert_called_once()
