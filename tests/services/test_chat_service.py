from dataclasses import dataclass
from unittest.mock import Mock, patch

from ai_unifier_assesment.services.memory_service import MemoryService
import pytest
from assertpy import assert_that

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.chat_service import ChatService
from ai_unifier_assesment.services.stream_metrics import StreamMetrics


@dataclass
class Chunk:
    content: str | None


async def async_iter(items):
    for item in items:
        yield item


def create_mock_chain_with_history(contents):
    """Create a mock chain with history that streams the given contents."""
    chunks = [Chunk(c) for c in contents]
    mock_chain = Mock()
    mock_chain.astream.return_value = async_iter(chunks)
    return mock_chain


def create_service_with_mocks(mock_chain):
    """Create ChatService with all required mocks."""
    mock_model = Mock(spec=Model)
    mock_model.get_chat_model.return_value = Mock()

    mock_metrics = Mock(spec=StreamMetrics)
    mock_metrics.extract_tokens.return_value = (100, 50)
    mock_metrics.build_stats.return_value = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost": 0.00075,
        "latency_ms": 500,
    }

    mock_memory_service = Mock(spec=MemoryService)
    mock_memory_service.get_trimmer.return_value = Mock()
    mock_memory_service.get_session_history.return_value = Mock()

    service = ChatService(mock_model, mock_metrics, mock_memory_service)
    return service, mock_model, mock_metrics, mock_memory_service


@pytest.mark.asyncio
async def test_should_return_stream_response():
    mock_chain = create_mock_chain_with_history(["Hello", " world", None])
    service, _, _, _ = create_service_with_mocks(mock_chain)

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_rwmh.return_value = mock_chain

        chunks = [chunk async for chunk in service.stream_response("test", "session_id")]

    assert_that(chunks).is_equal_to(
        [
            "data: Hello\n\n",
            "data:  world\n\n",
            'event: stats\ndata: {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.00075, "latency_ms": 500}\n\n',
        ]
    )


@pytest.mark.asyncio
async def test_should_call_chain_with_message_and_session_config():
    mock_chain = create_mock_chain_with_history(["Hello"])
    service, _, _, _ = create_service_with_mocks(mock_chain)

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_chain.astream.assert_called_once_with(
            {"message": "test"}, config={"configurable": {"session_id": "session_id"}}
        )


@pytest.mark.asyncio
async def test_should_extract_tokens_from_chunks():
    mock_chain = create_mock_chain_with_history(["Hello"])
    service, _, mock_metrics, _ = create_service_with_mocks(mock_chain)

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_metrics.extract_tokens.assert_called()


@pytest.mark.asyncio
async def test_should_build_stats_at_end():
    mock_chain = create_mock_chain_with_history(["Hello"])
    service, _, mock_metrics, _ = create_service_with_mocks(mock_chain)

    with patch("ai_unifier_assesment.services.chat_service.RunnableWithMessageHistory") as mock_rwmh:
        mock_rwmh.return_value = mock_chain

        [chunk async for chunk in service.stream_response("test", "session_id")]

        mock_metrics.build_stats.assert_called_once()
