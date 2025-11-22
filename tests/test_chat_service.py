from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
from assertpy import assert_that

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.chat_service import ChatService


@dataclass
class Chunk:
    content: str | None


async def async_iter(items):
    for item in items:
        yield item


def create_mock_chain(contents):
    """Create a mock chain that streams the given contents."""
    chunks = [Chunk(c) for c in contents]
    mock_chain = Mock()
    mock_chain.astream.return_value = async_iter(chunks)
    return mock_chain


@pytest.mark.asyncio
async def test_should_return_stream_response():
    mock_model = Mock(spec=Model)
    mock_chain = create_mock_chain(["Hello", " world", None])
    service = ChatService(mock_model)

    with patch("ai_unifier_assesment.services.chat_service.ChatPromptTemplate") as mock_template:
        mock_prompt = Mock()
        mock_template.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = Mock(return_value=mock_chain)

        chunks = [chunk async for chunk in service.stream_response("test")]

    assert_that(chunks).is_equal_to(["data: Hello\n\n", "data:  world\n\n"])


@pytest.mark.asyncio
async def test_should_call_llm_with_user_message():
    mock_model = Mock(spec=Model)
    mock_chain = create_mock_chain(["Hello", " world", None])
    service = ChatService(mock_model)

    with patch("ai_unifier_assesment.services.chat_service.ChatPromptTemplate") as mock_template:
        mock_prompt = Mock()
        mock_template.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = Mock(return_value=mock_chain)

        [chunk async for chunk in service.stream_response("test")]

        mock_chain.astream.assert_called_once_with({"message": "test"})


@pytest.mark.asyncio
async def test_should_call_llm_with_sytem_anduser_message():
    mock_model = Mock(spec=Model)
    mock_chain = create_mock_chain(["Hello", " world", None])
    service = ChatService(mock_model)

    with patch("ai_unifier_assesment.services.chat_service.ChatPromptTemplate") as mock_template:
        mock_prompt = Mock()
        mock_template.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = Mock(return_value=mock_chain)

        [chunk async for chunk in service.stream_response("test")]

        mock_template.from_messages.assert_called_once_with(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{message}"),
            ]
        )
