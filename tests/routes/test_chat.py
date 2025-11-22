from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient

from ai_unifier_assesment.app import app
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.chat_service import ChatService


@dataclass
class Chunk:
    content: str | None


async def async_iter(items):
    for item in items:
        yield item


@pytest.fixture
def mock_chat_service():
    mock_model = Mock(spec=Model)
    mock_chain = Mock()
    mock_chain.astream.return_value = async_iter([Chunk("Hello"), Chunk(" World")])

    with patch("ai_unifier_assesment.services.chat_service.ChatPromptTemplate") as mock_template:
        mock_prompt = Mock()
        mock_template.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        yield ChatService(mock_model)


@pytest.fixture
def client(mock_chat_service):
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_should_chat_stream_endpoint_exists(client):
    response = client.post("/api/chat/stream", json={"message": "Hello"})

    assert_that(response.headers["content-type"]).is_equal_to("text/event-stream; charset=utf-8")


def test_should_chat_stream_returns_sse_format(client):
    response = client.post("/api/chat/stream", json={"message": "test"})

    assert_that(response.text).contains("data: Hello").contains("data:  World")


def test_should_chat_stream_requires_message(client):
    response = client.post("/api/chat/stream", json={})

    assert_that(response.status_code).is_equal_to(422)
