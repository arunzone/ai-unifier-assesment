from unittest.mock import Mock

import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient

from ai_unifier_assesment.app import app
from ai_unifier_assesment.services.chat_service import ChatService


async def mock_stream_response(message: str, session_id: str):
    yield "data: Hello\n\n"
    yield "data:  World\n\n"
    yield 'event: stats\ndata: {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001, "latency_ms": 100}\n\n'


@pytest.fixture
def client():
    mock_service = Mock(spec=ChatService)
    mock_service.stream_response = mock_stream_response

    app.dependency_overrides[ChatService] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_should_chat_stream_endpoint_exists(client):
    response = client.post("/api/chat/stream", json={"message": "Hello", "session_id": "test-session"})

    assert_that(response.headers["content-type"]).is_equal_to("text/event-stream; charset=utf-8")


def test_should_chat_stream_returns_sse_format(client):
    response = client.post("/api/chat/stream", json={"message": "test", "session_id": "test-session"})

    assert_that(response.text).contains("data: Hello").contains("data:  World")


def test_should_chat_stream_requires_message(client):
    response = client.post("/api/chat/stream", json={"session_id": "test-session"})

    assert_that(response.status_code).is_equal_to(422)


def test_should_chat_stream_requires_session_id(client):
    response = client.post("/api/chat/stream", json={"message": "Hello"})

    assert_that(response.status_code).is_equal_to(422)
