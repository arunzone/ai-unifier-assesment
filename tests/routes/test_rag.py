from unittest.mock import MagicMock

import pytest
from assertpy import assert_that
from httpx import ASGITransport, AsyncClient

from ai_unifier_assesment.app import app
from ai_unifier_assesment.rag.qa_service import QAService


@pytest.fixture
def mock_qa_service():
    return MagicMock(spec=QAService)


@pytest.fixture
def override_qa_service(mock_qa_service):
    app.dependency_overrides[QAService] = lambda: mock_qa_service
    yield mock_qa_service
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_should_return_answer_with_citations(override_qa_service):
    override_qa_service.answer.return_value = {
        "answer": "The answer is 42 [Source 1]",
        "sources": [{"source": "test.pdf", "page": 1}],
        "retrieval_time_ms": 150.5,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/rag/qa", json={"question": "What is the answer?"})

    assert_that(response.status_code).is_equal_to(200)
    data = response.json()
    assert_that(data["answer"]).is_equal_to("The answer is 42 [Source 1]")
    assert_that(data["sources"]).is_length(1)
    assert_that(data["sources"][0]["source"]).is_equal_to("test.pdf")
    assert_that(data["retrieval_time_ms"]).is_equal_to(150.5)


@pytest.mark.asyncio
async def test_should_use_custom_collection_name(override_qa_service):
    override_qa_service.answer.return_value = {
        "answer": "Answer",
        "sources": [],
        "retrieval_time_ms": 100.0,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/rag/qa", json={"question": "Test?", "collection_name": "custom_collection"})

    override_qa_service.answer.assert_called_once_with("Test?", "custom_collection")


@pytest.mark.asyncio
async def test_should_retrieve_documents_without_answer(override_qa_service):
    override_qa_service.retrieve_only.return_value = {
        "documents": [
            {"content": "Document content", "source": "file.pdf", "page": 5},
        ],
        "retrieval_time_ms": 50.0,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/rag/retrieve", json={"question": "Find documents"})

    assert_that(response.status_code).is_equal_to(200)
    data = response.json()
    assert_that(data["documents"]).is_length(1)
    assert_that(data["documents"][0]["content"]).is_equal_to("Document content")
    assert_that(data["retrieval_time_ms"]).is_equal_to(50.0)


@pytest.mark.asyncio
async def test_should_return_422_for_missing_question(override_qa_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/rag/qa", json={})

    assert_that(response.status_code).is_equal_to(422)
