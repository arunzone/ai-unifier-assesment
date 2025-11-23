from unittest.mock import MagicMock, patch

from assertpy import assert_that
from langchain_core.documents import Document

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.rag.qa_service import QAService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


def test_should_create_llm_with_configured_base_url():
    settings = MagicMock(spec=Settings)
    settings.ollama.base_url = "http://ollama:11434"
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = QAService(settings, vector_store_service)

    with patch("ai_unifier_assesment.rag.qa_service.Ollama") as mock_ollama:
        service.get_llm()

        mock_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://ollama:11434",
        )


def test_should_format_docs_with_citations():
    settings = MagicMock(spec=Settings)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = QAService(settings, vector_store_service)
    docs = [
        Document(page_content="First content", metadata={"source": "file1.pdf", "page": 1}),
        Document(page_content="Second content", metadata={"source": "file2.pdf", "page": 5}),
    ]

    result = service.format_docs_with_citations(docs)

    assert_that(result).contains("[Source 1: file1.pdf, Page 1]")
    assert_that(result).contains("First content")
    assert_that(result).contains("[Source 2: file2.pdf, Page 5]")
    assert_that(result).contains("Second content")


def test_should_format_docs_with_unknown_source():
    settings = MagicMock(spec=Settings)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = QAService(settings, vector_store_service)
    docs = [Document(page_content="Content", metadata={})]

    result = service.format_docs_with_citations(docs)

    assert_that(result).contains("[Source 1: Unknown, Page N/A]")


def test_should_get_prompt_with_citation_instructions():
    settings = MagicMock(spec=Settings)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = QAService(settings, vector_store_service)
    prompt = service.get_prompt()

    prompt_str = prompt.format(context="test context", question="test question")
    assert_that(prompt_str).contains("citation")
    assert_that(prompt_str).contains("[Source X]")
    assert_that(prompt_str).contains("test context")
    assert_that(prompt_str).contains("test question")


def test_should_answer_question_and_return_sources():
    settings = MagicMock(spec=Settings)
    settings.ollama.base_url = "http://localhost:11434"
    vector_store_service = MagicMock(spec=VectorStoreService)

    mock_retriever = MagicMock()
    mock_docs = [
        Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1}),
    ]
    mock_retriever.invoke.return_value = mock_docs
    vector_store_service.get_retriever.return_value = mock_retriever

    service = QAService(settings, vector_store_service)

    with patch.object(service, "create_chain") as mock_chain:
        mock_chain_instance = MagicMock()
        mock_chain.return_value = mock_chain_instance
        mock_chain_instance.invoke.return_value = "Test answer [Source 1]"

        result = service.answer("What is the question?")

        assert_that(result["answer"]).is_equal_to("Test answer [Source 1]")
        assert_that(result["sources"]).is_length(1)
        assert_that(result["sources"][0]["source"]).is_equal_to("test.pdf")
        assert_that(result["sources"][0]["page"]).is_equal_to(1)
        assert_that(result).contains_key("retrieval_time_ms")


def test_should_retrieve_only_without_llm():
    settings = MagicMock(spec=Settings)
    vector_store_service = MagicMock(spec=VectorStoreService)

    mock_retriever = MagicMock()
    mock_docs = [
        Document(page_content="Content 1", metadata={"source": "file1.pdf", "page": 1}),
        Document(page_content="Content 2", metadata={"source": "file2.pdf", "page": 2}),
    ]
    mock_retriever.invoke.return_value = mock_docs
    vector_store_service.get_retriever.return_value = mock_retriever

    service = QAService(settings, vector_store_service)
    result = service.retrieve_only("test question", k=5)

    vector_store_service.get_retriever.assert_called_once_with("rag_corpus", 5)
    assert_that(result["documents"]).is_length(2)
    assert_that(result["documents"][0]["content"]).is_equal_to("Content 1")
    assert_that(result["documents"][0]["source"]).is_equal_to("file1.pdf")
    assert_that(result).contains_key("retrieval_time_ms")


def test_should_use_custom_collection_name():
    settings = MagicMock(spec=Settings)
    vector_store_service = MagicMock(spec=VectorStoreService)

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    vector_store_service.get_retriever.return_value = mock_retriever

    service = QAService(settings, vector_store_service)
    service.retrieve_only("test", collection_name="custom_collection")

    vector_store_service.get_retriever.assert_called_once_with("custom_collection", 5)
