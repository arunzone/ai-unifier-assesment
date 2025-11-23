from unittest.mock import MagicMock

from assertpy import assert_that
from langchain_core.documents import Document

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.rag.document_loader_service import DocumentLoaderService
from ai_unifier_assesment.rag.ingestion_service import IngestionService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


def test_should_ingest_pdf_and_return_chunk_count():
    settings = MagicMock(spec=Settings)
    document_loader = MagicMock(spec=DocumentLoaderService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    chunks = [
        Document(page_content="chunk1", metadata={}),
        Document(page_content="chunk2", metadata={}),
    ]
    document_loader.load_and_split.return_value = chunks

    mock_vector_store = MagicMock()
    vector_store_service.get_vector_store.return_value = mock_vector_store

    service = IngestionService(settings, document_loader, vector_store_service)
    result = service.ingest_pdf("test.pdf", "test_collection")

    document_loader.load_and_split.assert_called_once_with("test.pdf")
    vector_store_service.get_vector_store.assert_called_once_with("test_collection")
    mock_vector_store.add_documents.assert_called_once_with(chunks)
    assert_that(result).is_equal_to(2)


def test_should_ingest_directory_and_return_chunk_count():
    settings = MagicMock(spec=Settings)
    document_loader = MagicMock(spec=DocumentLoaderService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    chunks = [Document(page_content=f"chunk{i}", metadata={}) for i in range(5)]
    document_loader.load_and_split_directory.return_value = chunks

    mock_vector_store = MagicMock()
    vector_store_service.get_vector_store.return_value = mock_vector_store

    service = IngestionService(settings, document_loader, vector_store_service)
    result = service.ingest_directory("/test/dir", "test_collection")

    document_loader.load_and_split_directory.assert_called_once_with("/test/dir")
    assert_that(result).is_equal_to(5)


def test_should_return_zero_for_empty_chunks():
    settings = MagicMock(spec=Settings)
    document_loader = MagicMock(spec=DocumentLoaderService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    document_loader.load_and_split.return_value = []

    service = IngestionService(settings, document_loader, vector_store_service)
    result = service.ingest_pdf("empty.pdf", "test_collection")

    assert_that(result).is_equal_to(0)
    vector_store_service.get_vector_store.assert_not_called()


def test_should_get_collection_stats():
    settings = MagicMock(spec=Settings)
    document_loader = MagicMock(spec=DocumentLoaderService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_client.get_collection.return_value = mock_collection
    vector_store_service.get_client.return_value = mock_client

    service = IngestionService(settings, document_loader, vector_store_service)
    result = service.get_collection_stats("test_collection")

    assert_that(result["collection_name"]).is_equal_to("test_collection")
    assert_that(result["document_count"]).is_equal_to(100)


def test_should_return_zero_count_when_collection_not_found():
    settings = MagicMock(spec=Settings)
    document_loader = MagicMock(spec=DocumentLoaderService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")
    vector_store_service.get_client.return_value = mock_client

    service = IngestionService(settings, document_loader, vector_store_service)
    result = service.get_collection_stats("missing_collection")

    assert_that(result["collection_name"]).is_equal_to("missing_collection")
    assert_that(result["document_count"]).is_equal_to(0)
