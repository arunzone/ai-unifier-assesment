from unittest.mock import MagicMock, patch

from assertpy import assert_that

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.rag.embedding_service import EmbeddingService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


def test_should_create_chroma_client_with_configured_host_and_port():
    settings = MagicMock(spec=Settings)
    settings.chroma.host = "chroma-server"
    settings.chroma.port = 8001
    embedding_service = MagicMock(spec=EmbeddingService)

    service = VectorStoreService(settings, embedding_service)

    with patch("ai_unifier_assesment.rag.vector_store_service.chromadb.HttpClient") as mock_client:
        service.get_client()

        mock_client.assert_called_once_with(
            host="chroma-server",
            port=8001,
        )


def test_should_create_vector_store_with_default_collection_name():
    settings = MagicMock(spec=Settings)
    settings.chroma.host = "localhost"
    settings.chroma.port = 8000
    embedding_service = MagicMock(spec=EmbeddingService)
    mock_embeddings = MagicMock()
    embedding_service.get_embeddings.return_value = mock_embeddings

    service = VectorStoreService(settings, embedding_service)

    with patch("ai_unifier_assesment.rag.vector_store_service.chromadb.HttpClient") as mock_client:
        with patch("ai_unifier_assesment.rag.vector_store_service.Chroma") as mock_chroma:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            service.get_vector_store()

            mock_chroma.assert_called_once_with(
                client=mock_client_instance,
                collection_name="rag_corpus",
                embedding_function=mock_embeddings,
            )


def test_should_create_vector_store_with_custom_collection_name():
    settings = MagicMock(spec=Settings)
    settings.chroma.host = "localhost"
    settings.chroma.port = 8000
    embedding_service = MagicMock(spec=EmbeddingService)
    mock_embeddings = MagicMock()
    embedding_service.get_embeddings.return_value = mock_embeddings

    service = VectorStoreService(settings, embedding_service)

    with patch("ai_unifier_assesment.rag.vector_store_service.chromadb.HttpClient"):
        with patch("ai_unifier_assesment.rag.vector_store_service.Chroma") as mock_chroma:
            service.get_vector_store("custom_collection")

            assert_that(mock_chroma.call_args[1]["collection_name"]).is_equal_to("custom_collection")


def test_should_create_retriever_with_mmr_search():
    settings = MagicMock(spec=Settings)
    settings.chroma.host = "localhost"
    settings.chroma.port = 8000
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embeddings.return_value = MagicMock()

    service = VectorStoreService(settings, embedding_service)

    with patch("ai_unifier_assesment.rag.vector_store_service.chromadb.HttpClient"):
        with patch("ai_unifier_assesment.rag.vector_store_service.Chroma") as mock_chroma:
            mock_vector_store = MagicMock()
            mock_chroma.return_value = mock_vector_store
            mock_retriever = MagicMock()
            mock_vector_store.as_retriever.return_value = mock_retriever

            result = service.get_retriever()

            mock_vector_store.as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20},
            )
            assert_that(result).is_equal_to(mock_retriever)


def test_should_create_retriever_with_custom_k():
    settings = MagicMock(spec=Settings)
    settings.chroma.host = "localhost"
    settings.chroma.port = 8000
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embeddings.return_value = MagicMock()

    service = VectorStoreService(settings, embedding_service)

    with patch("ai_unifier_assesment.rag.vector_store_service.chromadb.HttpClient"):
        with patch("ai_unifier_assesment.rag.vector_store_service.Chroma") as mock_chroma:
            mock_vector_store = MagicMock()
            mock_chroma.return_value = mock_vector_store

            service.get_retriever(k=10)

            assert_that(mock_vector_store.as_retriever.call_args[1]["search_kwargs"]["k"]).is_equal_to(10)
