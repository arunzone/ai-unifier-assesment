from unittest.mock import MagicMock, patch

from assertpy import assert_that

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.rag.embedding_service import EmbeddingService


def test_should_create_embeddings_with_configured_model():
    settings = MagicMock(spec=Settings)
    settings.ollama.embedding_model = "nomic-embed-text"
    settings.ollama.base_url = "http://localhost:11434"

    service = EmbeddingService(settings)

    with patch("ai_unifier_assesment.rag.embedding_service.OllamaEmbeddings") as mock_embeddings:
        service.get_embeddings()

        mock_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


def test_should_use_custom_embedding_model_from_settings():
    settings = MagicMock(spec=Settings)
    settings.ollama.embedding_model = "custom-model"
    settings.ollama.base_url = "http://ollama:11434"

    service = EmbeddingService(settings)

    with patch("ai_unifier_assesment.rag.embedding_service.OllamaEmbeddings") as mock_embeddings:
        service.get_embeddings()

        mock_embeddings.assert_called_once_with(
            model="custom-model",
            base_url="http://ollama:11434",
        )


def test_should_return_ollama_embeddings_instance():
    settings = MagicMock(spec=Settings)
    settings.ollama.embedding_model = "nomic-embed-text"
    settings.ollama.base_url = "http://localhost:11434"

    service = EmbeddingService(settings)

    with patch("ai_unifier_assesment.rag.embedding_service.OllamaEmbeddings") as mock_embeddings:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        result = service.get_embeddings()

        assert_that(result).is_equal_to(mock_instance)
