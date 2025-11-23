from unittest.mock import MagicMock, patch

from assertpy import assert_that
from langchain_core.documents import Document

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.evaluation.testset_generator_service import TestsetGeneratorService


def test_should_create_llm_with_configured_model():
    settings = MagicMock(spec=Settings)
    settings.evaluation_llm_model = "llama3.1:8b-instruct-q4_K_M"
    settings.ollama.base_url = "http://ollama:11434"

    with patch("ai_unifier_assesment.evaluation.testset_generator_service.Model"):
        service = TestsetGeneratorService(settings)

        with patch("ai_unifier_assesment.evaluation.testset_generator_service.ChatOllama") as mock_chat_ollama:
            service.get_raw_llm()

            mock_chat_ollama.assert_called_once_with(
                model="llama3.1:8b-instruct-q4_K_M",
                base_url="http://ollama:11434",
                format="json",
            )


def test_should_create_embeddings_with_configured_model():
    settings = MagicMock(spec=Settings)
    settings.ollama.embedding_model = "nomic-embed-text"
    settings.ollama.base_url = "http://ollama:11434"

    with patch("ai_unifier_assesment.evaluation.testset_generator_service.Model"):
        service = TestsetGeneratorService(settings)

        with patch("ai_unifier_assesment.evaluation.testset_generator_service.OllamaEmbeddings") as mock_embeddings:
            service.get_raw_embeddings()

            mock_embeddings.assert_called_once_with(
                model="nomic-embed-text",
                base_url="http://ollama:11434",
            )


def test_should_use_default_test_size_from_settings():
    settings = MagicMock(spec=Settings)
    settings.evaluation.test_size = 25
    settings.evaluation_llm_model = "llama3.1:8b-instruct-q4_K_M"
    settings.ollama.base_url = "http://localhost:11434"
    settings.ollama.embedding_model = "nomic-embed-text"

    with patch("ai_unifier_assesment.evaluation.testset_generator_service.Model"):
        service = TestsetGeneratorService(settings)
        documents = [Document(page_content="Test content", metadata={"filename": "test.pdf"})]

        with patch.object(service, "build_knowledge_graph") as mock_build_kg:
            mock_kg = MagicMock()
            mock_build_kg.return_value = mock_kg

            with patch("ai_unifier_assesment.evaluation.testset_generator_service.TestsetGenerator") as mock_gen_class:
                with patch("ai_unifier_assesment.evaluation.testset_generator_service.LangchainLLMWrapper"):
                    with patch("ai_unifier_assesment.evaluation.testset_generator_service.embedding_factory"):
                        mock_generator = MagicMock()
                        mock_gen_class.return_value = mock_generator

                        mock_testset = MagicMock()
                        mock_df = MagicMock()
                        mock_df.iterrows.return_value = []
                        mock_testset.to_pandas.return_value = mock_df
                        mock_generator.generate.return_value = mock_testset

                        service.generate(documents)

                        mock_generator.generate.assert_called_once()
                        call_args = mock_generator.generate.call_args
                        assert_that(call_args[1]["testset_size"]).is_equal_to(25)


def test_should_use_custom_test_size():
    settings = MagicMock(spec=Settings)
    settings.evaluation.test_size = 25
    settings.evaluation_llm_model = "llama3.1:8b-instruct-q4_K_M"
    settings.ollama.base_url = "http://localhost:11434"
    settings.ollama.embedding_model = "nomic-embed-text"

    with patch("ai_unifier_assesment.evaluation.testset_generator_service.Model"):
        service = TestsetGeneratorService(settings)
        documents = [Document(page_content="Test content", metadata={"filename": "test.pdf"})]

        with patch.object(service, "build_knowledge_graph") as mock_build_kg:
            mock_kg = MagicMock()
            mock_build_kg.return_value = mock_kg

            with patch("ai_unifier_assesment.evaluation.testset_generator_service.TestsetGenerator") as mock_gen_class:
                with patch("ai_unifier_assesment.evaluation.testset_generator_service.LangchainLLMWrapper"):
                    with patch("ai_unifier_assesment.evaluation.testset_generator_service.embedding_factory"):
                        mock_generator = MagicMock()
                        mock_gen_class.return_value = mock_generator

                        mock_testset = MagicMock()
                        mock_df = MagicMock()
                        mock_df.iterrows.return_value = []
                        mock_testset.to_pandas.return_value = mock_df
                        mock_generator.generate.return_value = mock_testset

                        service.generate(documents, test_size=50)

                        call_args = mock_generator.generate.call_args
                        assert_that(call_args[1]["testset_size"]).is_equal_to(50)
