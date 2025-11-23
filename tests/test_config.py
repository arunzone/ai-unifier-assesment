import os
from unittest.mock import patch

import pytest
from assertpy import assert_that

from ai_unifier_assesment.config import (
    ChromaConfig,
    EvaluationConfig,
    OllamaConfig,
    OpenAIConfig,
    PostgresConfig,
    PricingConfig,
    RAGConfig,
    Settings,
)


def test_should_load_model_details_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "MODEL_NAME": "gpt-4-turbo",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.openai).is_equal_to(
        OpenAIConfig(base_url="https://api.com", api_key="sk-test", model_name="gpt-4-turbo")
    )


def test_should_use_default_model_name_when_not_set():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.openai.model_name).is_equal_to("Gpt4o")


def test_should_load_fastapi_port_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "FASTAPI_PORT": "9000",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.fastapi.port).is_equal_to(9000)


def test_should_load_fastapi_host_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "FASTAPI_HOST": "127.0.0.1",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.fastapi.host).is_equal_to("127.0.0.1")


def test_should_use_default_fastapi_port_when_not_set():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.fastapi.port).is_equal_to(8000)


def test_should_use_default_fastapi_host_when_not_set():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.fastapi.host).is_equal_to("0.0.0.0")


def test_should_raise_error_when_openai_base_url_missing():
    env_vars = {
        "OPENAI_API_KEY": "sk-test",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(Exception):
            Settings()


def test_should_raise_error_when_openai_api_key_missing():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(Exception):
            Settings()


def test_should_load_pricing_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "PRICING_INPUT_COST_PER_1M": "5.00",
        "PRICING_OUTPUT_COST_PER_1M": "15.00",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.pricing).is_equal_to(PricingConfig(input_cost_per_1m=5.00, output_cost_per_1m=15.00))


def test_should_load_ollama_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_EMBEDDING_MODEL": "custom-embed",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.ollama).is_equal_to(
        OllamaConfig(base_url="http://ollama:11434", embedding_model="custom-embed")
    )


def test_should_use_default_ollama_base_url_when_not_set():
    settings = Settings()

    assert_that(settings.ollama.base_url).is_equal_to("http://localhost:11434")


def test_should_use_default_ollama_embedding_model_when_not_set():
    settings = Settings()

    assert_that(settings.ollama.embedding_model).is_equal_to("nomic-embed-text")


def test_should_load_chroma_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "CHROMA_HOST": "chroma-server",
        "CHROMA_PORT": "8001",
        "CHROMA_COLLECTION_NAME": "custom_collection",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.chroma).is_equal_to(
        ChromaConfig(host="chroma-server", port=8001, collection_name="custom_collection")
    )


def test_should_use_default_chroma_host_when_not_set():
    settings = Settings()

    assert_that(settings.chroma.host).is_equal_to("localhost")


def test_should_use_default_chroma_port_when_not_set():
    settings = Settings()

    assert_that(settings.chroma.port).is_equal_to(8000)


def test_should_use_default_chroma_collection_name_when_not_set():
    settings = Settings()

    assert_that(settings.chroma.collection_name).is_equal_to("rag_corpus")


def test_should_load_rag_config_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "RAG_CHUNK_SIZE": "1000",
        "RAG_CHUNK_OVERLAP": "200",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.rag).is_equal_to(RAGConfig(chunk_size=1000, chunk_overlap=200))


def test_should_use_default_rag_chunk_size_when_not_set():
    settings = Settings()

    assert_that(settings.rag.chunk_size).is_equal_to(500)


def test_should_use_default_rag_chunk_overlap_when_not_set():
    settings = Settings()

    assert_that(settings.rag.chunk_overlap).is_equal_to(100)


def test_should_load_postgres_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "POSTGRES_HOST": "db-server",
        "POSTGRES_PORT": "5433",
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpass",
        "POSTGRES_DB": "testdb",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.postgres).is_equal_to(
        PostgresConfig(
            host="db-server",
            port=5433,
            user="testuser",
            password="testpass",
            database="testdb",
        )
    )


def test_should_use_default_postgres_values_when_not_set():
    settings = Settings()

    assert_that(settings.postgres.host).is_equal_to("localhost")
    assert_that(settings.postgres.port).is_equal_to(5432)
    assert_that(settings.postgres.user).is_equal_to("rag_user")
    assert_that(settings.postgres.database).is_equal_to("rag_evaluation")


def test_should_generate_postgres_connection_string():
    settings = Settings()

    assert_that(settings.postgres.connection_string).is_equal_to(
        "postgresql://rag_user:rag_password@localhost:5432/rag_evaluation"
    )


def test_should_load_evaluation_from_environment():
    env_vars = {
        "OPENAI_BASE_URL": "https://api.com",
        "OPENAI_API_KEY": "sk-test",
        "EVALUATION_TEST_SIZE": "50",
        "EVALUATION_LLM_MODEL": "llama3",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()

    assert_that(settings.evaluation).is_equal_to(EvaluationConfig(test_size=50, llm_model="llama3"))


def test_should_use_default_evaluation_values_when_not_set():
    settings = Settings()

    assert_that(settings.evaluation.test_size).is_equal_to(25)
    assert_that(settings.evaluation.llm_model).is_equal_to("llama3.1:8b-instruct-q4_K_M")
