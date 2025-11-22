import os
from unittest.mock import patch

import pytest
from assertpy import assert_that

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.config import OpenAIConfig


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
