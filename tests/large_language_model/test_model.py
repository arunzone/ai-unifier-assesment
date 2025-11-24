from unittest.mock import Mock, patch

from assertpy import assert_that

from ai_unifier_assesment.config import OpenAIConfig, Settings
from ai_unifier_assesment.large_language_model.model import Model


def create_mock_settings() -> Settings:
    settings = Mock(spec=Settings)
    settings.openai = OpenAIConfig(
        base_url="https://api.example.com",
        api_key="sk-test-key",
        model_name="gpt-4o",
    )
    return settings


def test_model_should_store_settings():
    settings = create_mock_settings()

    model = Model(settings)

    assert_that(model._settings).is_same_as(settings)


def test_get_chat_model_should_return_chat_openai():
    settings = create_mock_settings()
    model = Model(settings)

    with patch("ai_unifier_assesment.large_language_model.model.ChatOpenAI") as mock_chat:
        mock_chat.return_value = Mock()

        result = model.get_chat_model()

        mock_chat.assert_called_once_with(
            base_url="https://api.example.com",
            api_key="sk-test-key",
            model="gpt-4o",
            streaming=True,
            stream_usage=True,
            model_kwargs={"stream_options": {"include_usage": True}},
        )
        assert_that(result).is_not_none()
