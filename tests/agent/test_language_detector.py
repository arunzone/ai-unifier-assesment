import pytest
from assertpy import assert_that
from pytest_httpx import HTTPXMock
from httpx import Request, Response

from ai_unifier_assesment.agent.language import Language
from ai_unifier_assesment.agent.language_detector import LanguageDetector
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader
from llm_helper import ai_response_for, extract_user_content


PYTHON_RESPONSE = ai_response_for('{"language": "python"}')
RUST_RESPONSE = ai_response_for('{"language": "rust"}')
RUST_RESPONSE = ai_response_for('{"language": "rust"}')


def llm_response(request: Request):
    request_content = request.read().decode("utf-8")
    print(f"Request content: {request_content}")
    user_content = extract_user_content(request_content)

    if "Python function" in user_content:
        return Response(status_code=200, json=PYTHON_RESPONSE)
    elif "Rust function" in user_content:
        return Response(status_code=200, json=RUST_RESPONSE)
    return None


@pytest.mark.asyncio
async def test_should_detect_python_for_explicit_python_task(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(llm_response)
    state = CodeHealingState(
        task_description="Write a Python function to sort an array using quicksort",
    )
    settings = Settings()
    detector = LanguageDetector(model=Model(settings), prompt_loader=PromptLoader(), settings=settings)

    result = await detector.detect_language(state)

    assert_that(result).has_language(Language.PYTHON)


@pytest.mark.asyncio
async def test_should_detect_rust_for_explicit_rust_task(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(llm_response)
    state = CodeHealingState(
        task_description="Write a Rust function to sort an array using quicksort",
    )
    settings = Settings()
    detector = LanguageDetector(model=Model(settings), prompt_loader=PromptLoader(), settings=settings)

    result = await detector.detect_language(state)

    assert_that(result).has_language(Language.RUST)
