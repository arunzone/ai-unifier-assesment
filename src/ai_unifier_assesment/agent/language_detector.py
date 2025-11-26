import logging
from typing import Annotated, Any, Literal

from fastapi import Depends
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ai_unifier_assesment.agent.language import Language
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.dependencies import get_settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class DetectedLanguage(BaseModel):
    language: Literal["python", "rust"] = Field(description="Detected programming language(python or rust)")


class LanguageDetector:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        prompt_loader: Annotated[PromptLoader, Depends(PromptLoader)],
        settings: Annotated[object, Depends(get_settings)],
    ):
        self._model: Runnable[Any, Any] = model.simple_model().with_structured_output(DetectedLanguage)
        self._prompt_loader = prompt_loader
        self._settings = settings

    async def detect_language(self, state: CodeHealingState) -> dict[str, Language]:
        logger.info("--- NODE: Detecting programming language ---")

        language_prompt = self._prompt_loader.load("language_detection")

        messages = [
            SystemMessage(content=language_prompt),
            HumanMessage(content=f"Task: {state.task_description}"),
        ]

        response = await self._model.ainvoke(messages)

        detected_language = Language(response.language)

        logger.info(f"✓ Language detected by LLM: {detected_language.value}")

        return {"language": detected_language}
