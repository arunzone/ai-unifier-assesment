import logging
from typing import Annotated

from fastapi import Depends
from langchain_core.messages import HumanMessage, SystemMessage

from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.dependencies import get_settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class InitialCodeGenerator:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        prompt_loader: Annotated[PromptLoader, Depends(PromptLoader)],
        settings: Annotated[object, Depends(get_settings)],
    ):
        self._model = model
        self._prompt_loader = prompt_loader
        self._settings = settings

    async def generate_initial_code(self, state: CodeHealingState) -> CodeHealingState:
        logger.info("Generating initial code...")

        system_prompt = self._prompt_loader.load("code_healing_system")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Task: {state.task_description}\nLanguage: {state.language.value}"),
        ]

        llm = self._model.simple_model()
        response = await llm.ainvoke(messages)

        state.current_code = response.content
        logger.info(f"Generated {len(state.current_code)} characters of code")

        return state
