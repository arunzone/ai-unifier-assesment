import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated, AsyncGenerator, Dict, Literal

from fastapi import Depends
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ai_unifier_assesment.agent.code_healing_event_processor import CodeHealingEventProcessor
from ai_unifier_assesment.agent.language_detector import LanguageDetector
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.agent.tools.code_tester_tool import CodeTesterTool
from ai_unifier_assesment.agent.tools.code_writer_tool import (
    CodeWriterInput,
    CodeWriterTool,
)
from ai_unifier_assesment.agent.tools.tester_models import CodeTesterInput
from ai_unifier_assesment.dependencies import get_settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader
from ai_unifier_assesment.agent.language import Language

logger = logging.getLogger(__name__)


class CodingAgent:
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        prompt_loader: Annotated[PromptLoader, Depends(PromptLoader)],
        code_writer: Annotated[CodeWriterTool, Depends(CodeWriterTool)],
        code_tester: Annotated[CodeTesterTool, Depends(CodeTesterTool)],
        event_processor: Annotated[CodeHealingEventProcessor, Depends(CodeHealingEventProcessor)],
        language_detector: Annotated[LanguageDetector, Depends(LanguageDetector)],
        settings: Annotated[object, Depends(get_settings)],
    ):
        self._model = model
        self._prompt_loader = prompt_loader
        self._code_writer = code_writer
        self._code_tester = code_tester
        self._event_processor = event_processor
        self._language_detector = language_detector
        self._settings = settings

    async def _detect_language_node(self, state: CodeHealingState) -> dict[str, Language]:
        response: Dict[str, Language] = await self._language_detector.detect_language(state)
        return response

    def _setup_working_directory_node(self, state: CodeHealingState) -> dict[str, str]:
        logger.info("--- NODE: Setting up working directory ---")

        workspace_root = Path("/app")
        temp_base = workspace_root / ".code_healing_temp"
        temp_base.mkdir(exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix=f"code_healing_{state.language.value}_", dir=temp_base))
        logger.info(f"Working directory: {temp_dir}")

        return {"working_directory": str(temp_dir)}

    async def _code_generator_router_node(self, state: CodeHealingState) -> dict[str, str]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ATTEMPT {state.attempt_number + 1} / {self.MAX_ATTEMPTS}")
        logger.info(f"{'=' * 60}")

        if state.attempt_number == 0:
            updated_state = await self._generate_initial_code(state)
        else:
            updated_state = await self._fix_code(state)

        if not updated_state.current_code:
            updated_state.final_message = f"Failed to generate code on attempt {state.attempt_number + 1}"
            logger.error(updated_state.final_message)

        return {"current_code": updated_state.current_code, "final_message": updated_state.final_message}

    def _write_code_node(self, state: CodeHealingState) -> dict:
        logger.info("--- NODE: Writing code to disk ---")
        self._write_code_to_disk(state)
        return {}  # No state updates needed, _write_code_to_disk modifies state in place

    def _run_tests_node(self, state: CodeHealingState) -> dict:
        logger.info("--- NODE: Running tests ---")
        updated_state = self._run_tests(state)

        if updated_state.success:
            updated_state.final_message = f"Success! All tests passed on attempt {state.attempt_number + 1}"
            logger.info(f"✓ {updated_state.final_message}")
        else:
            logger.warning(f"✗ Tests failed on attempt {state.attempt_number + 1}")

        return {
            "success": updated_state.success,
            "test_output": updated_state.test_output,
            "final_message": updated_state.final_message,
        }

    def _decide_next_step(self, state: CodeHealingState) -> Literal["retry", "success", "failure"]:
        if state.success:
            logger.info("--- DECISION: Tests passed! Ending with SUCCESS ---")
            return "success"
        elif state.attempt_number < self.MAX_ATTEMPTS - 1:
            logger.warning(
                f"--- DECISION: Tests failed. Retrying (attempt {state.attempt_number + 2}/{self.MAX_ATTEMPTS}) ---"
            )
            return "retry"
        else:
            logger.error(f"--- DECISION: Max attempts ({self.MAX_ATTEMPTS}) reached. Ending with FAILURE ---")
            return "failure"

    def _increment_attempt_node(self, state: CodeHealingState) -> dict:
        new_attempt = state.attempt_number + 1
        logger.info(f"--- NODE: Incrementing attempt counter to {new_attempt} ---")
        return {"attempt_number": new_attempt}

    def _finalize_node(self, state: CodeHealingState) -> dict:
        logger.info("--- NODE: Finalizing state ---")
        if not state.success:
            final_message = f"Failed after {self.MAX_ATTEMPTS} attempts. Last error:\n{state.test_output}"
            logger.error(final_message)
        else:
            final_message = state.final_message

        logger.info(f"\nFinal working directory: {state.working_directory}")
        return {
            "final_message": final_message,
            "final_code": state.current_code,
            "working_directory": state.working_directory,
            "attempts": state.attempt_number + 1,
        }

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(CodeHealingState)

        # Add nodes
        graph.add_node("detect_language", self._detect_language_node)
        graph.add_node("setup_workdir", self._setup_working_directory_node)
        graph.add_node("code_generator", self._code_generator_router_node)
        graph.add_node("write_code", self._write_code_node)
        graph.add_node("run_tests", self._run_tests_node)
        graph.add_node("increment_attempt", self._increment_attempt_node)
        graph.add_node("finalize", self._finalize_node)

        # Set entry point - start with language detection
        graph.set_entry_point("detect_language")

        # Define edges
        graph.add_edge("detect_language", "setup_workdir")
        graph.add_edge("setup_workdir", "code_generator")
        graph.add_edge("code_generator", "write_code")
        graph.add_edge("write_code", "run_tests")

        # Conditional edge after tests
        graph.add_conditional_edges(
            "run_tests",
            self._decide_next_step,
            {
                "retry": "increment_attempt",
                "success": "finalize",
                "failure": "finalize",
            },
        )

        # Loop back to code generator after incrementing attempt
        graph.add_edge("increment_attempt", "code_generator")

        # End after finalization
        graph.add_edge("finalize", END)

        return graph

    async def code_stream(self, task_description: str) -> AsyncGenerator[str, None]:
        graph = self._build_graph().compile()
        initial_state = CodeHealingState(task_description=task_description)

        logger.info("Starting LangGraph streaming execution...")

        async for event in graph.astream(initial_state, stream_mode=["updates"]):
            async for sse_chunk in self._event_processor.process_graph_event(event):
                yield sse_chunk

    async def _generate_initial_code(self, state: CodeHealingState) -> CodeHealingState:
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

    async def _fix_code(self, state: CodeHealingState) -> CodeHealingState:
        logger.info("Fixing code based on errors...")

        fix_prompt_template = self._prompt_loader.load("code_healing_fix")

        fix_prompt = fix_prompt_template.format(
            previous_code=state.current_code,
            test_output=state.test_output,
        )

        messages = [HumanMessage(content=fix_prompt)]

        llm = self._model.simple_model()
        response = await llm.ainvoke(messages)

        state.current_code = response.content
        logger.info(f"Generated {len(state.current_code)} characters of fixed code")

        return state

    def _write_code_to_disk(self, state: CodeHealingState) -> CodeHealingState:
        logger.info("Writing code to disk...")

        files = self._parse_code_files(state.current_code, state.language.value)

        if not files:
            logger.error("No files found in generated code")
            return state

        for filename, content in files.items():
            file_path = Path(state.working_directory) / filename
            logger.info(f"Writing {filename} ({len(content)} chars)")

            write_input = CodeWriterInput(
                code=content,
                file_path=str(file_path),
                language=state.language.value,
            )

            result = self._code_writer.write(write_input)

            if not result.success:
                logger.error(f"Failed to write {filename}: {result.message}")
            else:
                logger.info(f"✓ Wrote {filename}")

        return state

    def _run_tests(self, state: CodeHealingState) -> CodeHealingState:
        logger.info("Running tests...")

        test_input = CodeTesterInput(
            working_directory=state.working_directory,
            language=state.language.value,
            timeout=30,
        )

        result = self._code_tester.test(test_input)

        state.success = result.success
        state.test_output = self._format_test_output(result.stdout, result.stderr)

        if result.success:
            logger.info("✓ All tests passed!")
        else:
            logger.warning("✗ Tests failed")
            logger.info(f"Exit code: {result.exit_code}")
            if result.stderr:
                logger.info(f"Errors:\n{result.stderr[:500]}")

        return state

    def _parse_code_files(self, code_content: str, language: str) -> dict[str, str]:
        files = {}

        pattern = r"FILE:\s*(\S+)\s*```(?:\w+)?\s*\n(.*?)```"

        matches = re.finditer(pattern, code_content, re.DOTALL)

        for match in matches:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            files[filename] = content
            logger.info(f"Parsed file: {filename} ({len(content)} chars)")

        if not files:
            logger.warning("No FILE: markers found, attempting to extract code blocks")
            files = self._fallback_parse(code_content, language)

        return files

    @staticmethod
    def _parse_python_code_blocks(code_blocks: list[str]) -> dict[str, str]:
        files = {}
        for block in code_blocks:
            if "import pytest" in block or "def test_" in block:
                files["test_main.py"] = block.strip()
            else:
                files["main.py"] = block.strip()
        return files

    @staticmethod
    def _parse_rust_code_blocks(code_blocks: list[str]) -> dict[str, str]:
        files = {}
        if code_blocks:
            files["lib.rs"] = code_blocks[0].strip()
        return files

    def _fallback_parse(self, code_content: str, language: str) -> dict[str, str]:
        code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", code_content, re.DOTALL)

        if language == "python":
            return self._parse_python_code_blocks(code_blocks)
        else:
            return self._parse_rust_code_blocks(code_blocks)

    def _format_test_output(self, stdout: str, stderr: str) -> str:
        output_parts = []

        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        if stdout:
            output_parts.append(f"STDOUT:\n{stdout}")

        return "\n\n".join(output_parts) if output_parts else "No output captured"
