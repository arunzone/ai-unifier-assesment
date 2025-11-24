"""Self-Healing Code Generation Agent.

This agent implements a self-correcting loop that:
1. Generates code from natural language descriptions
2. Writes code to disk and runs tests
3. Captures errors and iteratively fixes the code
4. Stops after success or 3 attempts
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from langchain_core.messages import HumanMessage, SystemMessage

from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.agent.tools.code_tester_tool import (
    CodeTesterInput,
    CodeTesterTool,
)
from ai_unifier_assesment.agent.tools.code_writer_tool import (
    CodeWriterInput,
    CodeWriterTool,
)
from ai_unifier_assesment.dependencies import get_settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class SelfHealingAgent:
    """Self-healing code generation agent with iterative error correction.

    This agent orchestrates the code generation, testing, and self-healing loop.
    It uses LLM to generate code, writes it to disk, runs tests, and iteratively
    fixes errors until all tests pass or max attempts are reached.
    """

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        prompt_loader: Annotated[PromptLoader, Depends(PromptLoader)],
        code_writer: Annotated[CodeWriterTool, Depends(CodeWriterTool)],
        code_tester: Annotated[CodeTesterTool, Depends(CodeTesterTool)],
        settings: Annotated[object, Depends(get_settings)],
    ):
        """Initialize the self-healing agent.

        Args:
            model: LLM model provider
            prompt_loader: Loads system prompts from disk
            code_writer: Tool for writing code to disk
            code_tester: Tool for running tests
            settings: Application settings
        """
        self._model = model
        self._prompt_loader = prompt_loader
        self._code_writer = code_writer
        self._code_tester = code_tester
        self._settings = settings

    def _setup_initial_state(self, task_description: str, language: str) -> CodeHealingState:
        """Setup initial state and working directory."""
        logger.info(f"Starting self-healing agent for task: {task_description}")
        logger.info(f"Language: {language}")

        # Create temporary working directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"code_healing_{language}_"))
        logger.info(f"Working directory: {temp_dir}")

        # Initialize state
        return CodeHealingState(
            task_description=task_description,
            language=language,
            working_directory=str(temp_dir),
        )

    async def _execute_attempt(self, state: CodeHealingState, attempt: int) -> CodeHealingState:
        """Execute a single healing attempt."""
        state.attempt_number = attempt
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ATTEMPT {attempt + 1} / {self.MAX_ATTEMPTS}")
        logger.info(f"{'=' * 60}")

        if attempt == 0:
            # First attempt: Generate initial code
            state = await self._generate_initial_code(state)
        else:
            # Subsequent attempts: Fix code based on errors
            state = await self._fix_code(state)

        if not state.current_code:
            state.final_message = f"Failed to generate code on attempt {attempt + 1}"
            logger.error(state.final_message)
            return state

        # Write code to disk
        state = self._write_code_to_disk(state)

        # Run tests
        state = self._run_tests(state)

        # Check if tests passed
        if state.success:
            state.final_message = f"Success! All tests passed on attempt {attempt + 1}"
            logger.info(f"✓ {state.final_message}")
        else:
            logger.warning(f"✗ Tests failed on attempt {attempt + 1}")

        return state

    def _finalize_state(self, state: CodeHealingState) -> CodeHealingState:
        """Finalize state after all attempts."""
        if not state.success:
            state.final_message = f"Failed after {self.MAX_ATTEMPTS} attempts. Last error:\n{state.test_output}"
            logger.error(state.final_message)

        logger.info(f"\nFinal working directory: {state.working_directory}")
        return state

    async def heal(self, task_description: str, language: str) -> CodeHealingState:
        """Execute the self-healing code generation loop.

        Args:
            task_description: Natural language description of coding task
            language: Programming language ('python' or 'rust')

        Returns:
            Final state with code, test results, and success status
        """
        state = self._setup_initial_state(task_description, language)

        # Main self-healing loop
        for attempt in range(self.MAX_ATTEMPTS):
            state = await self._execute_attempt(state, attempt)
            if state.success:
                break

        return self._finalize_state(state)

    async def _generate_initial_code(self, state: CodeHealingState) -> CodeHealingState:
        """Generate initial code from task description.

        Args:
            state: Current state

        Returns:
            Updated state with generated code
        """
        logger.info("Generating initial code...")

        # Load system prompt
        system_prompt = self._prompt_loader.load("code_healing_system")

        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Task: {state.task_description}\nLanguage: {state.language}"),
        ]

        # Generate code
        llm = self._model.simple_model()
        response = await llm.ainvoke(messages)

        # Extract code from response
        state.current_code = response.content
        logger.info(f"Generated {len(state.current_code)} characters of code")

        return state

    async def _fix_code(self, state: CodeHealingState) -> CodeHealingState:
        """Fix code based on previous test errors.

        Args:
            state: Current state with previous code and errors

        Returns:
            Updated state with fixed code
        """
        logger.info("Fixing code based on errors...")

        # Load fix prompt template
        fix_prompt_template = self._prompt_loader.load("code_healing_fix")

        # Fill in template
        fix_prompt = fix_prompt_template.format(
            previous_code=state.current_code,
            test_output=state.test_output,
        )

        # Create messages (no system prompt on fixes, just the fix instructions)
        messages = [HumanMessage(content=fix_prompt)]

        # Generate fixed code
        llm = self._model.simple_model()
        response = await llm.ainvoke(messages)

        # Extract code from response
        state.current_code = response.content
        logger.info(f"Generated {len(state.current_code)} characters of fixed code")

        return state

    def _write_code_to_disk(self, state: CodeHealingState) -> CodeHealingState:
        """Parse generated code and write files to disk.

        Args:
            state: Current state with generated code

        Returns:
            Updated state (unchanged, writes are side effects)
        """
        logger.info("Writing code to disk...")

        # Parse the code response to extract files
        files = self._parse_code_files(state.current_code, state.language)

        if not files:
            logger.error("No files found in generated code")
            return state

        # Write each file
        for filename, content in files.items():
            file_path = Path(state.working_directory) / filename
            logger.info(f"Writing {filename} ({len(content)} chars)")

            write_input = CodeWriterInput(
                code=content,
                file_path=str(file_path),
                language=state.language,
            )

            result = self._code_writer.write(write_input)

            if not result.success:
                logger.error(f"Failed to write {filename}: {result.message}")
            else:
                logger.info(f"✓ Wrote {filename}")

        return state

    def _run_tests(self, state: CodeHealingState) -> CodeHealingState:
        """Run tests and capture output.

        Args:
            state: Current state

        Returns:
            Updated state with test results
        """
        logger.info("Running tests...")

        test_input = CodeTesterInput(
            working_directory=state.working_directory,
            language=state.language,
            timeout=30,
        )

        result = self._code_tester.test(test_input)

        # Update state with results
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
        """Parse LLM response to extract file contents.

        Expected format:
        FILE: filename.ext
        ```language
        code content
        ```

        Args:
            code_content: Raw LLM response
            language: Programming language

        Returns:
            Dictionary mapping filenames to their contents
        """
        files = {}

        # Pattern to match FILE: filename followed by code block
        pattern = r"FILE:\s*(\S+)\s*```(?:\w+)?\s*\n(.*?)```"

        matches = re.finditer(pattern, code_content, re.DOTALL)

        for match in matches:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            files[filename] = content
            logger.info(f"Parsed file: {filename} ({len(content)} chars)")

        # If no files found with FILE: marker, try to extract code blocks directly
        if not files:
            logger.warning("No FILE: markers found, attempting to extract code blocks")
            files = self._fallback_parse(code_content, language)

        return files

    @staticmethod
    def _parse_python_code_blocks(code_blocks: list[str]) -> dict[str, str]:
        """Parse Python code blocks into test and main files."""
        files = {}
        for i, block in enumerate(code_blocks):
            if "import pytest" in block or "def test_" in block:
                files["test_main.py"] = block.strip()
            else:
                files["main.py"] = block.strip()
        return files

    @staticmethod
    def _parse_rust_code_blocks(code_blocks: list[str]) -> dict[str, str]:
        """Parse Rust code blocks into lib.rs."""
        files = {}
        if code_blocks:
            files["lib.rs"] = code_blocks[0].strip()
        return files

    def _fallback_parse(self, code_content: str, language: str) -> dict[str, str]:
        """Fallback parser when FILE: markers are missing.

        Args:
            code_content: Raw LLM response
            language: Programming language

        Returns:
            Dictionary with default filenames
        """
        # Extract all code blocks
        code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", code_content, re.DOTALL)

        if language == "python":
            return self._parse_python_code_blocks(code_blocks)
        else:  # rust
            return self._parse_rust_code_blocks(code_blocks)

    def _format_test_output(self, stdout: str, stderr: str) -> str:
        """Format test output for error feedback.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            Formatted error message
        """
        output_parts = []

        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        if stdout:
            output_parts.append(f"STDOUT:\n{stdout}")

        return "\n\n".join(output_parts) if output_parts else "No output captured"
